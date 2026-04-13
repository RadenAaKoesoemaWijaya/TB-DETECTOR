"""
TB Detector v3 - Fully Integrated Pipeline
Complete workflow: ZIP Upload → Preprocessing → Training → Visualization → Save
"""

import os
import io
import zipfile
import shutil
import json
import tempfile
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import threading
import queue

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import soundfile as sf
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from app.models.backbones import BackboneFactory, get_backbone_dim
from app.models.classifier import TBClassifier, get_tb_loss_fn
from app.models.preprocessing import AudioPreprocessor, CoughSegmenter, DataAugmentation
from app.utils.feature_fusion import SimpleConcatFusion
from app.utils.metadata_encoder import SimpleMetadataEncoder
from app.model_manager import get_model_manager

# Setup
app = FastAPI(
    title="TB Detector v3 - Integrated Pipeline",
    description="Complete TB Detection Pipeline: Upload → Preprocess → Train → Visualize → Save",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 5 * SAMPLE_RATE

# Folder Validation - Auto-create required directories
def ensure_directories():
    """Ensure all required directories exist"""
    required_dirs = [
        "app/models/weights",
        "data",
        "data/uploaded_dataset"
    ]
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("[INFO] All required directories verified/created")

ensure_directories()

# Global state untuk pipeline
global_state = {
    'dataset_uploaded': False,
    'dataset_path': None,
    'preprocessed': False,
    'preprocessed_data': None,
    'training_in_progress': False,
    'training_results': {},
    'current_task': None,
    'task_queue': queue.Queue(),
    'progress': 0,
    'logs': []
}

# ============== DATA MODELS ==============

class TrainingConfig(BaseModel):
    """Konfigurasi untuk training"""
    backbones: List[str] = Field(default=["wav2vec2-base", "wav2vec2-xlsr", "hear"])
    epochs: int = Field(default=30, ge=5, le=100)
    batch_size: int = Field(default=16, ge=4, le=64)
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-2)
    patience: int = Field(default=10, ge=3, le=20)
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    val_size: float = Field(default=0.15, ge=0.1, le=0.3)
    augment: bool = Field(default=True)
    pos_weight: float = Field(default=2.0, ge=1.0, le=5.0)


class SaveModelRequest(BaseModel):
    """Request untuk save model"""
    model_name: str
    description: Optional[str] = ""
    tags: Optional[List[str]] = []


class PipelineStatus(BaseModel):
    """Status pipeline"""
    dataset_uploaded: bool
    preprocessed: bool
    training_in_progress: bool
    training_completed: bool
    available_models: int
    current_task: Optional[str]
    progress: int


# ============== UTILITY FUNCTIONS ==============

def log_message(message: str):
    """Log message dengan timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    global_state['logs'].append(log_entry)
    print(log_entry)
    
    # Keep only last 100 logs
    if len(global_state['logs']) > 100:
        global_state['logs'] = global_state['logs'][-100:]


def update_progress(value: int, task: str = None):
    """Update progress global"""
    global_state['progress'] = min(100, max(0, value))
    if task:
        global_state['current_task'] = task
    log_message(f"Progress: {value}% - {task or global_state.get('current_task', '')}")


# ============== DATASET HANDLING ==============

@app.post("/dataset/upload")
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="ZIP file containing CODA TB DREAM dataset"),
    extract_only: bool = Form(default=False, description="Only extract, don't preprocess")
):
    """
    Upload dataset ZIP file (CODA TB DREAM format)
    Structure expected: dataset/audio/*.wav + dataset/metadata.csv
    """
    try:
        log_message(f"Received dataset upload: {file.filename}")
        
        # Validate ZIP
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="File harus berformat ZIP")
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="tb_dataset_")
        zip_path = os.path.join(temp_dir, "dataset.zip")
        
        # Save uploaded file
        content = await file.read()
        with open(zip_path, 'wb') as f:
            f.write(content)
        
        log_message(f"Saved ZIP file ({len(content)} bytes)")
        
        # Extract ZIP
        extract_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        log_message(f"Extracted to: {extract_dir}")
        
        # Find dataset structure
        audio_dir, metadata_file = find_dataset_structure(extract_dir)
        
        if not audio_dir or not metadata_file:
            shutil.rmtree(temp_dir)
            raise HTTPException(
                status_code=400, 
                detail="Struktur dataset tidak valid. Harus mengandung folder audio dan file CSV metadata"
            )
        
        # Move to permanent location
        dataset_dir = "data/uploaded_dataset"
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        shutil.move(extract_dir, dataset_dir)
        
        # Cleanup temp
        shutil.rmtree(temp_dir)
        
        global_state['dataset_uploaded'] = True
        global_state['dataset_path'] = dataset_dir
        global_state['preprocessed'] = False
        
        # Count files
        audio_files = list(Path(dataset_dir).rglob("*.wav")) + list(Path(dataset_dir).rglob("*.mp3"))
        
        log_message(f"Dataset uploaded successfully: {len(audio_files)} audio files found")
        
        # Auto-preprocess if not extract_only
        if not extract_only:
            background_tasks.add_task(run_preprocessing)
        
        return {
            "success": True,
            "message": "Dataset berhasil diupload",
            "dataset_path": dataset_dir,
            "audio_count": len(audio_files),
            "metadata_file": os.path.basename(metadata_file),
            "auto_preprocess": not extract_only
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_message(f"Error uploading dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def find_dataset_structure(extract_dir: str) -> tuple:
    """Find audio directory and metadata file in extracted dataset"""
    audio_dir = None
    metadata_file = None
    
    # Walk through directory
    for root, dirs, files in os.walk(extract_dir):
        # Look for audio directory
        if 'audio' in dirs:
            audio_dir = os.path.join(root, 'audio')
        
        # Look for metadata CSV
        for file in files:
            if file.endswith('.csv') and ('metadata' in file.lower() or 'dataset' in file.lower()):
                metadata_file = os.path.join(root, file)
            elif file.endswith('.csv') and not metadata_file:
                # Fallback: any CSV
                metadata_file = os.path.join(root, file)
    
    return audio_dir, metadata_file


# ============== PREPROCESSING ==============

@app.post("/dataset/preprocess")
async def start_preprocessing(background_tasks: BackgroundTasks):
    """Start preprocessing pipeline"""
    if not global_state['dataset_uploaded']:
        raise HTTPException(status_code=400, detail="Dataset belum diupload")
    
    if global_state['training_in_progress']:
        raise HTTPException(status_code=400, detail="Training sedang berlangsung, tunggu selesai")
    
    background_tasks.add_task(run_preprocessing)
    
    return {
        "success": True,
        "message": "Preprocessing dimulai di background",
        "task": "preprocessing"
    }


def run_preprocessing():
    """Background task untuk preprocessing"""
    try:
        global_state['preprocessed'] = False
        update_progress(0, "Preprocessing dataset...")
        
        dataset_dir = global_state['dataset_path']
        audio_dir, metadata_file = find_dataset_structure(dataset_dir)
        
        log_message(f"Loading metadata from: {metadata_file}")
        update_progress(10, "Loading metadata...")
        
        # Load metadata
        df = pd.read_csv(metadata_file)
        
        # Validate required columns
        required_cols = ['audio_filename', 'patient_id', 'age', 'gender', 'tb_label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            # Try to auto-detect columns
            log_message(f"Missing columns: {missing_cols}. Attempting auto-mapping...")
            df = auto_map_columns(df)
        
        update_progress(20, "Preprocessing audio files...")
        
        # Preprocess audio
        preprocessor = AudioPreprocessor(SAMPLE_RATE)
        segmenter = CoughSegmenter(SAMPLE_RATE)
        
        processed_data = []
        total = len(df)
        
        for idx, row in df.iterrows():
            try:
                audio_path = os.path.join(audio_dir, row['audio_filename'])
                
                if not os.path.exists(audio_path):
                    log_message(f"Audio file not found: {row['audio_filename']}")
                    continue
                
                # Load and preprocess
                waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
                
                # Segment cough
                segments = segmenter.segment(waveform)
                
                if len(segments) > 0:
                    start, end = max(segments, key=lambda x: x[1] - x[0])
                    cough_audio = waveform[start:end]
                else:
                    cough_audio = waveform
                
                # Pad/truncate
                if len(cough_audio) < MAX_AUDIO_LENGTH:
                    cough_audio = np.pad(cough_audio, (0, MAX_AUDIO_LENGTH - len(cough_audio)))
                else:
                    cough_audio = cough_audio[:MAX_AUDIO_LENGTH]
                
                processed_data.append({
                    'audio_filename': row['audio_filename'],
                    'patient_id': row['patient_id'],
                    'audio_data': cough_audio,
                    'segments_found': len(segments),
                    **{k: v for k, v in row.items() if k not in ['audio_filename', 'patient_id']}
                })
                
            except Exception as e:
                log_message(f"Error processing {row.get('audio_filename', 'unknown')}: {e}")
                continue
            
            # Update progress
            progress = 20 + int((idx / total) * 70)
            if idx % 10 == 0:
                update_progress(progress, f"Processing audio {idx+1}/{total}")
        
        # Save preprocessed data
        global_state['preprocessed_data'] = processed_data
        global_state['preprocessed'] = True
        
        update_progress(100, f"Preprocessing complete! {len(processed_data)} samples ready")
        log_message(f"Preprocessed {len(processed_data)} samples successfully")
        
    except Exception as e:
        log_message(f"Preprocessing error: {str(e)}")
        update_progress(0, f"Preprocessing failed: {str(e)}")


def auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-map column names jika format berbeda"""
    column_mapping = {
        'file': 'audio_filename',
        'filename': 'audio_filename',
        'audio': 'audio_filename',
        'id': 'patient_id',
        'patient': 'patient_id',
        'subject': 'patient_id',
        'sex': 'gender',
        'label': 'tb_label',
        'tb': 'tb_label',
        'class': 'tb_label',
        'target': 'tb_label'
    }
    
    # Rename columns
    new_columns = {}
    for old_col in df.columns:
        old_lower = old_col.lower()
        if old_lower in column_mapping:
            new_columns[old_col] = column_mapping[old_lower]
        else:
            new_columns[old_col] = old_col
    
    df = df.rename(columns=new_columns)
    
    # Add default values for missing columns
    if 'cough_duration_days' not in df.columns:
        df['cough_duration_days'] = 0
    if 'has_fever' not in df.columns:
        df['has_fever'] = False
    if 'has_night_sweats' not in df.columns:
        df['has_night_sweats'] = False
    if 'has_weight_loss' not in df.columns:
        df['has_weight_loss'] = False
    if 'has_chest_pain' not in df.columns:
        df['has_chest_pain'] = False
    if 'has_shortness_breath' not in df.columns:
        df['has_shortness_breath'] = False
    if 'previous_tb_history' not in df.columns:
        df['previous_tb_history'] = False
    
    return df


# ============== TRAINING ==============

@app.post("/training/start")
async def start_training(
    background_tasks: BackgroundTasks,
    config: TrainingConfig
):
    """Start training dengan multiple backbones"""
    if not global_state['preprocessed']:
        raise HTTPException(status_code=400, detail="Data belum dipreprocessing")
    
    if global_state['training_in_progress']:
        raise HTTPException(status_code=400, detail="Training sedang berlangsung")
    
    background_tasks.add_task(run_training, config)
    global_state['training_in_progress'] = True
    
    return {
        "success": True,
        "message": "Training dimulai dengan konfigurasi:",
        "config": config.dict(),
        "task": "training"
    }


def run_training(config: TrainingConfig):
    """Background training task"""
    try:
        global_state['training_results'] = {}
        
        data = global_state['preprocessed_data']
        df = pd.DataFrame(data)
        
        # Split data
        from sklearn.model_selection import GroupShuffleSplit
        
        gss = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=df['patient_id']))
        
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        gss_val = GroupShuffleSplit(n_splits=1, test_size=config.val_size, random_state=42)
        train_idx_final, val_idx = next(gss_val.split(train_df, groups=train_df['patient_id']))
        
        train_final_df = train_df.iloc[train_idx_final].copy()
        val_df = train_df.iloc[val_idx].copy()
        
        log_message(f"Data split: Train={len(train_final_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Train each backbone
        total_backbones = len(config.backbones)
        
        for idx, backbone_name in enumerate(config.backbones):
            progress_base = (idx / total_backbones) * 100
            update_progress(int(progress_base), f"Training {backbone_name}...")
            
            try:
                result = train_single_backbone(
                    backbone_name, train_final_df, val_df, test_df, config, progress_base
                )
                global_state['training_results'][backbone_name] = result
                log_message(f"Completed training {backbone_name}: AUROC={result['metrics']['auroc']:.4f}")
                
            except Exception as e:
                log_message(f"Error training {backbone_name}: {str(e)}")
                continue
        
        # Compare and save results
        if len(global_state['training_results']) > 0:
            compare_and_save_results()
        
        global_state['training_in_progress'] = False
        update_progress(100, "Training complete!")
        
    except Exception as e:
        log_message(f"Training error: {str(e)}")
        global_state['training_in_progress'] = False
        update_progress(0, f"Training failed: {str(e)}")


def train_single_backbone(
    backbone_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: TrainingConfig,
    progress_base: float
):
    """Train single backbone model"""
    
    # Load backbone
    backbone = BackboneFactory.create(backbone_name, str(DEVICE))
    audio_dim = backbone.output_dim
    
    log_message(f"Extracting features with {backbone_name}...")
    
    # Extract features for all splits
    train_features = extract_features_for_dataframe(train_df, backbone)
    val_features = extract_features_for_dataframe(val_df, backbone)
    test_features = extract_features_for_dataframe(test_df, backbone)
    
    update_progress(int(progress_base + 20), f"Training classifier for {backbone_name}...")
    
    # Initialize classifier components
    metadata_encoder = SimpleMetadataEncoder(embedding_dim=32).to(DEVICE)
    feature_fusion = SimpleConcatFusion(
        audio_dim=audio_dim, metadata_dim=32, fused_dim=512
    ).to(DEVICE)
    classifier = TBClassifier(
        input_dim=512, hidden_dims=[256, 128], num_heads=4, dropout=0.3
    ).to(DEVICE)
    
    # Optimizer & loss
    params = (
        list(metadata_encoder.parameters()) +
        list(feature_fusion.parameters()) +
        list(classifier.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=0.01)
    criterion = get_tb_loss_fn(pos_weight=config.pos_weight)
    
    # Training loop
    best_auroc = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_auroc': []}
    
    for epoch in range(config.epochs):
        # Training
        metadata_encoder.train()
        feature_fusion.train()
        classifier.train()
        
        epoch_loss = 0
        for sample in train_features:
            audio_feat = sample['audio_features'].unsqueeze(0).to(DEVICE)
            metadata = metadata_encoder.encode(sample['metadata']).to(DEVICE)
            label = torch.tensor([sample['label']], dtype=torch.long).to(DEVICE)
            
            fused = feature_fusion(audio_feat, metadata)
            logits = classifier(fused)
            
            loss = criterion(logits, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        val_metrics = evaluate_model(
            metadata_encoder, feature_fusion, classifier,
            val_features, criterion, DEVICE
        )
        
        history['train_loss'].append(epoch_loss / len(train_features))
        history['val_auroc'].append(val_metrics['auroc'])
        
        if val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'backbone_name': backbone_name,
                'backbone_dim': audio_dim,
                'metadata_encoder': metadata_encoder.state_dict(),
                'feature_fusion': feature_fusion.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch,
                'auroc': best_auroc,
                'history': history,
                'config': config.dict()
            }
            
            model_path = f"app/models/weights/{backbone_name}_model.pth"
            torch.save(checkpoint, model_path)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                log_message(f"Early stopping for {backbone_name} at epoch {epoch+1}")
                break
        
        if epoch % 5 == 0:
            update_progress(
                int(progress_base + 20 + (epoch / config.epochs) * 60),
                f"{backbone_name} - Epoch {epoch+1}/{config.epochs}, AUROC: {val_metrics['auroc']:.4f}"
            )
    
    # Final test evaluation
    checkpoint = torch.load(f"app/models/weights/{backbone_name}_model.pth", map_location=DEVICE)
    metadata_encoder.load_state_dict(checkpoint['metadata_encoder'])
    feature_fusion.load_state_dict(checkpoint['feature_fusion'])
    classifier.load_state_dict(checkpoint['classifier'])
    
    test_metrics = evaluate_model(
        metadata_encoder, feature_fusion, classifier,
        test_features, criterion, DEVICE
    )
    
    return {
        'backbone': backbone_name,
        'metrics': test_metrics,
        'history': history,
        'model_path': f"app/models/weights/{backbone_name}_model.pth"
    }


def extract_features_for_dataframe(df: pd.DataFrame, backbone):
    """Extract audio features untuk dataframe"""
    features = []
    
    backbone.model.eval()
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {backbone.model_name}"):
        try:
            audio_data = row['audio_data']
            waveform_tensor = torch.tensor(audio_data, dtype=torch.float32).to(DEVICE)
            
            with torch.no_grad():
                audio_features = backbone.extract_features(waveform_tensor)
            
            metadata = {
                'age': int(row['age']),
                'gender': row['gender'],
                'has_fever': bool(row.get('has_fever', False)),
                'has_cough': True,
                'cough_duration_days': int(row.get('cough_duration_days', 0)),
                'has_night_sweats': bool(row.get('has_night_sweats', False)),
                'has_weight_loss': bool(row.get('has_weight_loss', False)),
                'has_chest_pain': bool(row.get('has_chest_pain', False)),
                'has_shortness_breath': bool(row.get('has_shortness_breath', False)),
                'previous_tb_history': bool(row.get('previous_tb_history', False))
            }
            
            features.append({
                'audio_features': audio_features.cpu().squeeze(0),
                'metadata': metadata,
                'label': int(row['tb_label'])
            })
        except Exception as e:
            continue
    
    return features


def evaluate_model(metadata_encoder, feature_fusion, classifier, data, criterion, device):
    """Evaluate model pada dataset"""
    metadata_encoder.eval()
    feature_fusion.eval()
    classifier.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    
    with torch.no_grad():
        for sample in data:
            audio_feat = sample['audio_features'].unsqueeze(0).to(device)
            metadata = metadata_encoder.encode(sample['metadata']).to(device)
            label = torch.tensor([sample['label']], dtype=torch.long).to(device)
            
            fused = feature_fusion(audio_feat, metadata)
            logits = classifier(fused)
            probs = torch.softmax(logits, dim=1)
            
            loss = criterion(logits, label)
            total_loss += loss.item()
            
            pred = torch.argmax(logits, dim=1)
            all_preds.append(pred.item())
            all_labels.append(sample['label'])
            all_probs.append(probs[0][1].item())
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except:
        auroc = 0.5
    
    cm = confusion_matrix(all_labels, all_preds)
    
    if cm.shape == (2, 2):
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    else:
        sensitivity = recall
        specificity = 0
    
    return {
        'loss': total_loss / len(data),
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
        'auroc': auroc,
        'confusion_matrix': cm.tolist()
    }


def compare_and_save_results():
    """Compare models dan save comparison"""
    results = global_state['training_results']
    
    comparison = []
    for backbone_name, result in results.items():
        m = result['metrics']
        comparison.append({
            'name': backbone_name,
            'backbone': backbone_name,
            'accuracy': m['accuracy'],
            'sensitivity': m['sensitivity'],
            'specificity': m['specificity'],
            'f1': m['f1'],
            'auroc': m['auroc'],
            'is_best': False
        })
    
    # Sort by AUROC and mark best
    comparison.sort(key=lambda x: x['auroc'], reverse=True)
    if comparison:
        comparison[0]['is_best'] = True
    
    # Save comparison
    comparison_path = 'app/models/weights/model_comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump({
            'models': comparison,
            'timestamp': datetime.now().isoformat(),
            'total_models': len(comparison)
        }, f, indent=2)
    
    # Save best model info
    if comparison:
        best_info = {
            'best_backbone': comparison[0]['name'],
            'metrics': comparison[0],
            'timestamp': datetime.now().isoformat()
        }
        with open('app/models/weights/best_model.json', 'w') as f:
            json.dump(best_info, f, indent=2)
    
    # Generate comparison plot
    generate_comparison_plot(comparison)
    
    log_message("Model comparison saved and visualization generated")


def generate_comparison_plot(comparison: list):
    """Generate comparison visualization"""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Comparison - TB Detection Performance', fontsize=14, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'AUROC']
        metric_keys = ['accuracy', 'precision', 'sensitivity', 'specificity', 'f1', 'auroc']
        
        for idx, (metric_name, metric_key) in enumerate(zip(metrics, metric_keys)):
            ax = axes[idx // 3, idx % 3]
            
            names = [c['name'] for c in comparison]
            values = [c[metric_key] for c in comparison]
            colors = ['#4CAF50' if c['is_best'] else '#667eea' for c in comparison]
            
            bars = ax.bar(names, values, color=colors)
            ax.set_ylabel(metric_name)
            ax.set_ylim(0, 1)
            ax.set_title(metric_name, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plot_path = 'app/models/weights/model_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        log_message(f"Comparison plot saved to: {plot_path}")
        
    except Exception as e:
        log_message(f"Error generating plot: {e}")


# ============== SAVE MODEL ==============

@app.post("/models/save")
async def save_model(request: SaveModelRequest):
    """Save trained model dengan metadata"""
    try:
        model_name = request.model_name

        # Check if model exists in training results
        if model_name not in global_state['training_results']:
            raise HTTPException(status_code=404, detail=f"Model {model_name} tidak ditemukan dalam hasil training")

        result = global_state['training_results'][model_name]

        # Create model package
        model_package = {
            'model_name': model_name,
            'description': request.description,
            'tags': request.tags,
            'metrics': result['metrics'],
            'history': result['history'],
            'saved_at': datetime.now().isoformat(),
            'config': result.get('config', {})
        }

        # Save metadata
        package_path = f"app/models/weights/{model_name}_package.json"
        with open(package_path, 'w') as f:
            json.dump(model_package, f, indent=2)

        log_message(f"Model {model_name} saved with metadata")

        return {
            "success": True,
            "message": f"Model {model_name} berhasil disimpan",
            "model_path": result['model_path'],
            "metadata_path": package_path,
            "metrics": result['metrics']
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/download/{model_name}")
async def download_model(model_name: str):
    """Download trained model as a packaged file"""
    try:
        # Check if model exists
        model_path = f"app/models/weights/{model_name}_model.pth"
        metadata_path = f"app/models/weights/{model_name}_package.json"

        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model {model_name} tidak ditemukan")

        # Create a temporary ZIP file containing model and metadata
        import tempfile
        zip_path = os.path.join(tempfile.gettempdir(), f"{model_name}_tb_model.zip")

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add model weights
            zf.write(model_path, f"{model_name}_model.pth")
            # Add metadata if exists
            if os.path.exists(metadata_path):
                zf.write(metadata_path, f"{model_name}_metadata.json")

        log_message(f"Model {model_name} packaged for download")

        return FileResponse(
            zip_path,
            media_type='application/zip',
            filename=f"{model_name}_tb_model.zip"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/upload")
async def upload_model(
    file: UploadFile = File(..., description="ZIP file containing model files (.pth and optional .json)")
):
    """Upload and import a trained model"""
    try:
        log_message(f"Received model upload: {file.filename}")

        # Validate ZIP
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="File harus berformat ZIP")

        # Read uploaded file
        content = await file.read()

        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="model_upload_")
        zip_path = os.path.join(temp_dir, "model.zip")

        with open(zip_path, 'wb') as f:
            f.write(content)

        # Extract ZIP
        extract_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Find model file and metadata
        model_file = None
        metadata_file = None

        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                if f.endswith('_model.pth'):
                    model_file = os.path.join(root, f)
                elif f.endswith('_metadata.json') or f.endswith('_package.json'):
                    metadata_file = os.path.join(root, f)

        if not model_file:
            shutil.rmtree(temp_dir)
            raise HTTPException(status_code=400, detail="File model .pth tidak ditemukan dalam ZIP")

        # Get model name from filename
        model_name = os.path.basename(model_file).replace('_model.pth', '')

        # Copy to weights directory
        weights_dir = "app/models/weights"
        os.makedirs(weights_dir, exist_ok=True)

        dest_model_path = os.path.join(weights_dir, f"{model_name}_model.pth")
        shutil.copy2(model_file, dest_model_path)

        if metadata_file:
            dest_metadata_path = os.path.join(weights_dir, f"{model_name}_package.json")
            shutil.copy2(metadata_file, dest_metadata_path)

        # Reload model manager
        manager = get_model_manager()
        manager._scan_models()

        # Cleanup temp
        shutil.rmtree(temp_dir)

        log_message(f"Model {model_name} uploaded and imported successfully")

        return {
            "success": True,
            "message": f"Model {model_name} berhasil diupload dan diimport",
            "model_name": model_name,
            "model_path": dest_model_path
        }

    except HTTPException:
        raise
    except Exception as e:
        log_message(f"Error uploading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== MODEL MANAGEMENT ==============

@app.get("/models/list")
async def list_models():
    """List all available trained models"""
    try:
        manager = get_model_manager()
        models = manager.list_models()

        return {
            "success": True,
            "models": models,
            "total": len(models),
            "current_model": manager.current_model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/load")
async def load_model_endpoint(model_name: str = Form(...)):
    """Load a specific model for prediction"""
    try:
        manager = get_model_manager()
        success = manager.load_model(model_name)

        if success:
            return {
                "success": True,
                "message": f"Model {model_name} berhasil dimuat",
                "model_info": manager.get_current_model_info()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Gagal memuat model {model_name}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== STATUS & STREAMING ==============

@app.get("/pipeline/status")
async def get_pipeline_status():
    """Get current pipeline status"""
    return {
        "dataset_uploaded": global_state['dataset_uploaded'],
        "preprocessed": global_state['preprocessed'],
        "preprocessed_samples": len(global_state['preprocessed_data']) if global_state['preprocessed_data'] else 0,
        "training_in_progress": global_state['training_in_progress'],
        "training_completed": len(global_state['training_results']) > 0,
        "available_models": len(global_state['training_results']),
        "current_task": global_state['current_task'],
        "progress": global_state['progress'],
        "logs": global_state['logs'][-20:]  # Last 20 logs
    }


@app.get("/pipeline/logs")
async def get_logs(limit: int = 50):
    """Get recent logs"""
    return {
        "logs": global_state['logs'][-limit:],
        "total_logs": len(global_state['logs'])
    }


@app.get("/training/results")
async def get_training_results():
    """Get training results"""
    return {
        "training_completed": len(global_state['training_results']) > 0,
        "models": list(global_state['training_results'].keys()),
        "results": {
            name: {
                'metrics': result['metrics'],
                'history': result.get('history', {})
            }
            for name, result in global_state['training_results'].items()
        },
        "comparison_available": os.path.exists('app/models/weights/model_comparison.json')
    }


@app.get("/training/visualization")
async def get_visualization():
    """Get comparison visualization"""
    plot_path = 'app/models/weights/model_comparison.png'
    
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="Visualization belum tersedia")
    
    return StreamingResponse(
        open(plot_path, 'rb'),
        media_type='image/png'
    )


# ============== PREDICTION ==============

@app.post("/predict")
async def predict(
    audio: UploadFile = File(...),
    age: int = Form(...),
    gender: str = Form(...),
    has_fever: bool = Form(default=False),
    has_cough: bool = Form(default=True),
    cough_duration_days: int = Form(default=0),
    has_night_sweats: bool = Form(default=False),
    has_weight_loss: bool = Form(default=False),
    has_chest_pain: bool = Form(default=False),
    has_shortness_breath: bool = Form(default=False),
    previous_tb_history: bool = Form(default=False),
    model_name: Optional[str] = Form(default=None)
):
    """Prediction dengan trained model"""
    try:
        # Load model
        manager = get_model_manager()
        
        if model_name:
            if not manager.load_model(model_name):
                return {"success": False, "error": f"Model {model_name} tidak ditemukan"}
        elif not manager.is_ready():
            manager.load_best_model()
        
        # Process audio
        audio_bytes = await audio.read()
        audio_buffer = io.BytesIO(audio_bytes)
        
        waveform, sr = librosa.load(audio_buffer, sr=SAMPLE_RATE, mono=True)
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        
        # Segment
        segmenter = CoughSegmenter(SAMPLE_RATE)
        segments = segmenter.segment(waveform)
        
        if len(segments) == 0:
            return {"success": False, "error": "Tidak terdeteksi suara batuk"}
        
        start, end = max(segments, key=lambda x: x[1] - x[0])
        cough_audio = waveform[start:end]
        
        if len(cough_audio) < MAX_AUDIO_LENGTH:
            cough_audio = np.pad(cough_audio, (0, MAX_AUDIO_LENGTH - len(cough_audio)))
        else:
            cough_audio = cough_audio[:MAX_AUDIO_LENGTH]
        
        # Extract features
        current_info = manager.get_current_model_info()
        backbone_name = current_info['backbone'] if current_info else 'wav2vec2-xlsr'
        
        if backbone_name not in backbone_instances:
            backbone = BackboneFactory.create(backbone_name, str(DEVICE))
            backbone_instances[backbone_name] = backbone
        
        backbone = backbone_instances[backbone_name]
        waveform_tensor = torch.tensor(cough_audio, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            audio_features = backbone.extract_features(waveform_tensor)
        
        # Classify
        components = manager.get_components()
        metadata = {'age': age, 'gender': gender, 'has_fever': has_fever, 'has_cough': has_cough,
                   'cough_duration_days': cough_duration_days, 'has_night_sweats': has_night_sweats,
                   'has_weight_loss': has_weight_loss, 'has_chest_pain': has_chest_pain,
                   'has_shortness_breath': has_shortness_breath, 'previous_tb_history': previous_tb_history}
        
        metadata_embedding = components['metadata_encoder'].encode(metadata).to(DEVICE)
        
        with torch.no_grad():
            fused = components['feature_fusion'](audio_features, metadata_embedding)
            logits = components['classifier'](fused)
            probs = torch.softmax(logits, dim=1)
            tb_prob = probs[0][1].item()
        
        # Risk level
        if tb_prob < 0.3:
            risk = "RENDAH"
            rec = "Skor TB rendah. Kemungkinan TB kecil. Lanjutkan monitoring."
        elif tb_prob < 0.7:
            risk = "MENENGAH"
            rec = "Skor TB menengah. Disarankan pemeriksaan tambahan."
        else:
            risk = "TINGGI"
            rec = "Skor TB tinggi. SEGERA rujuk ke fasilitas kesehatan."
        
        return {
            "success": True,
            "result": {
                "tb_probability": round(tb_prob, 4),
                "risk_level": risk,
                "recommendation": rec,
                "model_used": manager.current_model,
                "backbone_used": backbone_name
            }
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


# Global backbone instances
backbone_instances = {}


# ============== FRONTEND ==============

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve integrated UI"""
    index_path = os.path.join(os.path.dirname(__file__), "static", "index_v3.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return {"message": "TB Detector v3 API - Use /docs for documentation"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
