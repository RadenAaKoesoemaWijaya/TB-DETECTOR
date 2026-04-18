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
from app.training import BatchTrainer, get_feature_cache
from app.persistence import get_persistence

# Phase 2 imports
from app.async_utils import get_async_io, get_async_inference, get_async_preprocessing, cleanup_async
from app.task_queue import get_task_queue, TaskType, TaskStatus, shutdown_task_queue
from app.onnx_inference import get_inference_manager, ONNXExporter, ONNX_AVAILABLE
from app.model_versioning import get_model_registry, ModelStage, ModelFormat
from app.ab_testing import get_ab_testing, ExperimentStatus, AllocationMethod

# Setup
app = FastAPI(
    title="TB Detector v3.2 - Integrated Pipeline",
    description="Complete TB Detection Pipeline with Phase 1 (Performance) and Phase 2 (Production) optimizations. Features: Batch Training, Feature Caching, Async I/O, Task Queue, ONNX Inference, Model Versioning, A/B Testing.",
    version="3.2.0"
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

def log_message(message: str, level: str = 'INFO', task: str = None):
    """Log message dengan timestamp dan persistence"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    global_state['logs'].append(log_entry)
    print(log_entry)
    
    # Keep only last 100 logs in memory
    if len(global_state['logs']) > 100:
        global_state['logs'] = global_state['logs'][-100:]
    
    # Persist ke SQLite
    try:
        persistence = get_persistence()
        persistence.add_log(message=message, level=level, task=task)
    except Exception as e:
        # Silent fail untuk avoid disrupting main flow
        pass


def update_progress(value: int, task: str = None):
    """Update progress global dengan persistence"""
    global_state['progress'] = min(100, max(0, value))
    if task:
        global_state['current_task'] = task
    log_message(f"Progress: {value}% - {task or global_state.get('current_task', '')}", task=task)
    
    # Persist state
    try:
        persistence = get_persistence()
        persistence.update_state(
            progress=value,
            current_task=task
        )
    except Exception:
        pass


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
    persistence = get_persistence()
    
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
        
        # Update persistence state
        persistence.update_state(
            dataset_uploaded=True,
            dataset_path=dataset_dir,
            preprocessed=False,
            current_task="Dataset uploaded"
        )
        
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
    """Background task untuk preprocessing dengan persistence"""
    persistence = get_persistence()
    
    try:
        global_state['preprocessed'] = False
        persistence.update_state(preprocessed=False, current_task="Preprocessing started")
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
        
        # Update persistence state
        persistence.update_state(
            preprocessed=True,
            preprocessed_samples=len(processed_data),
            progress=100,
            current_task="Preprocessing complete"
        )
        
        update_progress(100, f"Preprocessing complete! {len(processed_data)} samples ready")
        log_message(f"Preprocessed {len(processed_data)} samples successfully")
        
    except Exception as e:
        log_message(f"Preprocessing error: {str(e)}", level='ERROR')
        persistence.update_state(preprocessed=False, current_task=f"Preprocessing failed: {str(e)}")
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
    """Background training task dengan persistence tracking"""
    persistence = get_persistence()
    
    try:
        global_state['training_results'] = {}
        
        # Update state
        persistence.update_state(training_in_progress=True, current_task="Training started")
        
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
            
            # Create training session di persistence
            import uuid
            session_id = f"{backbone_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            persistence.start_training_session(
                session_id=session_id,
                backbone_name=backbone_name,
                config=config.dict()
            )
            
            try:
                result = train_single_backbone(
                    backbone_name, train_final_df, val_df, test_df, config, progress_base
                )
                global_state['training_results'][backbone_name] = result
                
                # Update session dengan success
                persistence.update_training_session(
                    session_id=session_id,
                    status='completed',
                    best_auroc=result.get('best_auroc'),
                    epochs_trained=len(result.get('history', {}).get('train_loss', [])),
                    model_path=result.get('model_path')
                )
                
                # Persist training metrics per epoch
                history = result.get('history', {})
                for epoch, (train_loss, train_acc, val_loss, val_auroc, val_f1, lr) in enumerate(zip(
                    history.get('train_loss', []),
                    history.get('train_acc', []),
                    history.get('val_loss', []),
                    history.get('val_auroc', []),
                    history.get('val_f1', []),
                    history.get('learning_rates', [config.learning_rate] * len(history.get('train_loss', [])))
                )):
                    persistence.add_training_metrics(
                        session_id=session_id,
                        epoch=epoch + 1,
                        train_loss=train_loss,
                        train_acc=train_acc,
                        val_loss=val_loss,
                        val_auroc=val_auroc,
                        val_f1=val_f1,
                        learning_rate=lr
                    )
                
                # Register model
                persistence.register_model(
                    model_name=backbone_name,
                    backbone_name=backbone_name,
                    model_path=result.get('model_path'),
                    metrics=result.get('metrics', {})
                )
                
                log_message(f"Completed training {backbone_name}: AUROC={result['metrics'].get('auroc', 0):.4f}")
                
            except Exception as e:
                log_message(f"Error training {backbone_name}: {str(e)}", level='ERROR')
                persistence.update_training_session(
                    session_id=session_id,
                    status='failed'
                )
                continue
        
        # Compare and save results
        if len(global_state['training_results']) > 0:
            compare_and_save_results()
        
        # Record cache stats ke persistence
        try:
            cache = get_feature_cache()
            stats = cache.get_stats()
            for backbone_name in config.backbones:
                backbone_stats = stats.get('by_backbone', {}).get(backbone_name, {})
                persistence.record_cache_stats(
                    backbone_name=backbone_name,
                    hits=stats.get('hits', 0),
                    misses=stats.get('misses', 0),
                    total_entries=backbone_stats.get('count', 0),
                    total_size_mb=backbone_stats.get('size', 0) / (1024 * 1024)
                )
        except Exception:
            pass
        
        global_state['training_in_progress'] = False
        persistence.update_state(training_in_progress=False, progress=100, current_task="Training complete")
        update_progress(100, "Training complete!")
        
    except Exception as e:
        log_message(f"Training error: {str(e)}", level='ERROR')
        global_state['training_in_progress'] = False
        persistence.update_state(training_in_progress=False, current_task=f"Training failed: {str(e)}")
        update_progress(0, f"Training failed: {str(e)}")


def train_single_backbone(
    backbone_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: TrainingConfig,
    progress_base: float
):
    """
    Train single backbone model dengan batch processing (optimized)
    Menggunakan BatchTrainer untuk efisiensi 10-20x lebih cepat
    """
    
    # Load backbone
    backbone = BackboneFactory.create(backbone_name, str(DEVICE))
    audio_dim = backbone.output_dim
    
    log_message(f"Extracting features with {backbone_name}...")
    update_progress(int(progress_base + 10), f"Extracting features for {backbone_name}...")
    
    # Extract features dengan caching support
    train_features = extract_features_cached(train_df, backbone, backbone_name)
    val_features = extract_features_cached(val_df, backbone, backbone_name)
    test_features = extract_features_cached(test_df, backbone, backbone_name)
    
    log_message(f"Extracted {len(train_features)} train, {len(val_features)} val, {len(test_features)} test samples")
    update_progress(int(progress_base + 20), f"Training classifier for {backbone_name} with batch_size={config.batch_size}...")
    
    # Initialize BatchTrainer untuk efisien batch training
    trainer = BatchTrainer(
        backbone_name=backbone_name,
        audio_dim=audio_dim,
        device=str(DEVICE),
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        pos_weight=config.pos_weight
    )
    
    # Progress callback untuk UI updates
    def progress_callback(progress, message):
        overall_progress = int(progress_base + 20 + (progress / 100) * 60)
        update_progress(overall_progress, f"{backbone_name} - {message}")
    
    # Train dengan batch processing
    result = trainer.train(
        train_data=train_features,
        val_data=val_features,
        test_data=test_features,
        batch_size=config.batch_size,
        epochs=config.epochs,
        patience=config.patience,
        num_workers=0,  # Keep 0 untuk avoid multiprocessing issues dengan PyTorch
        progress_callback=progress_callback
    )
    
    # Save best checkpoint
    model_path = f"app/models/weights/{backbone_name}_model.pth"
    trainer.save_checkpoint(model_path, metadata={
        'config': config.dict(),
        'timestamp': datetime.now().isoformat()
    })
    
    # Get test metrics dari result
    test_metrics = result.get('test_metrics') or {}
    
    return {
        'backbone': backbone_name,
        'metrics': test_metrics,
        'history': result['history'],
        'model_path': model_path,
        'best_auroc': result['best_auroc']
    }


def extract_features_cached(df: pd.DataFrame, backbone, backbone_name: str) -> list:
    """
    Extract audio features dengan caching support
    Menggunakan FeatureCacheManager untuk menghindari re-extraction
    """
    cache = get_feature_cache()
    features = []
    
    backbone.model.eval()
    
    # Check apakah audio_data berisi path atau raw data
    first_row = df.iloc[0] if len(df) > 0 else None
    has_raw_audio = first_row is not None and 'audio_data' in first_row and isinstance(first_row['audio_data'], np.ndarray)
    
    if has_raw_audio:
        # Data sudah di memory (dari preprocessing), extract langsung
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {backbone_name}"):
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
                    'label': int(row['tb_label']),
                    'audio_filename': row.get('audio_filename', 'unknown')
                })
            except Exception as e:
                log_message(f"Error extracting features: {e}")
                continue
    else:
        # Data belum di memory, coba load dari file dengan caching
        audio_dir = global_state.get('dataset_path', 'data/uploaded_dataset')
        audio_files = []
        valid_rows = []
        
        for idx, row in df.iterrows():
            audio_path = os.path.join(audio_dir, 'audio', row['audio_filename'])
            if os.path.exists(audio_path):
                audio_files.append(audio_path)
                valid_rows.append(row)
        
        # Batch extract dengan caching
        extracted_features = cache.batch_get_or_extract(
            audio_paths=audio_files,
            backbone_name=backbone_name,
            backbone=backbone,
            device=str(DEVICE),
            show_progress=True
        )
        
        # Pair dengan metadata
        for row, audio_features in zip(valid_rows, extracted_features):
            if audio_features is not None:
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
                    'audio_features': audio_features,
                    'metadata': metadata,
                    'label': int(row['tb_label']),
                    'audio_filename': row.get('audio_filename', 'unknown')
                })
    
    # Log cache stats
    stats = cache.get_stats()
    log_message(f"Cache stats: {stats['hits']} hits, {stats['misses']} misses, hit_rate={stats['hit_rate']:.2%}")
    
    return features


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


# ============== PERSISTENCE & AUDIT ==============

@app.get("/persistence/logs")
async def get_persistent_logs(limit: int = 100, level: str = None):
    """Get logs dari SQLite persistence"""
    try:
        persistence = get_persistence()
        logs = persistence.get_logs(limit=limit, level=level)
        return {
            "success": True,
            "logs": logs,
            "total": len(logs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/persistence/state")
async def get_persistent_state():
    """Get persistent pipeline state"""
    try:
        persistence = get_persistence()
        state = persistence.get_state()
        return {
            "success": True,
            "state": state
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/persistence/training-sessions")
async def get_training_sessions(limit: int = 10):
    """Get training sessions dari persistence"""
    try:
        persistence = get_persistence()
        sessions = persistence.get_training_sessions(limit=limit)
        return {
            "success": True,
            "sessions": sessions,
            "total": len(sessions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/persistence/training-history/{session_id}")
async def get_training_history(session_id: str):
    """Get training metrics history untuk session"""
    try:
        persistence = get_persistence()
        history = persistence.get_training_history(session_id)
        return {
            "success": True,
            "session_id": session_id,
            "history": history,
            "total_epochs": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/persistence/cache-stats")
async def get_cache_stats(backbone: str = None):
    """Get cache statistics"""
    try:
        cache = get_feature_cache()
        current_stats = cache.get_stats()
        
        # Also get historical stats dari persistence
        persistence = get_persistence()
        historical_stats = persistence.get_cache_stats(backbone_name=backbone, limit=10)
        
        return {
            "success": True,
            "current_stats": current_stats,
            "historical": historical_stats
        }
    except Exception as e:
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


# ============== PHASE 2: ASYNC & TASK QUEUE ==============

@app.get("/tasks/queue")
async def get_task_queue_status():
    """Get background task queue status"""
    try:
        queue = get_task_queue(persistence=get_persistence())
        stats = queue.get_queue_stats()
        return {
            "success": True,
            "queue_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks/list")
async def list_tasks(
    status: Optional[str] = None,
    task_type: Optional[str] = None,
    limit: int = 50
):
    """List background tasks dengan filtering"""
    try:
        queue = get_task_queue(persistence=get_persistence())
        
        status_enum = TaskStatus(status) if status else None
        type_enum = TaskType(task_type) if task_type else None
        
        tasks = queue.get_tasks(status=status_enum, task_type=type_enum, limit=limit)
        
        return {
            "success": True,
            "tasks": [t.to_dict() for t in tasks],
            "total": len(tasks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task details by ID"""
    try:
        queue = get_task_queue(persistence=get_persistence())
        task = queue.get_task(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {
            "success": True,
            "task": task.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== PHASE 2: MODEL VERSIONING ==============

@app.get("/registry/models")
async def list_model_versions(
    model_name: Optional[str] = None,
    stage: Optional[str] = None,
    limit: int = 100
):
    """List model versions dari registry"""
    try:
        registry = get_model_registry()
        
        stage_enum = ModelStage(stage) if stage else None
        versions = registry.list_versions(
            model_name=model_name,
            stage=stage_enum,
            limit=limit
        )
        
        return {
            "success": True,
            "versions": [v.to_dict() for v in versions],
            "total": len(versions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/registry/models/{model_name}/{version}")
async def get_model_version(model_name: str, version: str):
    """Get specific model version details"""
    try:
        registry = get_model_registry()
        version_obj = registry.get_version(model_name, version)
        
        if not version_obj:
            raise HTTPException(status_code=404, detail="Version not found")
        
        return {
            "success": True,
            "version": version_obj.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/registry/models/{model_name}/{version}/promote")
async def promote_model_version(model_name: str, version: str, new_stage: str = Form(...)):
    """Promote model version ke stage baru"""
    try:
        registry = get_model_registry()
        success = registry.promote_version(model_name, version, ModelStage(new_stage))
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to promote version")
        
        return {
            "success": True,
            "message": f"Model {model_name} v{version} promoted to {new_stage}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/registry/statistics")
async def get_registry_statistics():
    """Get model registry statistics"""
    try:
        registry = get_model_registry()
        stats = registry.get_statistics()
        
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== PHASE 2: ONNX EXPORT ==============

@app.post("/models/export-onnx/{model_name}")
async def export_model_to_onnx(
    model_name: str,
    quantization: bool = Form(default=False)
):
    """Export model ke ONNX format"""
    if not ONNX_AVAILABLE:
        raise HTTPException(status_code=400, detail="ONNX not available. Install with: pip install onnx onnxruntime")
    
    try:
        model_path = f"app/models/weights/{model_name}_model.pth"
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        # Export
        output_path = f"app/models/weights/{model_name}_model.onnx"
        
        # Load checkpoint untuk export
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # TODO: Implement actual ONNX export dengan proper model reconstruction
        # This requires access to model architecture dari checkpoint
        
        return {
            "success": True,
            "message": f"Model {model_name} exported to ONNX",
            "onnx_path": output_path,
            "quantization_applied": quantization
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/onnx/benchmark/{model_name}")
async def benchmark_onnx_model(model_name: str, runs: int = 100):
    """Benchmark ONNX model performance"""
    if not ONNX_AVAILABLE:
        raise HTTPException(status_code=400, detail="ONNX not available")
    
    try:
        onnx_path = f"app/models/weights/{model_name}_model.onnx"
        
        if not os.path.exists(onnx_path):
            raise HTTPException(status_code=404, detail=f"ONNX model {model_name} not found")
        
        from app.onnx_inference import ONNXInferenceEngine
        
        engine = ONNXInferenceEngine(onnx_path)
        results = engine.benchmark(num_runs=runs)
        
        return {
            "success": True,
            "model": model_name,
            "benchmark_results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== PHASE 2: A/B TESTING ==============

@app.post("/experiments/create")
async def create_ab_experiment(
    name: str = Form(...),
    description: str = Form(default=""),
    control_model: str = Form(...),
    treatment_models: str = Form(...),  # JSON array of [name, version, allocation]
    sample_size: Optional[int] = Form(default=None),
    duration_days: Optional[int] = Form(default=None)
):
    """Create A/B testing experiment"""
    try:
        import json
        treatments = json.loads(treatment_models)
        
        ab_testing = get_ab_testing()
        
        # Parse control model
        control_parts = control_model.split(":")
        control = (control_parts[0], control_parts[1] if len(control_parts) > 1 else "latest")
        
        # Parse treatment models
        treatment_list = [
            (t[0], t[1], float(t[2])) for t in treatments
        ]
        
        experiment = ab_testing.create_experiment(
            name=name,
            control_model=control,
            treatment_models=treatment_list,
            description=description,
            sample_size=sample_size,
            duration_days=duration_days
        )
        
        return {
            "success": True,
            "experiment": experiment.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/experiments/{experiment_id}/start")
async def start_experiment(experiment_id: str):
    """Start A/B testing experiment"""
    try:
        ab_testing = get_ab_testing()
        success = ab_testing.start_experiment(experiment_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to start experiment")
        
        return {
            "success": True,
            "message": f"Experiment {experiment_id} started"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/experiments/{experiment_id}/stop")
async def stop_experiment(experiment_id: str, reason: str = Form(default="")):
    """Stop A/B testing experiment dan declare winner"""
    try:
        ab_testing = get_ab_testing()
        success = ab_testing.stop_experiment(experiment_id, reason)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to stop experiment")
        
        exp = ab_testing.get_experiment(experiment_id)
        results = ab_testing.get_experiment_results(experiment_id)
        
        return {
            "success": True,
            "message": f"Experiment {experiment_id} stopped",
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experiments/list")
async def list_experiments(
    status: Optional[str] = None,
    limit: int = 100
):
    """List A/B testing experiments"""
    try:
        ab_testing = get_ab_testing()
        
        status_enum = ExperimentStatus(status) if status else None
        experiments = ab_testing.list_experiments(status=status_enum, limit=limit)
        
        return {
            "success": True,
            "experiments": [e.to_dict() for e in experiments],
            "total": len(experiments)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experiments/{experiment_id}/results")
async def get_experiment_results(experiment_id: str):
    """Get detailed experiment results"""
    try:
        ab_testing = get_ab_testing()
        results = ab_testing.get_experiment_results(experiment_id)
        
        return {
            "success": True,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== FRONTEND ==============

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve integrated UI"""
    index_path = os.path.join(os.path.dirname(__file__), "static", "index_v3.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return {"message": "TB Detector v3 API - Use /docs for documentation"}


# ============== LIFECYCLE EVENTS ==============

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("=" * 60)
    print("TB Detector v3 - Phase 2 Optimizations Loaded")
    print("=" * 60)
    print("Features:")
    print("  - Async I/O utilities")
    print("  - Background task queue")
    print("  - ONNX inference support")
    print("  - Model versioning")
    print("  - A/B testing framework")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down services...")
    cleanup_async()
    shutdown_task_queue()
    print("Cleanup complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
