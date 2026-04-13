"""
Multi-Backbone Training & Model Comparison
Train and compare: Google HeAR, Wav2Vec 2.0, XLS-R
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix,
    classification_report, roc_curve
)
from tqdm import tqdm
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from app.models.backbones import BackboneFactory, get_backbone_dim
from app.models.classifier import TBClassifier, get_tb_loss_fn
from app.utils.feature_fusion import SimpleConcatFusion
from app.utils.metadata_encoder import SimpleMetadataEncoder

# Set seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {DEVICE}")

# Constants
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 5 * SAMPLE_RATE
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 10


class TBCoughDataset(Dataset):
    """Dataset dengan caching features"""
    
    def __init__(self, data_df, audio_dir, backbone, augment=False):
        self.data = data_df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.backbone = backbone
        self.augment = augment
        
        self.audio_features = []
        self.labels = []
        self.metadata_list = []
        
        self._preprocess_all()
    
    def _preprocess_all(self):
        print("Preprocessing dataset...")
        
        for idx in tqdm(range(len(self.data))):
            row = self.data.iloc[idx]
            
            try:
                audio_path = os.path.join(self.audio_dir, row['audio_filename'])
                
                # Load audio
                waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
                
                # Pad/truncate
                if len(waveform) < MAX_AUDIO_LENGTH:
                    waveform = np.pad(waveform, (0, MAX_AUDIO_LENGTH - len(waveform)))
                else:
                    waveform = waveform[:MAX_AUDIO_LENGTH]
                
                # Convert to tensor
                waveform_tensor = torch.tensor(waveform, dtype=torch.float32).to(DEVICE)
                
                # Extract features
                with torch.no_grad():
                    features = self.backbone.extract_features(waveform_tensor)
                    features = features.cpu().squeeze(0)
                
                self.audio_features.append(features)
                
                # Metadata
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
                
                self.metadata_list.append(metadata)
                self.labels.append(int(row['tb_label']))
                
            except Exception as e:
                print(f"Error processing {row.get('audio_filename', 'unknown')}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.audio_features)} samples")
    
    def __len__(self):
        return len(self.audio_features)
    
    def __getitem__(self, idx):
        return {
            'audio_features': self.audio_features[idx],
            'metadata': self.metadata_list[idx],
            'label': self.labels[idx]
        }


def collate_fn(batch):
    audio_features = torch.stack([item['audio_features'] for item in batch])
    metadata = [item['metadata'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    return {
        'audio_features': audio_features,
        'metadata': metadata,
        'labels': labels
    }


def train_epoch(model_components, dataloader, optimizer, criterion, device):
    """Train satu epoch"""
    feature_fusion = model_components['feature_fusion']
    classifier = model_components['classifier']
    metadata_encoder = model_components['metadata_encoder']
    
    feature_fusion.train()
    classifier.train()
    metadata_encoder.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        audio_features = batch['audio_features'].to(device)
        metadata_list = batch['metadata']
        labels = batch['labels'].to(device)
        
        # Encode metadata
        metadata_embeddings = []
        for meta in metadata_list:
            emb = metadata_encoder.encode(meta)
            metadata_embeddings.append(emb)
        metadata_embeddings = torch.cat(metadata_embeddings, dim=0).to(device)
        
        # Forward
        fused_features = feature_fusion(audio_features, metadata_embeddings)
        logits = classifier(fused_features)
        
        # Loss
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def evaluate(model_components, dataloader, criterion, device):
    """Evaluasi model"""
    feature_fusion = model_components['feature_fusion']
    classifier = model_components['classifier']
    metadata_encoder = model_components['metadata_encoder']
    
    feature_fusion.eval()
    classifier.eval()
    metadata_encoder.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            audio_features = batch['audio_features'].to(device)
            metadata_list = batch['metadata']
            labels = batch['labels'].to(device)
            
            metadata_embeddings = []
            for meta in metadata_list:
                emb = metadata_encoder.encode(meta)
                metadata_embeddings.append(emb)
            metadata_embeddings = torch.cat(metadata_embeddings, dim=0).to(device)
            
            fused_features = feature_fusion(audio_features, metadata_embeddings)
            logits = classifier(fused_features)
            probs = torch.softmax(logits, dim=1)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Metrics
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
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
        'auroc': auroc,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def train_single_model(backbone_name, data_df, audio_dir, output_dir, epochs=EPOCHS):
    """Train model dengan satu backbone"""
    
    print(f"\n{'='*60}")
    print(f"Training with backbone: {backbone_name.upper()}")
    print(f"{'='*60}\n")
    
    # Load backbone
    backbone = BackboneFactory.create(backbone_name, DEVICE)
    audio_dim = backbone.output_dim
    
    # Data split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(data_df, groups=data_df['patient_id']))
    
    train_df = data_df.iloc[train_idx].copy()
    test_df = data_df.iloc[test_idx].copy()
    
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_idx_final, val_idx = next(gss_val.split(train_df, groups=train_df['patient_id']))
    
    train_final_df = train_df.iloc[train_idx_final].copy()
    val_df = train_df.iloc[val_idx].copy()
    
    print(f"Data split - Train: {len(train_final_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets dengan caching
    train_dataset = TBCoughDataset(train_final_df, audio_dir, backbone, augment=False)
    val_dataset = TBCoughDataset(val_df, audio_dir, backbone, augment=False)
    test_dataset = TBCoughDataset(test_df, audio_dir, backbone, augment=False)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Initialize models
    metadata_encoder = SimpleMetadataEncoder(embedding_dim=32)
    feature_fusion = SimpleConcatFusion(
        audio_dim=audio_dim, metadata_dim=32, fused_dim=512
    )
    classifier = TBClassifier(
        input_dim=512, hidden_dims=[256, 128], num_heads=4, dropout=0.3
    )
    
    metadata_encoder.to(DEVICE)
    feature_fusion.to(DEVICE)
    classifier.to(DEVICE)
    
    # Optimizer
    params = (
        list(metadata_encoder.parameters()) +
        list(feature_fusion.parameters()) +
        list(classifier.parameters())
    )
    optimizer = optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    criterion = get_tb_loss_fn(pos_weight=2.0)
    
    model_components = {
        'metadata_encoder': metadata_encoder,
        'feature_fusion': feature_fusion,
        'classifier': classifier
    }
    
    # Training loop
    best_auroc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_metrics': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        train_loss, train_acc = train_epoch(
            model_components, train_loader, optimizer, criterion, DEVICE
        )
        
        val_metrics = evaluate(model_components, val_loader, criterion, DEVICE)
        
        scheduler.step(val_metrics['auroc'])
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val AUROC: {val_metrics['auroc']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_metrics'].append(val_metrics)
        
        # Save best
        if val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            patience_counter = 0
            
            checkpoint = {
                'backbone_name': backbone_name,
                'backbone_dim': audio_dim,
                'metadata_encoder': metadata_encoder.state_dict(),
                'feature_fusion': feature_fusion.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch,
                'auroc': best_auroc,
                'history': history
            }
            
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, f'{backbone_name}_model.pth')
            torch.save(checkpoint, model_path)
            print(f"Saved best model (AUROC: {best_auroc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Final test evaluation
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION - {backbone_name.upper()}")
    print(f"{'='*60}")
    
    checkpoint = torch.load(os.path.join(output_dir, f'{backbone_name}_model.pth'))
    metadata_encoder.load_state_dict(checkpoint['metadata_encoder'])
    feature_fusion.load_state_dict(checkpoint['feature_fusion'])
    classifier.load_state_dict(checkpoint['classifier'])
    
    test_metrics = evaluate(model_components, test_loader, criterion, DEVICE)
    
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"Test Specificity: {test_metrics['specificity']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1']:.4f}")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f'{backbone_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    return {
        'backbone': backbone_name,
        'metrics': test_metrics,
        'model_path': model_path,
        'history': history
    }


def compare_models(results: list, output_dir: str):
    """Compare dan visualisasi hasil training"""
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}\n")
    
    comparison = []
    for result in results:
        m = result['metrics']
        comparison.append({
            'Backbone': result['backbone'],
            'Accuracy': m['accuracy'],
            'Precision': m['precision'],
            'Sensitivity': m['sensitivity'],
            'Specificity': m['specificity'],
            'F1-Score': m['f1'],
            'AUROC': m['auroc']
        })
    
    df = pd.DataFrame(comparison)
    print(df.to_string(index=False))
    
    # Save comparison
    comparison_path = os.path.join(output_dir, 'model_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Comparison - TB Detection', fontsize=14, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'AUROC']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(df['Backbone'], df[metric], color=['#667eea', '#764ba2', '#f093fb', '#4facfe'])
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.set_title(metric)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_path}")
    
    # Find best model
    best_idx = df['AUROC'].idxmax()
    best_model = df.iloc[best_idx]
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model['Backbone'].upper()}")
    print(f"AUROC: {best_model['AUROC']:.4f}")
    print(f"{'='*60}")
    
    # Save best model info
    best_info = {
        'best_backbone': best_model['Backbone'],
        'metrics': best_model.to_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'best_model.json'), 'w') as f:
        json.dump(best_info, f, indent=2)


def main():
    """Main training dengan multiple backbones"""
    
    # Load data
    print("Loading dataset...")
    data_df = pd.read_csv('data/coda_tb_dataset.csv')
    audio_dir = 'data/audio'
    
    output_dir = 'app/models/weights'
    os.makedirs(output_dir, exist_ok=True)
    
    # Backbones to compare
    backbones = [
        'wav2vec2-base',      # Wav2Vec 2.0 Base - 768d
        'wav2vec2-xlsr',      # XLS-R Large - 1024d (Multilingual)
        'hear',               # Google HeAR style - 1024d
        # 'hubert-base',      # Optional: HuBERT Base - 768d
    ]
    
    print(f"\nWill train {len(backbones)} models for comparison:")
    for b in backbones:
        print(f"  - {b}")
    
    # Train each model
    results = []
    for backbone_name in backbones:
        try:
            result = train_single_model(backbone_name, data_df, audio_dir, output_dir, epochs=EPOCHS)
            results.append(result)
        except Exception as e:
            print(f"Error training {backbone_name}: {e}")
            continue
    
    # Compare results
    if len(results) > 1:
        compare_models(results, output_dir)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"Models saved in: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
