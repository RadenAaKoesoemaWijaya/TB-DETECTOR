"""
Batch Training Module for TB Detection
Efficient batched training with DataLoader support
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime

from app.models.backbones import BackboneFactory
from app.models.classifier import TBClassifier, get_tb_loss_fn
from app.utils.feature_fusion import SimpleConcatFusion
from app.utils.metadata_encoder import SimpleMetadataEncoder


class TBCoughFeatureDataset(Dataset):
    """
    Dataset untuk pre-extracted features (batched)
    Mendukung caching dan lazy loading
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        device: str = 'cpu',
        cache_dir: Optional[str] = None
    ):
        self.data = data
        self.device = device
        self.cache_dir = cache_dir
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        
        # Convert to tensors jika belum
        audio_features = sample['audio_features']
        if not isinstance(audio_features, torch.Tensor):
            audio_features = torch.tensor(audio_features, dtype=torch.float32)
        
        label = sample['label']
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        
        return {
            'audio_features': audio_features,
            'metadata': sample['metadata'],
            'label': label,
            'audio_filename': sample.get('audio_filename', f'sample_{idx}')
        }


def collate_features(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function untuk batching
    Stack audio features dan labels, keep metadata as list
    """
    audio_features = torch.stack([item['audio_features'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    metadata = [item['metadata'] for item in batch]
    audio_filenames = [item['audio_filename'] for item in batch]
    
    return {
        'audio_features': audio_features,
        'metadata': metadata,
        'labels': labels,
        'audio_filenames': audio_filenames
    }


class BatchTrainer:
    """
    Efficient batch trainer dengan DataLoader support
    """
    
    def __init__(
        self,
        backbone_name: str,
        audio_dim: int,
        device: str = 'cpu',
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        pos_weight: float = 2.0
    ):
        self.backbone_name = backbone_name
        self.audio_dim = audio_dim
        self.device = device
        
        # Initialize components
        self.metadata_encoder = SimpleMetadataEncoder(embedding_dim=32).to(device)
        self.feature_fusion = SimpleConcatFusion(
            audio_dim=audio_dim,
            metadata_dim=32,
            fused_dim=512
        ).to(device)
        self.classifier = TBClassifier(
            input_dim=512,
            hidden_dims=[256, 128],
            num_heads=4,
            dropout=0.3
        ).to(device)
        
        # Optimizer dengan parameter groups untuk fine control
        self.optimizer = optim.AdamW([
            {'params': self.metadata_encoder.parameters(), 'lr': learning_rate},
            {'params': self.feature_fusion.parameters(), 'lr': learning_rate},
            {'params': self.classifier.parameters(), 'lr': learning_rate * 2}  # Classifier bisa lebih cepat
        ], weight_decay=weight_decay)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss function
        self.criterion = get_tb_loss_fn(pos_weight=pos_weight)
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_auroc': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        self.best_auroc = 0.0
        self.patience_counter = 0
        
    def encode_metadata_batch(self, metadata_list: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Encode batch metadata menggunakan metadata_encoder
        """
        embeddings = []
        for metadata in metadata_list:
            emb = self.metadata_encoder.encode(metadata)
            embeddings.append(emb)
        
        # Stack menjadi batch tensor [B, 32]
        return torch.cat(embeddings, dim=0).to(self.device)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train satu epoch dengan batch processing
        """
        self.metadata_encoder.train()
        self.feature_fusion.train()
        self.classifier.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Training {self.backbone_name}", leave=False)
        
        for batch in pbar:
            audio_features = batch['audio_features'].to(self.device)
            metadata_list = batch['metadata']
            labels = batch['labels'].to(self.device)
            
            batch_size = labels.size(0)
            
            # Forward pass
            metadata_embeddings = self.encode_metadata_batch(metadata_list)
            fused = self.feature_fusion(audio_features, metadata_embeddings)
            logits = self.classifier(fused)
            
            # Loss
            loss = self.criterion(logits, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping untuk stabilitas
            torch.nn.utils.clip_grad_norm_(
                list(self.metadata_encoder.parameters()) +
                list(self.feature_fusion.parameters()) +
                list(self.classifier.parameters()),
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{(correct/total):.4f}'
            })
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate dengan batch processing
        """
        self.metadata_encoder.eval()
        self.feature_fusion.eval()
        self.classifier.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                audio_features = batch['audio_features'].to(self.device)
                metadata_list = batch['metadata']
                labels = batch['labels'].to(self.device)
                
                batch_size = labels.size(0)
                
                # Forward
                metadata_embeddings = self.encode_metadata_batch(metadata_list)
                fused = self.feature_fusion(audio_features, metadata_embeddings)
                logits = self.classifier(fused)
                probs = torch.softmax(logits, dim=1)
                
                # Loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item() * batch_size
                
                # Metrics
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                total += batch_size
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )
        
        avg_loss = total_loss / total
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
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1': f1,
            'auroc': auroc,
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
    
    def train(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        test_data: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 16,
        epochs: int = 30,
        patience: int = 10,
        num_workers: int = 0,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Full training loop dengan batch processing
        """
        # Create datasets
        train_dataset = TBCoughFeatureDataset(train_data, device=self.device)
        val_dataset = TBCoughFeatureDataset(val_data, device=self.device)
        
        # Create dataloaders dengan proper batching
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_features,
            num_workers=num_workers,
            pin_memory=True if self.device == 'cuda' else False,
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,  # Larger batch for eval
            shuffle=False,
            collate_fn=collate_features,
            num_workers=num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        print(f"\n{'='*60}")
        print(f"Training {self.backbone_name} with Batch Size: {batch_size}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print(f"{'='*60}\n")
        
        best_auroc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            # Scheduler step
            self.scheduler.step(val_metrics['auroc'])
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auroc'].append(val_metrics['auroc'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Progress logging
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val AUROC: {val_metrics['auroc']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # Progress callback untuk UI
            if progress_callback:
                progress = int((epoch + 1) / epochs * 100)
                progress_callback(progress, f"Epoch {epoch+1}/{epochs} - AUROC: {val_metrics['auroc']:.4f}")
            
            # Early stopping check
            if val_metrics['auroc'] > best_auroc:
                best_auroc = val_metrics['auroc']
                patience_counter = 0
                
                # Save checkpoint
                self.best_auroc = best_auroc
                self.best_state = {
                    'metadata_encoder': self.metadata_encoder.state_dict(),
                    'feature_fusion': self.feature_fusion.state_dict(),
                    'classifier': self.classifier.state_dict(),
                    'epoch': epoch,
                    'auroc': best_auroc
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Final test evaluation jika tersedia
        test_metrics = None
        if test_data:
            test_dataset = TBCoughFeatureDataset(test_data, device=self.device)
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size * 2,
                shuffle=False,
                collate_fn=collate_features
            )
            
            # Load best model
            if hasattr(self, 'best_state'):
                self.metadata_encoder.load_state_dict(self.best_state['metadata_encoder'])
                self.feature_fusion.load_state_dict(self.best_state['feature_fusion'])
                self.classifier.load_state_dict(self.best_state['classifier'])
            
            test_metrics = self.evaluate(test_loader)
            print(f"\nTest Results - AUROC: {test_metrics['auroc']:.4f}, F1: {test_metrics['f1']:.4f}")
        
        return {
            'backbone': self.backbone_name,
            'best_auroc': best_auroc,
            'history': self.history,
            'test_metrics': test_metrics,
            'epochs_trained': len(self.history['train_loss'])
        }
    
    def get_best_state(self) -> Dict[str, Any]:
        """Get best model state dicts"""
        return self.best_state if hasattr(self, 'best_state') else None
    
    def save_checkpoint(self, path: str, metadata: Dict[str, Any] = None):
        """Save model checkpoint"""
        checkpoint = {
            'backbone_name': self.backbone_name,
            'backbone_dim': self.audio_dim,
            'metadata_encoder': self.metadata_encoder.state_dict(),
            'feature_fusion': self.feature_fusion.state_dict(),
            'classifier': self.classifier.state_dict(),
            'history': self.history,
            'best_auroc': self.best_auroc
        }
        if metadata:
            checkpoint.update(metadata)
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")
