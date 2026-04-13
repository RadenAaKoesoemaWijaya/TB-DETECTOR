"""
Metadata Encoder
Encode metadata klinis pasien menjadi embedding vector
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any


class MetadataEncoder(nn.Module):
    """
    Encode metadata klinis pasien:
    - Umur (continuous)
    - Jenis kelamin (binary)
    - Gejala (binary flags)
    """
    
    def __init__(self, embedding_dim: int = 32, num_symptoms: int = 7):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_symptoms = num_symptoms
        
        # Age embedding (bucketized)
        self.age_buckets = [0, 5, 12, 18, 30, 45, 60, 75, 100]
        self.age_embedding = nn.Embedding(len(self.age_buckets), 16)
        
        # Gender embedding
        self.gender_embedding = nn.Embedding(2, 8)  # 0: L, 1: P
        
        # Symptom embedding (each symptom has its own embedding)
        self.symptom_embeddings = nn.ModuleList([
            nn.Embedding(2, 4) for _ in range(num_symptoms)
        ])
        
        # Cough duration embedding (bucketized)
        self.duration_buckets = [0, 3, 7, 14, 21, 30, 100]
        self.duration_embedding = nn.Embedding(len(self.duration_buckets), 8)
        
        # Projection to final embedding dim
        total_dim = 16 + 8 + (num_symptoms * 4) + 8
        self.projection = nn.Sequential(
            nn.Linear(total_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
    
    def _bucketize(self, value: float, buckets: list) -> int:
        """Convert continuous value to bucket index"""
        for i, threshold in enumerate(buckets):
            if value < threshold:
                return i
        return len(buckets) - 1
    
    def encode_to_tensor(self, metadata: Any) -> torch.Tensor:
        """
        Encode metadata object to tensor
        """
        # Handle dict or object
        if isinstance(metadata, dict):
            age = metadata.get('age', 30)
            gender = 0 if metadata.get('gender', 'L') == 'L' else 1
            has_fever = int(metadata.get('has_fever', False))
            has_cough = int(metadata.get('has_cough', True))
            cough_duration = metadata.get('cough_duration_days', 0)
            has_night_sweats = int(metadata.get('has_night_sweats', False))
            has_weight_loss = int(metadata.get('has_weight_loss', False))
            has_chest_pain = int(metadata.get('has_chest_pain', False))
            has_shortness_breath = int(metadata.get('has_shortness_breath', False))
            previous_tb = int(metadata.get('previous_tb_history', False))
        else:
            age = getattr(metadata, 'age', 30)
            gender = 0 if getattr(metadata, 'gender', 'L') == 'L' else 1
            has_fever = int(getattr(metadata, 'has_fever', False))
            has_cough = int(getattr(metadata, 'has_cough', True))
            cough_duration = getattr(metadata, 'cough_duration_days', 0)
            has_night_sweats = int(getattr(metadata, 'has_night_sweats', False))
            has_weight_loss = int(getattr(metadata, 'has_weight_loss', False))
            has_chest_pain = int(getattr(metadata, 'has_chest_pain', False))
            has_shortness_breath = int(getattr(metadata, 'has_shortness_breath', False))
            previous_tb = int(getattr(metadata, 'previous_tb_history', False))
        
        # Bucketize age
        age_bucket = self._bucketize(age, self.age_buckets)
        
        # Bucketize duration
        duration_bucket = self._bucketize(cough_duration, self.duration_buckets)
        
        # Create tensors
        age_idx = torch.tensor([age_bucket], dtype=torch.long)
        gender_idx = torch.tensor([gender], dtype=torch.long)
        duration_idx = torch.tensor([duration_bucket], dtype=torch.long)
        
        symptoms = [
            has_fever, has_cough, has_night_sweats, has_weight_loss,
            has_chest_pain, has_shortness_breath, previous_tb
        ]
        symptom_indices = [torch.tensor([s], dtype=torch.long) for s in symptoms]
        
        return age_idx, gender_idx, symptom_indices, duration_idx
    
    def forward(self, metadata: Any) -> torch.Tensor:
        """
        Forward pass - encode metadata to embedding
        """
        age_idx, gender_idx, symptom_indices, duration_idx = self.encode_to_tensor(metadata)
        
        # Move to same device as model
        device = next(self.parameters()).device
        age_idx = age_idx.to(device)
        gender_idx = gender_idx.to(device)
        duration_idx = duration_idx.to(device)
        symptom_indices = [s.to(device) for s in symptom_indices]
        
        # Get embeddings
        age_emb = self.age_embedding(age_idx)             # [1, 16]
        gender_emb = self.gender_embedding(gender_idx)     # [1, 8]
        duration_emb = self.duration_embedding(duration_idx)  # [1, 8]
        
        symptom_embs = []
        for i, symp_idx in enumerate(symptom_indices):
            emb = self.symptom_embeddings[i](symp_idx)     # [1, 4]
            symptom_embs.append(emb)
        
        # Concatenate all embeddings
        all_embs = [age_emb, gender_emb] + symptom_embs + [duration_emb]
        combined = torch.cat(all_embs, dim=-1)            # [1, total_dim]
        
        # Project to final embedding
        output = self.projection(combined)                # [1, embedding_dim]
        
        return output
    
    def encode(self, metadata: Any) -> torch.Tensor:
        """Alias for forward"""
        return self.forward(metadata)


class SimpleMetadataEncoder(nn.Module):
    """
    Simpler version: direct numerical encoding with MLP
    """
    
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        
        # Input features: age, gender, cough_duration, 7 binary symptoms
        self.input_dim = 10
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )
    
    def encode_to_vector(self, metadata: Any) -> torch.Tensor:
        """Convert metadata to numerical vector"""
        if isinstance(metadata, dict):
            age = metadata.get('age', 30) / 100.0  # Normalize
            gender = 0.0 if metadata.get('gender', 'L') == 'L' else 1.0
            cough_duration = min(metadata.get('cough_duration_days', 0) / 30.0, 1.0)
            has_fever = float(metadata.get('has_fever', False))
            has_cough = float(metadata.get('has_cough', True))
            has_night_sweats = float(metadata.get('has_night_sweats', False))
            has_weight_loss = float(metadata.get('has_weight_loss', False))
            has_chest_pain = float(metadata.get('has_chest_pain', False))
            has_shortness_breath = float(metadata.get('has_shortness_breath', False))
            previous_tb = float(metadata.get('previous_tb_history', False))
        else:
            age = getattr(metadata, 'age', 30) / 100.0
            gender = 0.0 if getattr(metadata, 'gender', 'L') == 'L' else 1.0
            cough_duration = min(getattr(metadata, 'cough_duration_days', 0) / 30.0, 1.0)
            has_fever = float(getattr(metadata, 'has_fever', False))
            has_cough = float(getattr(metadata, 'has_cough', True))
            has_night_sweats = float(getattr(metadata, 'has_night_sweats', False))
            has_weight_loss = float(getattr(metadata, 'has_weight_loss', False))
            has_chest_pain = float(getattr(metadata, 'has_chest_pain', False))
            has_shortness_breath = float(getattr(metadata, 'has_shortness_breath', False))
            previous_tb = float(getattr(metadata, 'previous_tb_history', False))
        
        features = [
            age, gender, cough_duration, has_fever, has_cough,
            has_night_sweats, has_weight_loss, has_chest_pain,
            has_shortness_breath, previous_tb
        ]
        
        return torch.tensor([features], dtype=torch.float32)
    
    def forward(self, metadata: Any) -> torch.Tensor:
        x = self.encode_to_vector(metadata)
        device = next(self.parameters()).device
        x = x.to(device)
        return self.encoder(x)
    
    def encode(self, metadata: Any) -> torch.Tensor:
        return self.forward(metadata)
