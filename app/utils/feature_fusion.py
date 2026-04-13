"""
Feature Fusion Module
Gabungan Embedding Audio + Metadata Klinis Pasien
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    """
    Fusikan fitur audio (dari Wav2Vec2) dengan metadata klinis
    Menggunakan concatenation + projection
    """
    
    def __init__(
        self, 
        audio_dim: int = 1024,      # Wav2Vec2 large output
        metadata_dim: int = 32,      # Metadata embedding
        fused_dim: int = 512,       # Fused output dimension
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.metadata_dim = metadata_dim
        self.fused_dim = fused_dim
        
        # Projection layers untuk setiap modality
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.metadata_proj = nn.Sequential(
            nn.Linear(metadata_dim, fused_dim // 4),
            nn.LayerNorm(fused_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fused_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim + fused_dim // 4, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim)
        )
        
        self.norm = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, audio_features: torch.Tensor, metadata_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: [batch_size, audio_dim]
            metadata_embedding: [batch_size, metadata_dim]
        Returns:
            fused_features: [batch_size, fused_dim]
        """
        # Project each modality
        audio_proj = self.audio_proj(audio_features)           # [B, fused_dim]
        metadata_proj = self.metadata_proj(metadata_embedding)   # [B, fused_dim//4]
        
        # Add sequence dimension for attention
        audio_seq = audio_proj.unsqueeze(1)                    # [B, 1, fused_dim]
        
        # Cross-modal attention (audio attends to audio - self attention for refinement)
        attn_out, _ = self.cross_attention(audio_seq, audio_seq, audio_seq)
        audio_attended = self.norm(audio_seq + self.dropout(attn_out)).squeeze(1)
        
        # Concatenate audio and metadata
        combined = torch.cat([audio_attended, metadata_proj], dim=-1)
        
        # Final fusion
        fused = self.fusion_mlp(combined)
        
        return fused


class SimpleConcatFusion(nn.Module):
    """
    Simpler fusion: just concatenate and project
    """
    
    def __init__(
        self,
        audio_dim: int = 1024,
        metadata_dim: int = 32,
        fused_dim: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()
        
        total_dim = audio_dim + metadata_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fused_dim * 2),
            nn.LayerNorm(fused_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, audio_features: torch.Tensor, metadata_embedding: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([audio_features, metadata_embedding], dim=-1)
        return self.fusion(combined)


class GatedFusion(nn.Module):
    """
    Gated fusion: learn gates untuk mengontribusi setiap modality
    """
    
    def __init__(
        self,
        audio_dim: int = 1024,
        metadata_dim: int = 32,
        fused_dim: int = 512
    ):
        super().__init__()
        
        self.audio_gate = nn.Sequential(
            nn.Linear(audio_dim + metadata_dim, audio_dim),
            nn.Sigmoid()
        )
        
        self.metadata_gate = nn.Sequential(
            nn.Linear(audio_dim + metadata_dim, metadata_dim),
            nn.Sigmoid()
        )
        
        self.projection = nn.Sequential(
            nn.Linear(audio_dim + metadata_dim, fused_dim * 2),
            nn.ReLU(),
            nn.Linear(fused_dim * 2, fused_dim)
        )
    
    def forward(self, audio_features: torch.Tensor, metadata_embedding: torch.Tensor) -> torch.Tensor:
        # Concatenate for gate computation
        combined = torch.cat([audio_features, metadata_embedding], dim=-1)
        
        # Compute gates
        audio_gate = self.audio_gate(combined)
        metadata_gate = self.metadata_gate(combined)
        
        # Apply gates
        gated_audio = audio_features * audio_gate
        gated_metadata = metadata_embedding * metadata_gate
        
        # Concatenate and project
        fused = torch.cat([gated_audio, gated_metadata], dim=-1)
        output = self.projection(fused)
        
        return output
