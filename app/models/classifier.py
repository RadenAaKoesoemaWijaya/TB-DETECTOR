"""
TB Binary Classifier Architecture
Phase 2: Model Development - MLP + Transformer Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class TransformerHead(nn.Module):
    """
    Transformer-based classification head
    Menggunakan multi-head attention untuk capture pattern
    """
    
    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        
        # Ensure input_dim is divisible by num_heads
        self.head_dim = input_dim // num_heads
        self.embed_dim = self.head_dim * num_heads
        
        # Projection if needed
        self.proj = nn.Linear(input_dim, self.embed_dim) if input_dim != self.embed_dim else nn.Identity()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim * 2, self.embed_dim)
        )
        
        # Layer norm
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, input_dim]
        """
        # Project if needed
        x = self.proj(x)
        
        # Add sequence dimension for attention [batch, 1, embed_dim]
        x_seq = x.unsqueeze(1)
        
        # Multi-head attention
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        x_seq = self.norm1(x_seq + self.dropout(attn_out))
        
        # Feed-forward
        ffn_out = self.ffn(x_seq)
        x_seq = self.norm2(x_seq + self.dropout(ffn_out))
        
        # Remove sequence dimension
        return x_seq.squeeze(1)


class TBClassifier(nn.Module):
    """
    Binary Classifier untuk deteksi TB
    Arsitektur: MLP + Transformer Head
    """
    
    def __init__(
        self, 
        input_dim: int = 512,
        hidden_dims: List[int] = [256, 128],
        num_heads: int = 4,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Transformer head
        self.transformer_head = TransformerHead(
            input_dim=hidden_dims[-1],
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Final classifier
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        x: [batch_size, input_dim] - fused features from audio + metadata
        """
        # MLP layers
        x = self.mlp(x)
        
        # Transformer head
        x = self.transformer_head(x)
        
        # Final classification
        logits = self.classifier(x)
        
        return logits
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate embeddings (for analysis)"""
        x = self.mlp(x)
        x = self.transformer_head(x)
        return x


class AttentionMLPClassifier(nn.Module):
    """
    Alternative: Attention-based MLP classifier (simpler version)
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        
        # Feature extraction layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.Tanh(),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.feature_extractor(x)
        
        # Attention weights
        attn_weights = F.softmax(self.attention(features), dim=0)
        
        # Weighted features
        weighted = features * attn_weights
        
        # Classify
        logits = self.classifier(weighted)
        
        return logits


# Loss function dengan class weighting untuk handle imbalance
def get_tb_loss_fn(pos_weight: float = 2.0):
    """
    Weighted Cross Entropy Loss untuk TB detection
    (TB positive biasanya lebih sedikit dari negative)
    """
    weights = torch.tensor([1.0, pos_weight])
    
    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=weights.to(logits.device))
    
    return loss_fn
