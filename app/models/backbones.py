"""
Multi-Backbone Feature Extractors
Supports: Google HeAR, Wav2Vec 2.0, XLS-R
"""

import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2Processor, Wav2Vec2Model,
    AutoProcessor, AutoModel,
    AutoFeatureExtractor
)
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


class BaseBackbone(nn.Module):
    """Base class untuk audio feature extractors"""
    
    def __init__(self, model_name: str, sample_rate: int = 16000):
        super().__init__()
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.output_dim = None
        self.processor = None
        self.model = None
        
    def extract_features(self, audio_input: torch.Tensor) -> torch.Tensor:
        """Extract features from audio input"""
        raise NotImplementedError
    
    def freeze(self):
        """Freeze backbone parameters"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze backbone parameters"""
        for param in self.parameters():
            param.requires_grad = True


class Wav2Vec2Backbone(BaseBackbone):
    """
    Wav2Vec 2.0 Base Model
    768-dim hidden states
    """
    
    def __init__(self, device: str = 'cpu'):
        super().__init__('facebook/wav2vec2-base-960h', sample_rate=16000)
        self.device = device
        
        print(f"Loading Wav2Vec 2.0 Base...")
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2Model.from_pretrained(self.model_name)
        self.model.to(device)
        self.model.eval()
        
        self.output_dim = 768
        self.freeze()
        print(f"  Output dim: {self.output_dim}")
    
    def extract_features(self, audio_input: torch.Tensor) -> torch.Tensor:
        """
        audio_input: [batch_size, samples] atau [samples]
        returns: [batch_size, 768]
        """
        if audio_input.dim() == 1:
            audio_input = audio_input.unsqueeze(0)
        
        # Process through processor
        inputs = self.processor(
            audio_input.squeeze(0).cpu().numpy(), 
            sampling_rate=self.sample_rate, 
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_values)
            # Mean pooling over time
            features = outputs.last_hidden_state.mean(dim=1)
        
        return features


class Wav2Vec2XLSRBackbone(BaseBackbone):
    """
    Wav2Vec 2.0 XLS-R Large (53 languages)
    1024-dim hidden states - Multilingual
    """
    
    def __init__(self, device: str = 'cpu'):
        super().__init__('facebook/wav2vec2-large-xlsr-53', sample_rate=16000)
        self.device = device
        
        print(f"Loading Wav2Vec 2.0 XLS-R Large...")
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2Model.from_pretrained(self.model_name)
        self.model.to(device)
        self.model.eval()
        
        self.output_dim = 1024
        self.freeze()
        print(f"  Output dim: {self.output_dim}")
    
    def extract_features(self, audio_input: torch.Tensor) -> torch.Tensor:
        if audio_input.dim() == 1:
            audio_input = audio_input.unsqueeze(0)
        
        inputs = self.processor(
            audio_input.squeeze(0).cpu().numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_values)
            features = outputs.last_hidden_state.mean(dim=1)
        
        return features


class GoogleHeARBackbone(BaseBackbone):
    """
    Google Health Acoustic Representations (HeAR)
    Model untuk representasi akustik kesehatan
    """
    
    def __init__(self, device: str = 'cpu'):
        # HeAR menggunakan model yang mirip dengan wav2vec-based
        # atau bisa menggunakan model spesifik dari Google
        super().__init__('google/hear', sample_rate=16000)
        self.device = device
        
        print(f"Loading Google HeAR (using Wav2Vec2-Large as proxy)...")
        print("  Note: Google HeAR official model may require special access")
        
        # Gunakan Wav2Vec2-Large sebagai proxy untuk HeAR-style
        # atau bisa ganti dengan model HeAR yang sebenarnya jika tersedia
        self.model_name = 'facebook/wav2vec2-large-960h-lv60-self'
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2Model.from_pretrained(self.model_name)
        self.model.to(device)
        self.model.eval()
        
        self.output_dim = 1024
        self.freeze()
        print(f"  Output dim: {self.output_dim}")
    
    def extract_features(self, audio_input: torch.Tensor) -> torch.Tensor:
        if audio_input.dim() == 1:
            audio_input = audio_input.unsqueeze(0)
        
        inputs = self.processor(
            audio_input.squeeze(0).cpu().numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_values)
            features = outputs.last_hidden_state.mean(dim=1)
        
        return features


class HuBERTBackbone(BaseBackbone):
    """
    Facebook HuBERT Model
    Alternative untuk speech representation
    """
    
    def __init__(self, device: str = 'cpu', size: str = 'base'):
        if size == 'large':
            model_name = 'facebook/hubert-large-ls960-ft'
            self.output_dim = 1024
        else:
            model_name = 'facebook/hubert-base-ls960'
            self.output_dim = 768
        
        super().__init__(model_name, sample_rate=16000)
        self.device = device
        
        print(f"Loading HuBERT {size}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        self.freeze()
        print(f"  Output dim: {self.output_dim}")
    
    def extract_features(self, audio_input: torch.Tensor) -> torch.Tensor:
        if audio_input.dim() == 1:
            audio_input = audio_input.unsqueeze(0)
        
        inputs = self.processor(
            audio_input.squeeze(0).cpu().numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_values)
            features = outputs.last_hidden_state.mean(dim=1)
        
        return features


class BackboneFactory:
    """Factory untuk membuat backbone instances"""
    
    BACKBONES = {
        'wav2vec2': Wav2Vec2Backbone,
        'wav2vec2-base': Wav2Vec2Backbone,
        'xlsr': Wav2Vec2XLSRBackbone,
        'wav2vec2-xlsr': Wav2Vec2XLSRBackbone,
        'wav2vec2-large-xlsr-53': Wav2Vec2XLSRBackbone,
        'hear': GoogleHeARBackbone,
        'google-hear': GoogleHeARBackbone,
        'hubert-base': lambda device: HuBERTBackbone(device, 'base'),
        'hubert-large': lambda device: HuBERTBackbone(device, 'large'),
    }
    
    @classmethod
    def create(cls, backbone_name: str, device: str = 'cpu') -> BaseBackbone:
        """Create backbone instance"""
        backbone_name = backbone_name.lower().replace('_', '-')
        
        if backbone_name not in cls.BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone_name}. Available: {list(cls.BACKBONES.keys())}")
        
        return cls.BACKBONES[backbone_name](device)
    
    @classmethod
    def list_available(cls) -> list:
        """List available backbones"""
        return list(cls.BACKBONES.keys())


# Feature dimension mapping
BACKBONE_DIMS = {
    'wav2vec2': 768,
    'wav2vec2-base': 768,
    'xlsr': 1024,
    'wav2vec2-xlsr': 1024,
    'wav2vec2-large-xlsr-53': 1024,
    'hear': 1024,
    'google-hear': 1024,
    'hubert-base': 768,
    'hubert-large': 1024,
}


def get_backbone_dim(backbone_name: str) -> int:
    """Get output dimension for backbone"""
    backbone_name = backbone_name.lower().replace('_', '-')
    return BACKBONE_DIMS.get(backbone_name, 768)
