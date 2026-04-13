"""
TB Detection Models Module
"""

from .backbones import (
    BaseBackbone,
    Wav2Vec2Backbone,
    Wav2Vec2XLSRBackbone,
    GoogleHeARBackbone,
    HuBERTBackbone,
    BackboneFactory,
    get_backbone_dim
)
from .classifier import TBClassifier, TransformerHead, get_tb_loss_fn
from .preprocessing import (
    AudioPreprocessor,
    CoughSegmenter,
    DataAugmentation
)

__all__ = [
    'BaseBackbone',
    'Wav2Vec2Backbone',
    'Wav2Vec2XLSRBackbone',
    'GoogleHeARBackbone',
    'HuBERTBackbone',
    'BackboneFactory',
    'get_backbone_dim',
    'TBClassifier',
    'TransformerHead',
    'get_tb_loss_fn',
    'AudioPreprocessor',
    'CoughSegmenter',
    'DataAugmentation',
]
