"""
Training module for TB Detector
Provides batched training with proper DataLoader support
"""

from .batch_trainer import BatchTrainer, TBCoughFeatureDataset, collate_features
from .cache_manager import FeatureCacheManager, get_feature_cache

__all__ = ['BatchTrainer', 'TBCoughFeatureDataset', 'collate_features', 'FeatureCacheManager', 'get_feature_cache']
