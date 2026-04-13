"""
Model Manager - Handle multiple trained models
Load, compare, and select best model for inference
"""

import os
import json
import torch
import glob
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from app.models.backbones import BackboneFactory, get_backbone_dim
from app.models.classifier import TBClassifier
from app.utils.feature_fusion import SimpleConcatFusion
from app.utils.metadata_encoder import SimpleMetadataEncoder


@dataclass
class ModelInfo:
    """Model information container"""
    name: str
    backbone: str
    backbone_dim: int
    path: str
    metrics: Dict
    is_best: bool = False
    timestamp: Optional[str] = None


class ModelManager:
    """
    Manages multiple trained models
    Supports loading different backbones and comparing performance
    """
    
    def __init__(self, weights_dir: str = "app/models/weights"):
        self.weights_dir = weights_dir
        self.models: Dict[str, ModelInfo] = {}
        self.current_model: Optional[str] = None
        
        self.components = {
            'metadata_encoder': None,
            'feature_fusion': None,
            'classifier': None,
            'backbone_name': None,
            'backbone_dim': None
        }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._scan_models()
    
    def _scan_models(self):
        """Scan directory for available models"""
        if not os.path.exists(self.weights_dir):
            return

        # Find all model files
        model_files = glob.glob(os.path.join(self.weights_dir, "*_model.pth"))

        for model_path in model_files:
            name = os.path.basename(model_path).replace("_model.pth", "")
            metrics_path = model_path.replace("_model.pth", "_metrics.json")
            package_path = model_path.replace("_model.pth", "_package.json")

            metrics = {}

            # Try to load metrics from file
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            elif os.path.exists(package_path):
                with open(package_path, 'r') as f:
                    package = json.load(f)
                    metrics = package.get('metrics', {})

            # Load checkpoint to get info
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                backbone = checkpoint.get('backbone_name', 'unknown')
                backbone_dim = checkpoint.get('backbone_dim', 768)

                # Try to get metrics from checkpoint if not found in files
                if not metrics and 'metrics' in checkpoint:
                    metrics = checkpoint.get('metrics', {})

                # Get timestamp from checkpoint if available
                timestamp = None
                if 'timestamp' in checkpoint:
                    timestamp = checkpoint['timestamp']
                elif os.path.exists(package_path):
                    with open(package_path, 'r') as f:
                        package = json.load(f)
                        timestamp = package.get('saved_at')

                self.models[name] = ModelInfo(
                    name=name,
                    backbone=backbone,
                    backbone_dim=backbone_dim,
                    path=model_path,
                    metrics=metrics,
                    timestamp=timestamp
                )
            except Exception as e:
                print(f"Error loading model info for {name}: {e}")

        # Check for best model designation
        best_model_file = os.path.join(self.weights_dir, "best_model.json")
        if os.path.exists(best_model_file):
            with open(best_model_file, 'r') as f:
                best_info = json.load(f)
                best_name = best_info.get('best_backbone', '')
                if best_name in self.models:
                    self.models[best_name].is_best = True

        print(f"Found {len(self.models)} models:")
        for name, info in self.models.items():
            best_marker = " (BEST)" if info.is_best else ""
            auroc = info.metrics.get('auroc', 0)
            print(f"  - {name}: {info.backbone} (AUROC: {auroc:.4f}){best_marker}")
    
    def load_model(self, model_name: str) -> bool:
        """Load a specific model for inference"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return False
        
        model_info = self.models[model_name]
        
        try:
            checkpoint = torch.load(model_info.path, map_location=self.device)
            
            # Initialize components
            metadata_encoder = SimpleMetadataEncoder(embedding_dim=32)
            feature_fusion = SimpleConcatFusion(
                audio_dim=model_info.backbone_dim,
                metadata_dim=32,
                fused_dim=512
            )
            classifier = TBClassifier(
                input_dim=512,
                hidden_dims=[256, 128],
                num_heads=4,
                dropout=0.3
            )
            
            # Load weights
            metadata_encoder.load_state_dict(checkpoint['metadata_encoder'])
            feature_fusion.load_state_dict(checkpoint['feature_fusion'])
            classifier.load_state_dict(checkpoint['classifier'])
            
            # Move to device and set eval mode
            metadata_encoder.to(self.device)
            feature_fusion.to(self.device)
            classifier.to(self.device)
            
            metadata_encoder.eval()
            feature_fusion.eval()
            classifier.eval()
            
            # Store components
            self.components['metadata_encoder'] = metadata_encoder
            self.components['feature_fusion'] = feature_fusion
            self.components['classifier'] = classifier
            self.components['backbone_name'] = model_info.backbone
            self.components['backbone_dim'] = model_info.backbone_dim
            
            self.current_model = model_name
            
            print(f"Loaded model: {model_name} ({model_info.backbone})")
            return True
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return False
    
    def load_best_model(self) -> bool:
        """Load the best performing model"""
        best_model = None
        best_auroc = 0
        
        for name, info in self.models.items():
            if info.is_best:
                return self.load_model(name)
            auroc = info.metrics.get('auroc', 0)
            if auroc > best_auroc:
                best_auroc = auroc
                best_model = name
        
        if best_model:
            return self.load_model(best_model)
        
        # Fallback: load any available model
        if self.models:
            return self.load_model(list(self.models.keys())[0])
        
        return False
    
    def get_components(self) -> Dict:
        """Get current model components"""
        return self.components
    
    def list_models(self) -> List[Dict]:
        """List all available models with their info"""
        return [
            {
                'name': name,
                'backbone': info.backbone,
                'backbone_dim': info.backbone_dim,
                'metrics': info.metrics,
                'is_best': info.is_best,
                'path': info.path
            }
            for name, info in self.models.items()
        ]
    
    def compare_models(self) -> Dict:
        """Get comparison of all models"""
        comparison = []
        
        for name, info in self.models.items():
            m = info.metrics
            comparison.append({
                'name': name,
                'backbone': info.backbone,
                'accuracy': m.get('accuracy', 0),
                'precision': m.get('precision', 0),
                'sensitivity': m.get('sensitivity', 0),
                'specificity': m.get('specificity', 0),
                'f1_score': m.get('f1', 0),
                'auroc': m.get('auroc', 0),
                'is_best': info.is_best
            })
        
        # Sort by AUROC
        comparison.sort(key=lambda x: x['auroc'], reverse=True)
        
        return {
            'models': comparison,
            'total_models': len(comparison),
            'best_model': comparison[0]['name'] if comparison else None
        }
    
    def get_current_model_info(self) -> Optional[Dict]:
        """Get info about currently loaded model"""
        if not self.current_model:
            return None
        
        info = self.models.get(self.current_model)
        if not info:
            return None
        
        return {
            'name': info.name,
            'backbone': info.backbone,
            'backbone_dim': info.backbone_dim,
            'metrics': info.metrics,
            'is_best': info.is_best
        }
    
    def is_ready(self) -> bool:
        """Check if a model is loaded and ready"""
        return (
            self.components['classifier'] is not None and
            self.components['feature_fusion'] is not None and
            self.components['metadata_encoder'] is not None
        )


# Global model manager instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get or create model manager singleton"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
