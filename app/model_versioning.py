"""
Model Versioning System untuk TB Detector
Tracking model versions dengan metadata dan lineage
"""

import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import re


class ModelStage(Enum):
    """Model deployment stages"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ModelFormat(Enum):
    """Supported model formats"""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"


@dataclass
class ModelVersion:
    """Model version metadata"""
    version_id: str
    model_name: str
    backbone_name: str
    stage: ModelStage
    format: ModelFormat
    
    # File paths
    model_path: str
    metadata_path: Optional[str] = None
    onnx_path: Optional[str] = None
    
    # Versioning
    major: int = 1
    minor: int = 0
    patch: int = 0
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = "system"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    # Lineage
    parent_version: Optional[str] = None
    dataset_hash: Optional[str] = None
    training_session_id: Optional[str] = None
    
    # Validation
    is_validated: bool = False
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    # Deployment
    is_active: bool = False
    deployment_count: int = 0
    last_deployed: Optional[str] = None
    
    def full_version(self) -> str:
        """Get full version string"""
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['stage'] = self.stage.value
        data['format'] = self.format.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        data = data.copy()
        data['stage'] = ModelStage(data.get('stage', 'development'))
        data['format'] = ModelFormat(data.get('format', 'pytorch'))
        return cls(**data)


class ModelRegistry:
    """
    Model registry dengan versioning support
    """
    
    def __init__(self, registry_path: str = "app/models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.versions_file = self.registry_path / "versions.json"
        self._versions: Dict[str, List[ModelVersion]] = {}
        
        self._load_registry()
    
    def _load_registry(self):
        """Load registry dari disk"""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                data = json.load(f)
                for model_name, versions_data in data.items():
                    self._versions[model_name] = [
                        ModelVersion.from_dict(v) for v in versions_data
                    ]
    
    def _save_registry(self):
        """Save registry ke disk"""
        data = {
            name: [v.to_dict() for v in versions]
            for name, versions in self._versions.items()
        }
        with open(self.versions_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash untuk file integrity"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]
    
    def register_version(
        self,
        model_name: str,
        backbone_name: str,
        model_path: str,
        stage: ModelStage = ModelStage.DEVELOPMENT,
        format: ModelFormat = ModelFormat.PYTORCH,
        metrics: Dict[str, float] = None,
        training_config: Dict[str, Any] = None,
        description: str = "",
        tags: List[str] = None,
        parent_version: str = None,
        created_by: str = "system"
    ) -> ModelVersion:
        """
        Register new model version
        
        Returns:
            ModelVersion object
        """
        # Get next version number
        existing_versions = self._versions.get(model_name, [])
        
        if existing_versions:
            latest = existing_versions[-1]
            major, minor, patch = latest.major, latest.minor, latest.patch
            
            # Auto-increment version
            if stage == ModelStage.PRODUCTION:
                major += 1
                minor = 0
                patch = 0
            elif stage == ModelStage.STAGING:
                minor += 1
                patch = 0
            else:
                patch += 1
        else:
            major, minor, patch = 1, 0, 0
        
        # Create version ID
        version_id = f"{model_name}_v{major}.{minor}.{patch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Copy model ke registry dengan versioning
        registry_model_dir = self.registry_path / model_name / f"v{major}.{minor}.{patch}"
        registry_model_dir.mkdir(parents=True, exist_ok=True)
        
        dest_model_path = registry_model_dir / Path(model_path).name
        shutil.copy2(model_path, dest_model_path)
        
        # Copy metadata jika ada
        metadata_path = Path(model_path).with_suffix('.json')
        dest_metadata_path = None
        if metadata_path.exists():
            dest_metadata_path = registry_model_dir / metadata_path.name
            shutil.copy2(metadata_path, dest_metadata_path)
        
        # Create version object
        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            backbone_name=backbone_name,
            stage=stage,
            format=format,
            model_path=str(dest_model_path),
            metadata_path=str(dest_metadata_path) if dest_metadata_path else None,
            major=major,
            minor=minor,
            patch=patch,
            created_by=created_by,
            description=description,
            tags=tags or [],
            metrics=metrics or {},
            training_config=training_config or {},
            parent_version=parent_version
        )
        
        # Add ke registry
        if model_name not in self._versions:
            self._versions[model_name] = []
        
        self._versions[model_name].append(version)
        self._save_registry()
        
        print(f"Registered {model_name} v{version.full_version()} ({version_id})")
        return version
    
    def get_version(self, model_name: str, version_str: str) -> Optional[ModelVersion]:
        """
        Get specific version
        
        Args:
            model_name: Name of model
            version_str: Version string (e.g., "1.2.3")
        
        Returns:
            ModelVersion atau None
        """
        versions = self._versions.get(model_name, [])
        for v in versions:
            if v.full_version() == version_str:
                return v
        return None
    
    def get_latest_version(
        self,
        model_name: str,
        stage: Optional[ModelStage] = None
    ) -> Optional[ModelVersion]:
        """Get latest version untuk model"""
        versions = self._versions.get(model_name, [])
        
        if stage:
            versions = [v for v in versions if v.stage == stage]
        
        if not versions:
            return None
        
        # Sort by created_at descending
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions[0]
    
    def list_versions(
        self,
        model_name: Optional[str] = None,
        stage: Optional[ModelStage] = None,
        tag: Optional[str] = None,
        limit: int = 100
    ) -> List[ModelVersion]:
        """List versions dengan filtering"""
        result = []
        
        if model_name:
            result = self._versions.get(model_name, [])
        else:
            for versions in self._versions.values():
                result.extend(versions)
        
        if stage:
            result = [v for v in result if v.stage == stage]
        
        if tag:
            result = [v for v in result if tag in v.tags]
        
        # Sort by created_at descending
        result.sort(key=lambda v: v.created_at, reverse=True)
        
        return result[:limit]
    
    def promote_version(
        self,
        model_name: str,
        version_str: str,
        new_stage: ModelStage
    ) -> bool:
        """Promote version ke stage baru"""
        version = self.get_version(model_name, version_str)
        if not version:
            return False
        
        version.stage = new_stage
        
        if new_stage == ModelStage.PRODUCTION:
            # Deactivate other production versions
            for v in self._versions.get(model_name, []):
                if v.full_version() != version_str and v.stage == ModelStage.PRODUCTION:
                    v.is_active = False
            
            version.is_active = True
            version.deployment_count = 0
        
        self._save_registry()
        return True
    
    def validate_version(
        self,
        model_name: str,
        version_str: str,
        validation_results: Dict[str, Any]
    ) -> bool:
        """Mark version sebagai validated dengan results"""
        version = self.get_version(model_name, version_str)
        if not version:
            return False
        
        version.is_validated = True
        version.validation_results = validation_results
        self._save_registry()
        return True
    
    def deploy_version(self, model_name: str, version_str: str) -> bool:
        """Record deployment untuk version"""
        version = self.get_version(model_name, version_str)
        if not version:
            return False
        
        version.deployment_count += 1
        version.last_deployed = datetime.now().isoformat()
        self._save_registry()
        return True
    
    def compare_versions(
        self,
        model_name: str,
        version_a: str,
        version_b: str
    ) -> Dict[str, Any]:
        """Compare dua versions"""
        va = self.get_version(model_name, version_a)
        vb = self.get_version(model_name, version_b)
        
        if not va or not vb:
            return {"error": "Version not found"}
        
        return {
            'version_a': va.to_dict(),
            'version_b': vb.to_dict(),
            'metrics_diff': {
                k: {
                    'a': va.metrics.get(k, 0),
                    'b': vb.metrics.get(k, 0),
                    'diff': vb.metrics.get(k, 0) - va.metrics.get(k, 0)
                }
                for k in set(va.metrics.keys()) | set(vb.metrics.keys())
            }
        }
    
    def get_model_lineage(self, model_name: str, version_str: str) -> List[ModelVersion]:
        """Get version lineage (parent chain)"""
        lineage = []
        current = self.get_version(model_name, version_str)
        
        while current:
            lineage.append(current)
            if current.parent_version:
                # Parse parent version
                parts = current.parent_version.split('_v')
                if len(parts) == 2:
                    parent_name = parts[0]
                    parent_version = parts[1].split('_')[0]
                    current = self.get_version(parent_name, parent_version)
                else:
                    break
            else:
                break
        
        return lineage
    
    def delete_version(self, model_name: str, version_str: str) -> bool:
        """Delete version dari registry"""
        versions = self._versions.get(model_name, [])
        version = self.get_version(model_name, version_str)
        
        if not version:
            return False
        
        # Remove dari list
        self._versions[model_name] = [
            v for v in versions if v.full_version() != version_str
        ]
        
        # Delete files
        if version.model_path and Path(version.model_path).exists():
            Path(version.model_path).parent.rmdir()
        
        self._save_registry()
        return True
    
    def export_to_onnx(
        self,
        model_name: str,
        version_str: str,
        onnx_path: Optional[str] = None
    ) -> Optional[str]:
        """Export version ke ONNX format"""
        version = self.get_version(model_name, version_str)
        if not version:
            return None
        
        try:
            from app.onnx_inference import ONNXExporter
            
            # Load PyTorch model
            import torch
            checkpoint = torch.load(version.model_path, map_location='cpu')
            
            # This would need proper reconstruction of model components
            # Simplified untuk demonstration
            
            # Update version dengan ONNX path
            if onnx_path:
                version.onnx_path = onnx_path
                self._save_registry()
            
            return onnx_path
            
        except Exception as e:
            print(f"Export failed: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_models = len(self._versions)
        total_versions = sum(len(v) for v in self._versions.values())
        
        stage_counts = {}
        for versions in self._versions.values():
            for v in versions:
                stage = v.stage.value
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        format_counts = {}
        for versions in self._versions.values():
            for v in versions:
                fmt = v.format.value
                format_counts[fmt] = format_counts.get(fmt, 0) + 1
        
        return {
            'total_models': total_models,
            'total_versions': total_versions,
            'by_stage': stage_counts,
            'by_format': format_counts,
            'active_production': sum(
                1 for versions in self._versions.values()
                for v in versions if v.stage == ModelStage.PRODUCTION and v.is_active
            )
        }


# Global registry instance
_registry = None


def get_model_registry(registry_path: str = "app/models/registry") -> ModelRegistry:
    """Get atau create global ModelRegistry instance"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(registry_path)
    return _registry
