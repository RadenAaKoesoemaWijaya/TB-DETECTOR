"""
Feature Cache Manager
Efficient caching untuk pre-extracted audio features
"""

import os
import json
import hashlib
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import pickle
import time


class FeatureCacheManager:
    """
    Manages caching untuk extracted audio features
    Menghindari re-extraction yang mahal
    """
    
    def __init__(self, cache_dir: str = "data/feature_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Index file untuk tracking
        self.index_path = self.cache_dir / "cache_index.json"
        self.index = self._load_index()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        
    def _load_index(self) -> Dict[str, Any]:
        """Load cache index"""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Save cache index"""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _compute_hash(self, audio_path: str, backbone_name: str) -> str:
        """
        Compute unique hash untuk audio + backbone combination
        Menggunakan file content hash untuk validity check
        """
        # Get file stats untuk quick invalidation check
        stat = os.stat(audio_path)
        file_info = f"{audio_path}:{stat.st_size}:{stat.st_mtime}:{backbone_name}"
        
        # Jika file besar, hash sebagian content
        if stat.st_size > 10 * 1024 * 1024:  # > 10MB
            with open(audio_path, 'rb') as f:
                # Hash first and last 1MB untuk speed
                start = f.read(1024 * 1024)
                f.seek(-1024 * 1024, 2)
                end = f.read(1024 * 1024)
                content_hash = hashlib.md5(start + end).hexdigest()[:16]
        else:
            # Small file: hash full content
            with open(audio_path, 'rb') as f:
                content_hash = hashlib.md5(f.read()).hexdigest()[:16]
        
        return f"{backbone_name}_{content_hash}"
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get path untuk cache file"""
        # Organize by backbone untuk easier cleanup
        backbone = cache_key.split('_')[0]
        backbone_dir = self.cache_dir / backbone
        backbone_dir.mkdir(exist_ok=True)
        return backbone_dir / f"{cache_key}.pt"
    
    def get(
        self,
        audio_path: str,
        backbone_name: str
    ) -> Optional[torch.Tensor]:
        """
        Get cached features jika tersedia
        Returns: Tensor atau None jika cache miss
        """
        cache_key = self._compute_hash(audio_path, backbone_name)
        cache_path = self.get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                features = torch.load(cache_path, map_location='cpu')
                self.hits += 1
                
                # Update access time di index
                if cache_key in self.index:
                    self.index[cache_key]['last_access'] = time.time()
                    self.index[cache_key]['access_count'] = self.index[cache_key].get('access_count', 0) + 1
                    self._save_index()
                
                return features
            except Exception as e:
                # Cache corrupted, remove
                print(f"Cache corrupted for {cache_key}, removing: {e}")
                cache_path.unlink(missing_ok=True)
                if cache_key in self.index:
                    del self.index[cache_key]
        
        self.misses += 1
        return None
    
    def set(
        self,
        audio_path: str,
        backbone_name: str,
        features: torch.Tensor,
        metadata: Optional[Dict] = None
    ):
        """
        Save features ke cache
        """
        cache_key = self._compute_hash(audio_path, backbone_name)
        cache_path = self.get_cache_path(cache_key)
        
        # Save features
        torch.save(features, cache_path)
        
        # Update index
        self.index[cache_key] = {
            'audio_path': audio_path,
            'backbone_name': backbone_name,
            'created': time.time(),
            'last_access': time.time(),
            'access_count': 1,
            'file_size': cache_path.stat().st_size,
            'feature_shape': list(features.shape),
            'metadata': metadata or {}
        }
        self._save_index()
    
    def batch_get_or_extract(
        self,
        audio_paths: List[str],
        backbone_name: str,
        backbone,
        device: str = 'cpu',
        show_progress: bool = True
    ) -> List[torch.Tensor]:
        """
        Batch operation: get from cache atau extract untuk multiple files
        Returns list of features dalam urutan yang sama dengan input
        """
        from tqdm import tqdm
        
        features_list = []
        to_extract = []  # (index, path) tuples
        
        # Phase 1: Check cache
        for i, path in enumerate(audio_paths):
            cached = self.get(path, backbone_name)
            if cached is not None:
                features_list.append((i, cached))
            else:
                to_extract.append((i, path))
                features_list.append((i, None))  # Placeholder
        
        # Phase 2: Extract untuk cache misses
        if to_extract:
            iterator = tqdm(to_extract, desc=f"Extracting {backbone_name}") if show_progress else to_extract
            
            backbone.model.eval()
            with torch.no_grad():
                for idx, path in iterator:
                    try:
                        import librosa
                        waveform, sr = librosa.load(path, sr=16000, mono=True)
                        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
                        
                        # Pad/truncate ke 5 detik
                        max_len = 5 * 16000
                        if len(waveform) < max_len:
                            waveform = np.pad(waveform, (0, max_len - len(waveform)))
                        else:
                            waveform = waveform[:max_len]
                        
                        waveform_tensor = torch.tensor(waveform, dtype=torch.float32).to(device)
                        features = backbone.extract_features(waveform_tensor)
                        
                        # Cache it
                        self.set(path, backbone_name, features.cpu())
                        
                        # Update di list
                        features_list[idx] = (idx, features.cpu().squeeze(0))
                    except Exception as e:
                        print(f"Error extracting {path}: {e}")
                        # Keep None untuk error handling di atas
        
        # Sort by index dan return features only
        features_list.sort(key=lambda x: x[0])
        return [f for _, f in features_list]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = len(self.index)
        total_size = sum(
            info.get('file_size', 0)
            for info in self.index.values()
        )
        
        # Group by backbone
        backbone_stats = {}
        for key, info in self.index.items():
            backbone = info.get('backbone_name', 'unknown')
            if backbone not in backbone_stats:
                backbone_stats[backbone] = {'count': 0, 'size': 0}
            backbone_stats[backbone]['count'] += 1
            backbone_stats[backbone]['size'] += info.get('file_size', 0)
        
        return {
            'total_entries': total_entries,
            'total_size_mb': total_size / (1024 * 1024),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            'by_backbone': backbone_stats
        }
    
    def clear_backbone(self, backbone_name: str):
        """Clear all cache untuk specific backbone"""
        keys_to_remove = [
            k for k, v in self.index.items()
            if v.get('backbone_name') == backbone_name
        ]
        
        for key in keys_to_remove:
            cache_path = self.get_cache_path(key)
            cache_path.unlink(missing_ok=True)
            del self.index[key]
        
        self._save_index()
        print(f"Cleared {len(keys_to_remove)} cache entries for {backbone_name}")
    
    def clear_all(self):
        """Clear entire cache"""
        import shutil
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index = {}
        self._save_index()
        print("Cache cleared")
    
    def cleanup_old(self, max_age_days: int = 30):
        """Remove cache entries yang tidak diakses dalam X hari"""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        keys_to_remove = []
        for key, info in self.index.items():
            last_access = info.get('last_access', 0)
            if current_time - last_access > max_age_seconds:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            cache_path = self.get_cache_path(key)
            cache_path.unlink(missing_ok=True)
            del self.index[key]
        
        self._save_index()
        print(f"Cleaned up {len(keys_to_remove)} old cache entries")
        return len(keys_to_remove)


# Global cache instance
_feature_cache = None


def get_feature_cache(cache_dir: str = "data/feature_cache") -> FeatureCacheManager:
    """Get atau create global cache instance"""
    global _feature_cache
    if _feature_cache is None:
        _feature_cache = FeatureCacheManager(cache_dir)
    return _feature_cache
