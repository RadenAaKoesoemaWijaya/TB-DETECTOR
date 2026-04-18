"""
Async I/O Utilities untuk TB Detector
Optimasi file operations dan model loading dengan asyncio
"""

import asyncio
import aiofiles
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import io
import functools
from concurrent.futures import ThreadPoolExecutor
import time


class AsyncFileIO:
    """
    Async file operations untuk I/O-bound tasks
    """
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def read_file(self, path: str, mode: str = 'rb') -> bytes:
        """Async file read"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: open(path, mode).read()
        )
    
    async def write_file(self, path: str, data: bytes, mode: str = 'wb'):
        """Async file write"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            lambda: open(path, mode).write(data)
        )
    
    async def read_audio(self, path: str) -> Optional[np.ndarray]:
        """
        Async audio file reading dengan librosa
        """
        import librosa
        
        loop = asyncio.get_event_loop()
        
        def _load():
            try:
                waveform, sr = librosa.load(path, sr=16000, mono=True)
                return waveform
            except Exception:
                return None
        
        return await loop.run_in_executor(self.executor, _load)
    
    async def load_model_async(self, model_path: str, map_location: str = 'cpu') -> Optional[Dict]:
        """
        Async model loading
        """
        loop = asyncio.get_event_loop()
        
        def _load():
            try:
                return torch.load(model_path, map_location=map_location)
            except Exception as e:
                print(f"Error loading model: {e}")
                return None
        
        return await loop.run_in_executor(self.executor, _load)
    
    async def save_model_async(self, model_path: str, state_dict: Dict):
        """
        Async model saving
        """
        loop = asyncio.get_event_loop()
        
        def _save():
            try:
                torch.save(state_dict, model_path)
                return True
            except Exception as e:
                print(f"Error saving model: {e}")
                return False
        
        return await loop.run_in_executor(self.executor, _save)
    
    async def batch_read_files(self, paths: List[str], max_concurrent: int = 10) -> List[Optional[bytes]]:
        """
        Batch async file reading dengan concurrency limit
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _read_with_limit(path):
            async with semaphore:
                try:
                    return await self.read_file(path)
                except Exception:
                    return None
        
        tasks = [_read_with_limit(p) for p in paths]
        return await asyncio.gather(*tasks)
    
    def close(self):
        """Cleanup executor"""
        self.executor.shutdown(wait=False)


class AsyncInferenceEngine:
    """
    Async inference engine dengan model pooling
    """
    
    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.model_cache = {}
        self.warmup_models = set()
    
    async def predict_async(
        self,
        components: Dict[str, Any],
        audio_features: torch.Tensor,
        metadata: Dict[str, Any],
        device: str
    ) -> Dict[str, float]:
        """
        Async prediction
        """
        loop = asyncio.get_event_loop()
        
        def _predict():
            metadata_encoder = components['metadata_encoder']
            feature_fusion = components['feature_fusion']
            classifier = components['classifier']
            
            with torch.no_grad():
                metadata_emb = metadata_encoder.encode(metadata).to(device)
                fused = feature_fusion(audio_features, metadata_emb)
                logits = classifier(fused)
                probs = torch.softmax(logits, dim=1)
                tb_prob = probs[0][1].item()
            
            return {
                'tb_probability': tb_prob,
                'risk_level': 'RENDAH' if tb_prob < 0.3 else ('MENENGAH' if tb_prob < 0.7 else 'TINGGI')
            }
        
        return await loop.run_in_executor(self.executor, _predict)
    
    def close(self):
        """Cleanup"""
        self.executor.shutdown(wait=False)


class AsyncPreprocessing:
    """
    Async preprocessing untuk batch audio processing
    """
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.segmenter = None
        self.sample_rate = 16000
    
    async def preprocess_audio(
        self,
        audio_path: str,
        preprocessor,
        segmenter
    ) -> Optional[np.ndarray]:
        """
        Async single audio preprocessing
        """
        import librosa
        
        loop = asyncio.get_event_loop()
        
        def _process():
            try:
                # Load
                waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
                
                # Segment
                segments = segmenter.segment(waveform)
                
                if len(segments) > 0:
                    start, end = max(segments, key=lambda x: x[1] - x[0])
                    cough_audio = waveform[start:end]
                else:
                    cough_audio = waveform
                
                # Pad/truncate ke 5 detik
                max_len = 5 * self.sample_rate
                if len(cough_audio) < max_len:
                    cough_audio = np.pad(cough_audio, (0, max_len - len(cough_audio)))
                else:
                    cough_audio = cough_audio[:max_len]
                
                return cough_audio
            except Exception as e:
                print(f"Error preprocessing {audio_path}: {e}")
                return None
        
        return await loop.run_in_executor(self.executor, _process)
    
    async def batch_preprocess(
        self,
        audio_paths: List[str],
        preprocessor,
        segmenter,
        max_concurrent: int = 8,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[Optional[np.ndarray]]:
        """
        Batch async preprocessing dengan progress tracking
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        completed = 0
        total = len(audio_paths)
        
        async def _process_with_limit(path):
            nonlocal completed
            async with semaphore:
                result = await self.preprocess_audio(path, preprocessor, segmenter)
                completed += 1
                if progress_callback and completed % 10 == 0:
                    progress = int((completed / total) * 100)
                    progress_callback(progress, f"Preprocessed {completed}/{total}")
                return result
        
        tasks = [_process_with_limit(p) for p in audio_paths]
        return await asyncio.gather(*tasks)
    
    def close(self):
        """Cleanup"""
        self.executor.shutdown(wait=False)


# Global async instances
_async_io = None
_async_inference = None
_async_preprocessing = None


def get_async_io(max_workers: int = 4) -> AsyncFileIO:
    """Get atau create global AsyncFileIO instance"""
    global _async_io
    if _async_io is None:
        _async_io = AsyncFileIO(max_workers=max_workers)
    return _async_io


def get_async_inference(max_workers: int = 2) -> AsyncInferenceEngine:
    """Get atau create global AsyncInferenceEngine instance"""
    global _async_inference
    if _async_inference is None:
        _async_inference = AsyncInferenceEngine(max_workers=max_workers)
    return _async_inference


def get_async_preprocessing(max_workers: int = 4) -> AsyncPreprocessing:
    """Get atau create global AsyncPreprocessing instance"""
    global _async_preprocessing
    if _async_preprocessing is None:
        _async_preprocessing = AsyncPreprocessing(max_workers=max_workers)
    return _async_preprocessing


def cleanup_async():
    """Cleanup all async resources"""
    global _async_io, _async_inference, _async_preprocessing
    
    if _async_io:
        _async_io.close()
        _async_io = None
    
    if _async_inference:
        _async_inference.close()
        _async_inference = None
    
    if _async_preprocessing:
        _async_preprocessing.close()
        _async_preprocessing = None
