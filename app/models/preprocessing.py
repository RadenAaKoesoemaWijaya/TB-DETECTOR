"""
Audio Preprocessing & Cough Segmentation
Phase 1: Data Preparation & Preprocessing
"""

import numpy as np
import librosa
import torch
from typing import List, Tuple
import webrtcvad
import collections


class AudioPreprocessor:
    """
    Standarisasi Audio sesuai diagram:
    - Resampling 16kHz
    - Normalisasi Amplitudo
    """
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
    
    def load_and_standardize(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio dan standardisasi ke 16kHz"""
        waveform, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
        return waveform, sr
    
    def normalize_amplitude(self, waveform: np.ndarray) -> np.ndarray:
        """Normalisasi amplitudo ke range [-1, 1]"""
        max_amp = np.max(np.abs(waveform))
        if max_amp > 0:
            return waveform / max_amp
        return waveform
    
    def preprocess(self, file_path: str) -> np.ndarray:
        """Full preprocessing pipeline"""
        waveform, _ = self.load_and_standardize(file_path)
        waveform = self.normalize_amplitude(waveform)
        return waveform
    
    def preprocess_waveform(self, waveform: np.ndarray, orig_sr: int) -> np.ndarray:
        """Preprocessing dari waveform numpy"""
        # Resample if needed
        if orig_sr != self.target_sr:
            waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=self.target_sr)
        # Convert to mono if stereo
        if len(waveform.shape) > 1:
            waveform = librosa.to_mono(waveform)
        # Normalize
        waveform = self.normalize_amplitude(waveform)
        return waveform


class CoughSegmenter:
    """
    Segmentasi Batuk Otomatis (Cough Detection)
    Menggunakan Voice Activity Detection (VAD) + Energy-based detection
    """
    
    def __init__(self, sample_rate: int = 16000, frame_duration_ms: int = 30):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # WebRTC VAD (expects 16kHz, 30ms frames)
        self.vad = webrtcvad.Vad(2)  # Aggressiveness 0-3, higher = more aggressive filtering
        
        # Energy threshold for cough detection
        self.energy_threshold = 0.01
        self.min_cough_duration_sec = 0.1  # Minimum cough duration
        self.max_cough_duration_sec = 2.0  # Maximum cough duration
    
    def frame_generator(self, waveform: np.ndarray):
        """Generate frames for VAD"""
        n = len(waveform)
        offset = 0
        while offset + self.frame_size < n:
            yield waveform[offset:offset + self.frame_size]
            offset += self.frame_size
    
    def compute_frame_energy(self, frame: np.ndarray) -> float:
        """Compute RMS energy of frame"""
        return np.sqrt(np.mean(frame ** 2))
    
    def is_cough_candidate(self, frame: np.ndarray) -> bool:
        """Determine if frame is a cough candidate"""
        # Convert to bytes for VAD
        frame_bytes = (frame * 32767).astype(np.int16).tobytes()
        
        # Check VAD
        try:
            is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
        except:
            is_speech = False
        
        # Check energy
        energy = self.compute_frame_energy(frame)
        is_high_energy = energy > self.energy_threshold
        
        # Cough typically has high energy and is speech-like
        return is_speech and is_high_energy
    
    def segment(self, waveform: np.ndarray) -> List[Tuple[int, int]]:
        """
        Segment audio into cough regions
        Returns: List of (start_sample, end_sample) tuples
        """
        # Ensure int16 for VAD
        if waveform.dtype != np.int16:
            waveform_int16 = (waveform * 32767).astype(np.int16)
        else:
            waveform_int16 = waveform
            waveform = waveform.astype(np.float32) / 32767.0
        
        # Frame-level analysis
        frames = list(self.frame_generator(waveform_int16))
        num_frames = len(frames)
        
        if num_frames == 0:
            return []
        
        # Detect active regions
        is_active = []
        for frame in frames:
            frame_float = frame.astype(np.float32) / 32767.0
            is_active.append(self.is_cough_candidate(frame_float))
        
        # Group consecutive active frames into segments
        segments = []
        start_frame = None
        
        for i, active in enumerate(is_active):
            if active and start_frame is None:
                start_frame = i
            elif not active and start_frame is not None:
                # End of segment
                end_frame = i
                duration_sec = (end_frame - start_frame) * self.frame_duration_ms / 1000
                
                if self.min_cough_duration_sec <= duration_sec <= self.max_cough_duration_sec:
                    start_sample = start_frame * self.frame_size
                    end_sample = min(end_frame * self.frame_size, len(waveform))
                    segments.append((start_sample, end_sample))
                
                start_frame = None
        
        # Handle case where cough continues to end
        if start_frame is not None:
            end_frame = num_frames
            duration_sec = (end_frame - start_frame) * self.frame_duration_ms / 1000
            
            if self.min_cough_duration_sec <= duration_sec <= self.max_cough_duration_sec:
                start_sample = start_frame * self.frame_size
                end_sample = min(end_frame * self.frame_size, len(waveform))
                segments.append((start_sample, end_sample))
        
        # If no VAD-based segments, use energy-based fallback
        if len(segments) == 0:
            segments = self._energy_based_segmentation(waveform)
        
        return segments
    
    def _energy_based_segmentation(self, waveform: np.ndarray) -> List[Tuple[int, int]]:
        """Fallback: Energy-based cough detection"""
        hop_length = 512
        frame_length = 2048
        
        # Compute RMS energy
        rms = librosa.feature.rms(
            y=waveform, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        # Threshold
        threshold = np.mean(rms) + 0.5 * np.std(rms)
        
        # Find regions above threshold
        is_active = rms > threshold
        
        # Group into segments
        segments = []
        min_frames = int(self.min_cough_duration_sec * self.sample_rate / hop_length)
        
        start_idx = None
        for i, active in enumerate(is_active):
            if active and start_idx is None:
                start_idx = i
            elif not active and start_idx is not None:
                if i - start_idx >= min_frames:
                    start_sample = start_idx * hop_length
                    end_sample = min(i * hop_length, len(waveform))
                    segments.append((start_sample, end_sample))
                start_idx = None
        
        if start_idx is not None and len(is_active) - start_idx >= min_frames:
            start_sample = start_idx * hop_length
            end_sample = len(waveform)
            segments.append((start_sample, end_sample))
        
        return segments
    
    def extract_cough_events(self, waveform: np.ndarray) -> List[np.ndarray]:
        """Extract cough audio segments"""
        segments = self.segment(waveform)
        return [waveform[start:end] for start, end in segments]


class DataAugmentation:
    """
    Data Augmentation untuk Training
    - Noise injection
    - Pitch shifting
    - Time stretching
    """
    
    @staticmethod
    def add_noise(waveform: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """Add random noise"""
        noise = np.random.randn(len(waveform))
        augmented = waveform + noise_factor * noise
        return augmented / (np.max(np.abs(augmented)) + 1e-8)
    
    @staticmethod
    def pitch_shift(waveform: np.ndarray, sr: int, n_steps: float = 2.0) -> np.ndarray:
        """Shift pitch"""
        return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)
    
    @staticmethod
    def time_stretch(waveform: np.ndarray, rate: float = 1.1) -> np.ndarray:
        """Stretch time (speed up/down)"""
        stretched = librosa.effects.time_stretch(waveform, rate=rate)
        # Pad or truncate to original length
        if len(stretched) > len(waveform):
            stretched = stretched[:len(waveform)]
        else:
            stretched = np.pad(stretched, (0, len(waveform) - len(stretched)))
        return stretched
    
    @staticmethod
    def augment(waveform: np.ndarray, sr: int, aug_types: List[str] = None) -> np.ndarray:
        """Apply random augmentation"""
        if aug_types is None:
            aug_types = ['noise', 'pitch', 'stretch']
        
        aug_type = np.random.choice(aug_types)
        
        if aug_type == 'noise':
            noise_factor = np.random.uniform(0.001, 0.01)
            return DataAugmentation.add_noise(waveform, noise_factor)
        elif aug_type == 'pitch':
            n_steps = np.random.uniform(-3, 3)
            return DataAugmentation.pitch_shift(waveform, sr, n_steps)
        elif aug_type == 'stretch':
            rate = np.random.uniform(0.8, 1.2)
            return DataAugmentation.time_stretch(waveform, rate)
        
        return waveform
