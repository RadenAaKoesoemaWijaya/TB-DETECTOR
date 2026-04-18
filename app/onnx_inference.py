"""
ONNX Inference Optimization untuk TB Detector
Faster inference dengan ONNX Runtime
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import time
import warnings

# ONNX imports
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    warnings.warn("ONNX/ONNXRuntime not available. ONNX inference disabled.")


class ONNXInferenceEngine:
    """
    ONNX Runtime inference engine untuk TB Detection
    2-3x lebih cepat dari PyTorch untuk inference
    """
    
    def __init__(
        self,
        onnx_path: Optional[str] = None,
        providers: Optional[List[str]] = None
    ):
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available. Install with: pip install onnx onnxruntime")
        
        self.onnx_path = onnx_path
        self.session = None
        self.input_names = []
        self.output_names = []
        self.providers = providers or self._get_default_providers()
        
        # Metadata
        self.metadata = {}
        self.backbone_name = None
        
        if onnx_path:
            self.load_model(onnx_path)
    
    def _get_default_providers(self) -> List[str]:
        """Get default execution providers (GPU if available)"""
        providers = ['CPUExecutionProvider']
        
        # Check CUDA availability
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        
        return providers
    
    def load_model(self, onnx_path: str) -> bool:
        """
        Load ONNX model
        """
        try:
            # Session options untuk optimization
            sess_options = ort.SessionOptions()
            
            # Graph optimization level
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Enable all optimizations
            sess_options.enable_cpu_mem_arena = True
            sess_options.enable_mem_pattern = True
            
            # Load session
            self.session = ort.InferenceSession(
                onnx_path,
                sess_options,
                providers=self.providers
            )
            
            # Get input/output info
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            # Load metadata jika ada
            self._load_metadata(onnx_path)
            
            print(f"ONNX model loaded: {onnx_path}")
            print(f"  Providers: {self.session.get_providers()}")
            print(f"  Inputs: {self.input_names}")
            print(f"  Outputs: {self.output_names}")
            
            return True
            
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            return False
    
    def _load_metadata(self, onnx_path: str):
        """Load metadata dari ONNX model atau sidecar file"""
        import json
        
        # Try sidecar file
        metadata_path = Path(onnx_path).with_suffix('.metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.backbone_name = self.metadata.get('backbone_name', 'unknown')
        else:
            # Try load dari ONNX metadata
            try:
                model = onnx.load(onnx_path)
                # Check untuk metadata di model
                for prop in model.metadata_props:
                    if prop.key == 'backbone_name':
                        self.backbone_name = prop.value
            except Exception:
                pass
    
    def predict(
        self,
        audio_features: np.ndarray,
        metadata_features: np.ndarray
    ) -> Dict[str, float]:
        """
        Run inference dengan ONNX Runtime
        
        Args:
            audio_features: [1, audio_dim] numpy array
            metadata_features: [1, metadata_dim] numpy array
        
        Returns:
            Dictionary dengan probabilities dan risk level
        """
        if self.session is None:
            raise RuntimeError("No ONNX model loaded")
        
        # Prepare inputs
        inputs = {}
        
        # Map inputs berdasarkan nama
        for name in self.input_names:
            if 'audio' in name.lower() or 'feature' in name.lower():
                inputs[name] = audio_features.astype(np.float32)
            elif 'metadata' in name.lower():
                inputs[name] = metadata_features.astype(np.float32)
        
        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        
        # Get logits (assume first output)
        logits = outputs[0]
        
        # Softmax untuk probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        tb_prob = float(probs[0][1])
        
        # Risk level
        if tb_prob < 0.3:
            risk = "RENDAH"
        elif tb_prob < 0.7:
            risk = "MENENGAH"
        else:
            risk = "TINGGI"
        
        return {
            'tb_probability': tb_prob,
            'risk_level': risk,
            'logits': logits.tolist(),
            'probabilities': probs.tolist(),
            'model_type': 'onnx'
        }
    
    def predict_batch(
        self,
        audio_features: np.ndarray,
        metadata_features: np.ndarray
    ) -> List[Dict[str, float]]:
        """
        Batch prediction
        
        Args:
            audio_features: [batch_size, audio_dim]
            metadata_features: [batch_size, metadata_dim]
        
        Returns:
            List of prediction dictionaries
        """
        if self.session is None:
            raise RuntimeError("No ONNX model loaded")
        
        batch_size = audio_features.shape[0]
        
        # Prepare inputs
        inputs = {}
        for name in self.input_names:
            if 'audio' in name.lower() or 'feature' in name.lower():
                inputs[name] = audio_features.astype(np.float32)
            elif 'metadata' in name.lower():
                inputs[name] = metadata_features.astype(np.float32)
        
        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        logits = outputs[0]
        
        # Process each sample
        results = []
        for i in range(batch_size):
            exp_logits = np.exp(logits[i] - np.max(logits[i]))
            probs = exp_logits / np.sum(exp_logits)
            tb_prob = float(probs[1])
            
            if tb_prob < 0.3:
                risk = "RENDAH"
            elif tb_prob < 0.7:
                risk = "MENENGAH"
            else:
                risk = "TINGGI"
            
            results.append({
                'tb_probability': tb_prob,
                'risk_level': risk
            })
        
        return results
    
    def benchmark(self, num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark inference speed
        """
        if self.session is None:
            raise RuntimeError("No ONNX model loaded")
        
        # Dummy inputs
        audio_dim = 768 if 'base' in str(self.backbone_name) else 1024
        audio_features = np.random.randn(1, audio_dim).astype(np.float32)
        metadata_features = np.random.randn(1, 32).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            self.predict(audio_features, metadata_features)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.predict(audio_features, metadata_features)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert ke ms
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'fps': 1000.0 / np.mean(times)
        }


class ONNXExporter:
    """
    Export PyTorch models ke ONNX format
    """
    
    @staticmethod
    def export_model(
        model_components: Dict[str, torch.nn.Module],
        output_path: str,
        audio_dim: int = 768,
        metadata_dim: int = 32,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Export complete model (fusion + classifier) ke ONNX
        
        Args:
            model_components: Dict dengan 'feature_fusion', 'classifier'
            output_path: Path untuk output ONNX file
            audio_dim: Input audio dimension
            metadata_dim: Input metadata dimension
            metadata: Metadata untuk disimpan bersama model
        
        Returns:
            True jika berhasil
        """
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX not available")
        
        try:
            feature_fusion = model_components['feature_fusion']
            classifier = model_components['classifier']
            
            # Set ke eval mode
            feature_fusion.eval()
            classifier.eval()
            
            # Combined forward function
            class CombinedModel(torch.nn.Module):
                def __init__(self, fusion, clf):
                    super().__init__()
                    self.fusion = fusion
                    self.classifier = clf
                
                def forward(self, audio_features, metadata):
                    fused = self.fusion(audio_features, metadata)
                    logits = self.classifier(fused)
                    return logits
            
            combined = CombinedModel(feature_fusion, classifier)
            combined.eval()
            
            # Dummy inputs
            dummy_audio = torch.randn(1, audio_dim)
            dummy_metadata = torch.randn(1, metadata_dim)
            
            # Export
            torch.onnx.export(
                combined,
                (dummy_audio, dummy_metadata),
                output_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['audio_features', 'metadata_features'],
                output_names=['logits'],
                dynamic_axes={
                    'audio_features': {0: 'batch_size'},
                    'metadata_features': {0: 'batch_size'},
                    'logits': {0: 'batch_size'}
                }
            )
            
            # Verify
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            # Save metadata sebagai sidecar file
            if metadata:
                import json
                metadata_path = Path(output_path).with_suffix('.metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            print(f"Model exported to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    @staticmethod
    def quantize_model(
        onnx_path: str,
        output_path: str,
        quantization_mode: str = 'dynamic'
    ) -> bool:
        """
        Quantize ONNX model untuk faster inference
        
        Args:
            onnx_path: Input ONNX model path
            output_path: Output quantized model path
            quantization_mode: 'dynamic' atau 'static'
        """
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX not available")
        
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantize_dynamic(
                model_input=onnx_path,
                model_output=output_path,
                weight_type=QuantType.QInt8
            )
            
            print(f"Quantized model saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Quantization failed: {e}")
            return False


class ModelInferenceManager:
    """
    Manager untuk switching antara PyTorch dan ONNX inference
    """
    
    def __init__(self):
        self.pytorch_components = {}
        self.onnx_engine = None
        self.use_onnx = False
        self.model_name = None
    
    def load_pytorch(
        self,
        model_name: str,
        components: Dict[str, torch.nn.Module],
        device: str = 'cpu'
    ):
        """Load PyTorch model"""
        self.pytorch_components = components
        self.model_name = model_name
        self.use_onnx = False
        
        for comp in components.values():
            comp.to(device)
            comp.eval()
    
    def load_onnx(self, onnx_path: str) -> bool:
        """Load ONNX model"""
        try:
            self.onnx_engine = ONNXInferenceEngine(onnx_path)
            self.model_name = Path(onnx_path).stem
            self.use_onnx = True
            return True
        except Exception as e:
            print(f"Failed to load ONNX: {e}")
            return False
    
    def predict(
        self,
        audio_features: Any,
        metadata: Dict[str, Any],
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Unified prediction API
        """
        if self.use_onnx and self.onnx_engine:
            # Convert torch tensor ke numpy jika perlu
            if isinstance(audio_features, torch.Tensor):
                audio_features = audio_features.cpu().numpy()
            
            # Encode metadata untuk ONNX
            metadata_tensor = self._encode_metadata_numpy(metadata)
            
            return self.onnx_engine.predict(audio_features, metadata_tensor)
        else:
            # PyTorch inference
            from app.utils.metadata_encoder import SimpleMetadataEncoder
            
            metadata_encoder = self.pytorch_components.get('metadata_encoder')
            feature_fusion = self.pytorch_components.get('feature_fusion')
            classifier = self.pytorch_components.get('classifier')
            
            if not all([metadata_encoder, feature_fusion, classifier]):
                raise RuntimeError("PyTorch components not loaded")
            
            with torch.no_grad():
                metadata_emb = metadata_encoder.encode(metadata).to(device)
                
                if audio_features.dim() == 1:
                    audio_features = audio_features.unsqueeze(0)
                
                fused = feature_fusion(audio_features.to(device), metadata_emb)
                logits = classifier(fused)
                probs = torch.softmax(logits, dim=1)
                tb_prob = probs[0][1].item()
            
            if tb_prob < 0.3:
                risk = "RENDAH"
            elif tb_prob < 0.7:
                risk = "MENENGAH"
            else:
                risk = "TINGGI"
            
            return {
                'tb_probability': tb_prob,
                'risk_level': risk,
                'model_type': 'pytorch'
            }
    
    def _encode_metadata_numpy(self, metadata: Dict[str, Any]) -> np.ndarray:
        """Encode metadata untuk ONNX input"""
        # Simple numerical encoding
        age = metadata.get('age', 30) / 100.0
        gender = 0.0 if metadata.get('gender', 'L') == 'L' else 1.0
        cough_duration = min(metadata.get('cough_duration_days', 0) / 30.0, 1.0)
        has_fever = float(metadata.get('has_fever', False))
        has_cough = float(metadata.get('has_cough', True))
        has_night_sweats = float(metadata.get('has_night_sweats', False))
        has_weight_loss = float(metadata.get('has_weight_loss', False))
        has_chest_pain = float(metadata.get('has_chest_pain', False))
        has_shortness_breath = float(metadata.get('has_shortness_breath', False))
        previous_tb = float(metadata.get('previous_tb_history', False))
        
        features = [
            age, gender, cough_duration, has_fever, has_cough,
            has_night_sweats, has_weight_loss, has_chest_pain,
            has_shortness_breath, previous_tb
        ]
        
        return np.array([[features]], dtype=np.float32).reshape(1, -1)
    
    def benchmark(self, num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark current inference engine"""
        if self.use_onnx and self.onnx_engine:
            return {
                'engine': 'onnx',
                'model': self.model_name,
                'results': self.onnx_engine.benchmark(num_runs)
            }
        else:
            # PyTorch benchmark
            import time
            
            audio_features = torch.randn(1, 768)
            metadata = {'age': 30, 'gender': 'L'}
            
            # Warmup
            for _ in range(10):
                self.predict(audio_features, metadata)
            
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                self.predict(audio_features, metadata)
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            return {
                'engine': 'pytorch',
                'model': self.model_name,
                'results': {
                    'mean_ms': np.mean(times),
                    'std_ms': np.std(times),
                    'fps': 1000.0 / np.mean(times)
                }
            }


# Global instances
_inference_manager = None


def get_inference_manager() -> ModelInferenceManager:
    """Get atau create global inference manager"""
    global _inference_manager
    if _inference_manager is None:
        _inference_manager = ModelInferenceManager()
    return _inference_manager
