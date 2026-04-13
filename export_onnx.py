"""
Export model ke ONNX untuk deployment di mobile (Edge AI)
Phase 3: Client-Side Deployment
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import os

from app.models.classifier import TBClassifier
from app.utils.feature_fusion import SimpleConcatFusion
from app.utils.metadata_encoder import SimpleMetadataEncoder


def export_to_onnx(
    checkpoint_path: str = "app/models/weights/tb_classifier.pth",
    output_path: str = "app/models/weights/tb_detector.onnx"
):
    """
    Export PyTorch model ke ONNX format
    """
    print("Loading PyTorch model...")
    
    # Initialize models
    metadata_encoder = SimpleMetadataEncoder(embedding_dim=32)
    feature_fusion = SimpleConcatFusion(
        audio_dim=1024, metadata_dim=32, fused_dim=512
    )
    classifier = TBClassifier(
        input_dim=512, hidden_dims=[256, 128], num_heads=4, dropout=0.3
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    metadata_encoder.load_state_dict(checkpoint['metadata_encoder'])
    feature_fusion.load_state_dict(checkpoint['feature_fusion'])
    classifier.load_state_dict(checkpoint['classifier'])
    
    metadata_encoder.eval()
    feature_fusion.eval()
    classifier.eval()
    
    # Create combined model
    class TBDetectorONNX(torch.nn.Module):
        def __init__(self, metadata_encoder, feature_fusion, classifier):
            super().__init__()
            self.metadata_encoder = metadata_encoder
            self.feature_fusion = feature_fusion
            self.classifier = classifier
        
        def forward(self, audio_features, age, gender, cough_duration, 
                    has_fever, has_cough, has_night_sweats, has_weight_loss,
                    has_chest_pain, has_shortness_breath, previous_tb):
            # Reconstruct metadata tensor
            metadata = torch.stack([
                age / 100.0,
                gender,
                cough_duration / 30.0,
                has_fever,
                has_cough,
                has_night_sweats,
                has_weight_loss,
                has_chest_pain,
                has_shortness_breath,
                previous_tb
            ], dim=-1)
            
            # Forward pass
            metadata_embedding = self.metadata_encoder.encoder(metadata)
            fused = self.feature_fusion(audio_features, metadata_embedding)
            logits = self.classifier(fused)
            probs = torch.softmax(logits, dim=-1)
            
            return probs
    
    model = TBDetectorONNX(metadata_encoder, feature_fusion, classifier)
    model.eval()
    
    # Dummy inputs
    batch_size = 1
    audio_features = torch.randn(batch_size, 1024)
    age = torch.tensor([[30.0]])
    gender = torch.tensor([[0.0]])
    cough_duration = torch.tensor([[7.0]])
    has_fever = torch.tensor([[0.0]])
    has_cough = torch.tensor([[1.0]])
    has_night_sweats = torch.tensor([[0.0]])
    has_weight_loss = torch.tensor([[0.0]])
    has_chest_pain = torch.tensor([[0.0]])
    has_shortness_breath = torch.tensor([[0.0]])
    previous_tb = torch.tensor([[0.0]])
    
    dummy_inputs = (
        audio_features, age, gender, cough_duration,
        has_fever, has_cough, has_night_sweats, has_weight_loss,
        has_chest_pain, has_shortness_breath, previous_tb
    )
    
    input_names = [
        'audio_features', 'age', 'gender', 'cough_duration',
        'has_fever', 'has_cough', 'has_night_sweats', 'has_weight_loss',
        'has_chest_pain', 'has_shortness_breath', 'previous_tb'
    ]
    
    output_names = ['tb_probability']
    
    dynamic_axes = {
        'audio_features': {0: 'batch_size'},
        'age': {0: 'batch_size'},
        'gender': {0: 'batch_size'},
        'cough_duration': {0: 'batch_size'},
        'has_fever': {0: 'batch_size'},
        'has_cough': {0: 'batch_size'},
        'has_night_sweats': {0: 'batch_size'},
        'has_weight_loss': {0: 'batch_size'},
        'has_chest_pain': {0: 'batch_size'},
        'has_shortness_breath': {0: 'batch_size'},
        'previous_tb': {0: 'batch_size'},
        'tb_probability': {0: 'batch_size'}
    }
    
    print(f"Exporting to ONNX: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=11,
        do_constant_folding=True,
        verbose=False
    )
    
    print("ONNX export complete!")
    
    # Verify
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")
    
    # Test inference
    print("Testing ONNX Runtime inference...")
    session = ort.InferenceSession(output_path)
    
    # Prepare inputs
    ort_inputs = {
        'audio_features': audio_features.numpy(),
        'age': age.numpy(),
        'gender': gender.numpy(),
        'cough_duration': cough_duration.numpy(),
        'has_fever': has_fever.numpy(),
        'has_cough': has_cough.numpy(),
        'has_night_sweats': has_night_sweats.numpy(),
        'has_weight_loss': has_weight_loss.numpy(),
        'has_chest_pain': has_chest_pain.numpy(),
        'has_shortness_breath': has_shortness_breath.numpy(),
        'previous_tb': previous_tb.numpy()
    }
    
    ort_outputs = session.run(None, ort_inputs)
    print(f"Test output shape: {ort_outputs[0].shape}")
    print(f"Test output: {ort_outputs[0]}")
    
    print(f"\nExport successful! Model saved to: {output_path}")
    print("\nModel info:")
    print(f"  Input: audio_features [batch, 1024] + 10 metadata features")
    print(f"  Output: tb_probability [batch, 2] (class 0: negatif, class 1: positif)")


def export_backbone_onnx(output_path: str = "app/models/weights/wav2vec2_backbone.onnx"):
    """
    Export hanya Wav2Vec2 backbone ke ONNX (opsional, untuk server-side inference)
    """
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    
    print("Loading Wav2Vec2 model...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 16000 * 5)  # 5 seconds audio
    
    # Get processed input
    inputs = processor(dummy_input.squeeze(0), sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values
    
    print(f"Exporting Wav2Vec2 backbone to ONNX: {output_path}")
    
    torch.onnx.export(
        model,
        input_values,
        output_path,
        input_names=['input_values'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_values': {0: 'batch', 1: 'sequence'},
            'last_hidden_state': {0: 'batch', 1: 'sequence'}
        },
        opset_version=11,
        do_constant_folding=True
    )
    
    print("Backbone export complete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--backbone":
        export_backbone_onnx()
    else:
        export_to_onnx()
