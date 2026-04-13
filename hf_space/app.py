"""
TB DETECTOR - Hugging Face Spaces Demo
Simplified demo version for Hugging Face Spaces
"""

import gradio as gr
import torch
import torchaudio
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import os
import tempfile
from typing import Tuple, Optional

# Constants
SAMPLE_RATE = 16000
MAX_LENGTH = 5  # seconds

class SimpleTBDetector:
    """Simplified TB Detector for Hugging Face Demo"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self.classifier = None
        self._load_model()
    
    def _load_model(self):
        """Load pretrained Wav2Vec2 and simple classifier"""
        try:
            print("Loading Wav2Vec2 model...")
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            self.model.to(self.device)
            self.model.eval()
            
            # Simple classifier head
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(768, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(256, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
                torch.nn.Sigmoid()
            ).to(self.device)
            
            # Load dummy weights (in real deployment, load trained weights)
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Preprocess audio file"""
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        # Normalize
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        
        # Trim or pad to MAX_LENGTH
        target_length = SAMPLE_RATE * MAX_LENGTH
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        
        return torch.FloatTensor(waveform).unsqueeze(0)
    
    def predict(self, audio_path: str, age: int, gender: str, 
                has_fever: bool, has_cough: bool, has_night_sweats: bool,
                has_weight_loss: bool, has_chest_pain: bool) -> Tuple[str, float, str]:
        """
        Predict TB risk from audio and symptoms
        
        Returns:
            result_text: Prediction result description
            confidence: Probability score
            recommendation: Medical recommendation
        """
        try:
            # Preprocess audio
            waveform = self.preprocess_audio(audio_path)
            waveform = waveform.to(self.device)
            
            # Extract features using Wav2Vec2
            with torch.no_grad():
                inputs = self.processor(
                    waveform.squeeze().cpu().numpy(), 
                    sampling_rate=SAMPLE_RATE, 
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                audio_features = outputs.last_hidden_state.mean(dim=1)  # [1, 768]
                
                # Get prediction
                prediction = self.classifier(audio_features)
                tb_probability = prediction.item()
            
            # Simple symptom-based adjustment (demo purposes)
            symptom_score = sum([
                has_fever, has_cough, has_night_sweats, 
                has_weight_loss, has_chest_pain
            ]) / 5.0
            
            # Combine audio and symptom scores
            final_probability = 0.6 * tb_probability + 0.4 * symptom_score
            
            # Determine risk level
            if final_probability < 0.3:
                risk_level = "LOW"
                color = "🟢"
                recommendation = """
**Rendah Risiko TB**

Rekomendasi:
- Tetap jaga kesehatan
- Perhatikan gejala jika memburuk
- Konsultasi dokter jika khawatir

*Disclaimer: Ini adalah demo AI. Hasil bukan diagnosis medis.*
"""
            elif final_probability < 0.7:
                risk_level = "MODERATE"
                color = "🟡"
                recommendation = """
**Risiko Sedang TB**

Rekomendasi:
- Perhatikan gejala dalam 1-2 minggu
- Segera periksa ke fasilitas kesehatan
- Simpan hasil ini untuk dokter

*Disclaimer: Ini adalah demo AI. Hasil bukan diagnosis medis.*
"""
            else:
                risk_level = "HIGH"
                color = "🔴"
                recommendation = """
**Tinggi Risiko TB**

Rekomendasi:
- Segera kunjungi fasilitas kesehatan
- Bawa hasil ini ke dokter
- Jangan panik, TB dapat diobati

*Disclaimer: Ini adalah demo AI. Hasil bukan diagnosis medis.*
"""
            
            result_text = f"""
{color} **Level Risiko: {risk_level}**

📊 **Probabilitas TB: {final_probability:.1%}**

🎵 **Audio Analysis:** {tb_probability:.1%}
🏥 **Symptom Score:** {symptom_score:.1%}

**Input Data:**
- Usia: {age} tahun
- Gender: {gender}
- Gejala: {sum([has_fever, has_cough, has_night_sweats, has_weight_loss, has_chest_pain])}/5
"""
            
            return result_text, final_probability, recommendation
            
        except Exception as e:
            return f"Error: {str(e)}", 0.0, "Terjadi kesalahan. Silakan coba lagi."

# Initialize detector
detector = SimpleTBDetector()

# Gradio Interface
def create_interface():
    with gr.Blocks(title="TB DETECTOR - Hugging Face Demo", theme=gr.themes.Soft()) as demo:
        
        # Header
        gr.Markdown("""
        # 🫁 TB DETECTOR v3
        ## Tuberculosis Detection via Audio Recognition
        
        **Demo Version for Hugging Face Spaces**
        
        Deteksi risiko Tuberkulosis (TB) dari suara batuk dan gejala klinis menggunakan AI.
        
        ---
        """)
        
        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=1):
                gr.Markdown("### 🎵 Upload Audio Batuk")
                audio_input = gr.Audio(
                    label="Rekam atau Upload Audio Batuk (WAV/MP3)",
                    type="filepath",
                    source="upload"
                )
                
                gr.Markdown("### 👤 Data Pasien")
                
                age = gr.Slider(
                    minimum=1, maximum=100, value=30, step=1,
                    label="Usia (tahun)"
                )
                
                gender = gr.Radio(
                    choices=["Laki-laki", "Perempuan"],
                    value="Laki-laki",
                    label="Gender"
                )
                
                gr.Markdown("### 🏥 Gejala Klinis")
                
                has_fever = gr.Checkbox(label="Demam > 2 minggu")
                has_cough = gr.Checkbox(label="Batuk > 2 minggu")
                has_night_sweats = gr.Checkbox(label="Berkeringat malam")
                has_weight_loss = gr.Checkbox(label="Penurunan berat badan")
                has_chest_pain = gr.Checkbox(label="Nyeri dada")
                
                analyze_btn = gr.Button(
                    "🔍 Analisis Risiko TB",
                    variant="primary",
                    size="lg"
                )
            
            # Right column - Results
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Hasil Analisis")
                
                result_output = gr.Markdown(
                    label="Hasil",
                    value="Klik 'Analisis Risiko TB' untuk memulai..."
                )
                
                confidence_plot = gr.Plot(
                    label="Confidence Score",
                    visible=False
                )
                
                recommendation_output = gr.Markdown(
                    label="Rekomendasi"
                )
        
        # Footer
        gr.Markdown("""
        ---
        
        ### ⚠️ Important Disclaimer
        
        **This is a DEMO application for research and educational purposes only.**
        
        - Results are NOT a medical diagnosis
        - Always consult healthcare professionals
        - This demo uses simplified AI models
        - For production use, use the full TB DETECTOR v3 application
        
        **Full Version Features:**
        - Complete integrated pipeline
        - Multi-backbone training (HeAR, Wav2Vec2, XLS-R)
        - Model comparison and visualization
        - Production-ready inference
        
        [View on GitHub](https://github.com/yourusername/tb-detector) | 
        [Research Paper](https://arxiv.org/abs/xxxx)
        """)
        
        # Event handlers
        def analyze(audio, age_val, gender_val, fever, cough, sweats, weight_loss, pain):
            if audio is None:
                return "⚠️ **Error:** Silakan upload audio batuk terlebih dahulu.", 0.0, ""
            
            result, confidence, recommendation = detector.predict(
                audio, age_val, gender_val, 
                fever, cough, sweats, weight_loss, pain
            )
            
            return result, confidence, recommendation
        
        analyze_btn.click(
            fn=analyze,
            inputs=[
                audio_input, age, gender,
                has_fever, has_cough, has_night_sweats,
                has_weight_loss, has_chest_pain
            ],
            outputs=[result_output, confidence_plot, recommendation_output]
        )
    
    return demo

# Launch
demo = create_interface()

if __name__ == "__main__":
    demo.launch()
