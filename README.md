# TB DETECTOR v3 - Fully Integrated AI Pipeline

> **TB DETECTOR** (Tuberculosis Detection via Audio Recognition)

Aplikasi TB Detection dengan alur lengkap terintegrasi: **Upload Dataset → Preprocessing → Training → Visualization → Save Model → Prediction**

### 📋 Changelog Terbaru
- **v3.0.1** - Updated dependencies: `torch==2.6.0` (sebelumnya 2.1.1) untuk kompatibilitas Python 3.12+
- **v3.0.0** - Initial release dengan integrated pipeline

> **Catatan:** Jika mengalami error saat instalasi dependencies, jalankan `.\fix_dependencies.bat`

## 🎯 Fitur Utama v3

### 1. Complete Integrated Pipeline
- **Upload Dataset**: Upload ZIP file CODA TB DREAM langsung dari browser
- **Preprocessing**: Audio resampling, cough segmentation, normalization otomatis
- **Multi-Model Training**: Training beberapa backbone sekaligus (HeAR, Wav2Vec2, XLS-R)
- **Real-time Visualization**: Visualisasi performa model dalam bentuk chart
- **Model Management**: Simpan model terbaik dengan metadata
- **Prediction**: Inference dengan model yang telah dilatih

### 2. Interactive UI
- **5-Step Pipeline**: Visual progress indicator
- **Drag & Drop Upload**: Upload dataset dengan mudah
- **Real-time Logs**: Melihat progress training secara live
- **Comparison Charts**: Perbandingan visual metrics antar model
- **Model Cards**: Informasi detail untuk setiap model

### 3. Multi-Backbone Training
Train dan bandingkan otomatis:
- `wav2vec2-base` (768-dim)
- `wav2vec2-xlsr` (1024-dim) 
- `hear` (1024-dim)
- `hubert-base/large` (768/1024-dim)

## 📁 Struktur Dataset (ZIP)

```
dataset.zip
└── dataset/
    ├── audio/
    │   ├── cough_001.wav
    │   ├── cough_002.wav
    │   └── ...
    └── metadata.csv
```

### Format metadata.csv:
```csv
audio_filename,patient_id,age,gender,tb_label,cough_duration_days,has_fever,has_night_sweats,has_weight_loss,has_chest_pain,has_shortness_breath,previous_tb_history
cough_001.wav,P001,35,L,1,14,1,1,1,0,1,0
cough_002.wav,P002,28,P,0,3,0,0,0,0,0,0
...
```

## � Instalasi & Setup

### Persyaratan Sistem
- Python 3.9 atau lebih baru
- Windows 10/11 atau Linux/MacOS
- Minimal 4GB RAM (8GB direkomendasikan)
- ~2GB free disk space untuk dependencies

### Setup Pertama Kali

```bash
# 1. Fix dependencies (WAJIB untuk instalasi pertama)
.\fix_dependencies.bat

# Tunggu sampai selesai (butuh beberapa menit untuk download)
```

> **Catatan Penting:** Jika Anda mengalami error `torch==2.1.1 not found`, script `fix_dependencies.bat` sudah diperbarui dengan versi torch yang kompatibel (2.6.0).

### Troubleshooting Instalasi

| Error | Solusi |
|-------|--------|
| `torch==X.X.X not found` | Jalankan `.\fix_dependencies.bat` |
| `No module named uvicorn` | Reinstall dengan `.\fix_dependencies.bat` |
| Virtual environment corrupt | Hapus folder `venv/` dan jalankan fix script |

## �� Cara Menggunakan

### 1. Jalankan Server v3
```bash
# Windows (Command Prompt)
start_v3.bat

# Windows (PowerShell) - Perhatikan titik di depan!
.\start_v3.bat

# Linux/Mac
python -m uvicorn app.main_v3:app --host 0.0.0.0 --port 8000 --reload
```

> **Catatan PowerShell:** PowerShell tidak menjalankan script dari lokasi saat ini secara default untuk keamanan. Gunakan `\` sebelum nama file, contoh: `.\start_v3.bat`
```

### 2. Akses Aplikasi
Buka browser: **http://localhost:8000**

### 3. Alur Penggunaan

#### Step 1: Upload Dataset
- Klik area upload atau drag & drop file ZIP
- Struktur ZIP harus berisi folder `audio/` dan file `metadata.csv`
- Dataset akan di-extract dan siap untuk dipreprocessing

#### Step 2: Preprocessing
- Klik "Mulai Preprocessing"
- Sistem akan:
  - Resampling audio ke 16kHz
  - Normalisasi amplitude
  - Cough detection dan segmentation
  - Menyiapkan data untuk training

#### Step 3: Training
- Pilih backbone yang ingin dilatih (bisa multiple)
- Atur konfigurasi: epochs, batch size, learning rate, patience
- Klik "Mulai Training"
- Pantau progress dan logs secara real-time

#### Step 4: Results
- Lihat comparison chart performa model
- Model dengan AUROC tertinggi akan di-mark sebagai "BEST"
- Review metrics: Accuracy, Sensitivity, Specificity, F1, AUROC

#### Step 5: Save Model
- Klik "Simpan Model Terbaik" atau simpan model individual
- Model akan tersimpan di `app/models/weights/`

#### Step 6: Prediction
- Input data pasien (umur, gejala, dll)
- Upload audio batuk
- Jalankan prediksi dengan model terlatih

## 📊 API Endpoints v3

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Integrated UI |
| `/dataset/upload` | POST | Upload dataset ZIP |
| `/dataset/preprocess` | POST | Start preprocessing |
| `/training/start` | POST | Start training dengan konfigurasi |
| `/training/results` | GET | Get training results |
| `/training/visualization` | GET | Get comparison chart image |
| `/models/save` | POST | Save trained model |
| `/predict` | POST | Predict dengan model |
| `/pipeline/status` | GET | Get pipeline status & progress |
| `/pipeline/logs` | GET | Get recent logs |

## 📈 Alur Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE v3                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. UPLOAD                    2. PREPROCESS                 │
│  ┌──────────────┐             ┌──────────────┐             │
│  │ Dataset ZIP  │────────────→│ Resampling   │             │
│  │ (audio+csv)  │             │ Cough Detect │             │
│  └──────────────┘             │ Normalize    │             │
│                               └──────────────┘             │
│                                        │                    │
│                                        ▼                    │
│  3. TRAINING                    4. RESULTS                  │
│  ┌──────────────┐             ┌──────────────┐             │
│  │ HeAR         │────────────→│ Comparison   │             │
│  │ Wav2Vec2     │             │ Chart        │             │
│  │ XLS-R        │             │ Model Cards  │             │
│  └──────────────┘             └──────────────┘             │
│                                        │                    │
│                                        ▼                    │
│  5. SAVE                        6. PREDICT                  │
│  ┌──────────────┐             ┌──────────────┐             │
│  │ Best Model   │────────────→│ Input: Audio │             │
│  │ Metadata     │             │ + Metadata   │             │
│  └──────────────┘             │ Output: Risk │             │
│                               └──────────────┘             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🎨 UI Components

### Pipeline Steps Visual
- **Step 1**: Upload Dataset (📦)
- **Step 2**: Preprocessing (⚙️)
- **Step 3**: Training (🚀)
- **Step 4**: Results (📊)
- **Step 5**: Prediction (🔮)

### Real-time Monitoring
- **Progress Bar**: Visual progress untuk setiap tahap
- **Logs Console**: Real-time logs dengan color coding
- **Status Polling**: Auto-update status setiap 2 detik

### Model Comparison
- **Bar Chart**: Perbandingan metrics semua model
- **Model Cards**: Detail metrics per model
- **Best Badge**: Highlight model terbaik (highest AUROC)

## 📂 Output Files

### Setelah Training:
```
app/models/weights/
├── {backbone}_model.pth           # Model checkpoint
├── {backbone}_metrics.json        # Training metrics
├── model_comparison.json          # Comparison data
├── model_comparison.png           # Visualization chart
├── best_model.json                # Best model info
└── {backbone}_package.json        # Saved model metadata
```

### Metrics yang Disimpan:
- Accuracy, Precision, Sensitivity, Specificity
- F1-Score, AUROC (primary metric)
- Confusion Matrix
- Training history (loss curves)

## ⚙️ Training Configuration

### Default Configuration:
```json
{
  "backbones": ["wav2vec2-base", "wav2vec2-xlsr", "hear"],
  "epochs": 30,
  "batch_size": 16,
  "learning_rate": 0.0001,
  "patience": 10,
  "test_size": 0.2,
  "val_size": 0.15,
  "augment": true,
  "pos_weight": 2.0
}
```

### Custom Configuration via API:
```bash
curl -X POST "http://localhost:8000/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "backbones": ["wav2vec2-xlsr", "hear"],
    "epochs": 50,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "patience": 15
  }'
```

## 🔍 Preprocessing Details

### Audio Processing Pipeline:
1. **Load**: Load audio dengan librosa
2. **Resampling**: Convert ke 16kHz mono
3. **Normalization**: Amplitude normalization ke [-1, 1]
4. **Segmentation**: VAD + Energy-based cough detection
5. **Padding/Truncating**: Fixed length 5 seconds

### Metadata Processing:
- Auto-map columns jika nama berbeda
- Fill default values untuk missing columns
- Encode metadata ke embedding 32-dim

## 💾 Save Model Feature

### Save dengan Metadata:
```json
{
  "model_name": "wav2vec2-xlsr",
  "description": "Model terbaik dari training batch",
  "tags": ["trained", "v3", "production"],
  "metrics": { ... },
  "history": { ... },
  "saved_at": "2024-01-15T10:30:00",
  "config": { ... }
}
```

## 🔄 Dari v2 ke v3

### Perbedaan Utama:
| Fitur | v2 | v3 |
|-------|-----|-----|
| Dataset Upload | Manual copy | ZIP Upload |
| Preprocessing | Manual script | Integrated pipeline |
| Training | CLI script | Web UI dengan progress |
| Visualization | Static files | Real-time charts |
| Model Saving | Manual copy | One-click save dengan metadata |

### Migration:
- Dataset v2 tetap kompatibel
- Model weights v2 dapat digunakan di v3
- Gunakan `app/main_v2.py` untuk backward compatibility

## 🛠️ Troubleshooting

### "Dataset structure invalid"
- Pastikan ZIP berisi folder `audio/` dan file `metadata.csv`
- Audio files harus berformat WAV atau MP3

### "Out of memory during training"
- Kurangi batch size di konfigurasi
- Kurangi jumlah backbone yang dilatih bersamaan
- Gunakan GPU jika tersedia

### "Visualization not available"
- Training harus selesai minimal 1 model
- Cek file `app/models/weights/model_comparison.png`

## 🧹 Project Cleanup

**Status: ✅ COMPLETE**

Semua file versi lama (v1 dan v2) telah dihapus. Project sekarang hanya berisi **v3 (Integrated Pipeline)** sebagai versi final.

### File yang Dihapus:
- `app/main.py`, `app/main_v2.py` - Backend v1/v2
- `app/static/index.html`, `index_v2.html` - UI v1/v2
- `app/static/app.js`, `app_v2.js` - Frontend v1/v2
- `train.py` - Training script v1
- `start.bat`, `start_v2.bat`, `start.sh` - Launcher lama
- `README.md`, `README_v2.md` - Dokumentasi lama

### Struktur Final Project:
```
TB_DETECTOR/
├── app/
│   ├── main_v3.py          ✅ Backend v3
│   ├── model_manager.py    ✅ Model management
│   ├── models/             ✅ ML components
│   ├── utils/              ✅ Utilities
│   └── static/             ✅ Frontend v3
├── train_multi_backbone.py ✅ Training
├── export_onnx.py         ✅ Export utility
├── start_v3.bat          ✅ Launcher
└── README_v3.md          ✅ Dokumentasi ini
```

## 🧪 Testing

### Unit Tests
```bash
# Test API endpoints
python test_api.py

# Test model loading
python -c "from app.model_manager import get_model_manager; print('OK')"
```

### Integration Tests
1. Upload dataset ZIP (test data)
2. Run preprocessing
3. Train 1 epoch (quick test)
4. Verify visualization generated
5. Test prediction endpoint

## 🔒 Security & Validation

### Folder Validation (Auto-created on startup)
- `app/models/weights/` - Model checkpoints
- `data/` - Dataset storage
- `data/uploaded_dataset/` - Extracted datasets

### Input Validation
- Dataset ZIP format check
- Audio file format validation (WAV/MP3)
- Metadata CSV schema validation
- Form data type checking

## 📚 Referensi

- [CODA TB DREAM Dataset](https://coda-tb.org/)
- [Wav2Vec 2.0](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
- [Google HeAR](https://research.google/blog/hear-health-acoustic-representations/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

**TB DETECTOR v3** - Production-ready TB Detection System | Integrated AI Pipeline
**Version:** 3.0.0 | **Status:** Stable | **Last Updated:** 2024
