# TB DETECTOR v3.2 - Integrated AI Pipeline

> **TB Detection via Audio Recognition** - Alur lengkap: Upload → Preprocess → Train → Visualize → Predict

[![Python](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136.0-green)](https://fastapi.tiangolo.com/)
[![Status](https://img.shields.io/badge/Status-Stable-brightgreen)]()
[![Phase](https://img.shields.io/badge/Phase-2%20Complete-orange)]()

---

## ⚡ Quick Start (3 Langkah)

### 1. Install Dependencies
```bash
# Windows - Pilih salah satu sesuai Python version:

# Python 3.10-3.12 (Recommended - Paling Stabil)
.\fix_dependencies.bat

# Python 3.13-3.14 (Jika Python versi terbaru)
.\quick_install.bat
```

### 2. Run Server
```bash
.\start_v3.bat
```

### 3. Open Browser
**[http://localhost:8000](http://localhost:8000)**

API Docs: **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 📋 Requirements & Installation

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.9 | **3.10-3.12** |
| **RAM** | 4GB | 8GB+ (untuk training) |
| **OS** | Windows 10/11 / Linux / macOS | Windows 11 |
| **Disk** | 1GB | 2GB+ (models + cache) |

### ⚠️ Important: Python Version Compatibility

| Python Version | Script | Status | Notes |
|---------------|--------|--------|-------|
| 3.10 - 3.12 | `fix_dependencies.bat` | ✅ **Recommended** | Paling stabil, semua packages tersedia |
| 3.13 | `fix_dependencies.bat` | ✅ Supported | Beberapa packages perlu build dari source |
| 3.14 | `quick_install.bat` | ⚠️ Experimental | Gunakan latest package versions |

### Installation Methods

```bash
# Method 1: Standard Install (Python 3.10-3.12) ✅ RECOMMENDED
.\fix_dependencies.bat

# Method 2: Quick Install (Python 3.13-3.14)
.\quick_install.bat

# Method 3: Manual Install (jika script gagal)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn python-multipart transformers
pip install librosa soundfile pydub numpy pandas scikit-learn
```

---

## � What's New in v3.2

### Phase 2: Production Features ⭐ NEW
| Feature | Description | Speedup |
|---------|-------------|---------|
| **Async I/O** | Non-blocking file operations & preprocessing | 2-3x throughput |
| **Task Queue** | Background job scheduling dengan priority | Non-blocking API |
| **ONNX Inference** | Optimized inference dengan ONNX Runtime | **2-3x faster** |
| **Model Versioning** | Dev → Staging → Production workflow | - |
| **A/B Testing** | Statistical model comparison | - |

### Phase 1: Performance Optimizations
| Feature | Description | Speedup |
|---------|-------------|---------|
| **Batch Training** | DataLoader dengan batch_size up to 64 | **10-20x** |
| **Feature Caching** | Persistent cache pre-extracted features | ~5x repeated training |
| **SQLite Persistence** | State durability & history tracking | - |

### Core Pipeline (v3.0)
- ✅ **Integrated Pipeline:** Upload → Preprocess → Train → Visualize → Predict
- ✅ **Multi-Backbone:** HeAR, Wav2Vec 2.0, XLS-R, HuBERT
- ✅ **Real-time Training:** Live logs & visualization
- ✅ **Interactive UI:** Drag & drop, charts, model cards

---

## 📁 Dataset Structure

```
dataset.zip
└── dataset/
    ├── audio/        (*.wav/*.mp3)
    └── metadata.csv
```

**metadata.csv format:**
```csv
audio_filename,patient_id,age,gender,tb_label,cough_duration_days,has_fever,...
cough_001.wav,P001,35,L,1,14,1,...
```

---

## 🔧 Troubleshooting

### 🚨 Quick Fixes (Masalah Umum)

```bash
# Masalah: Module not found / Import error
rmdir /s /q venv
.\fix_dependencies.bat    # atau .\quick_install.bat untuk Python 3.14

# Masalah: Python not found
.\check_python.bat        # Diagnose Python installation

# Masalah: SQLite locked / Database error
del data\pipeline.db      # Reset database (WARNING: data akan hilang)

# Masalah: Cache corrupted
rmdir /s /q data\feature_cache    # Clear feature cache
```

### ⚠️ Python Installation Issues

#### Error: `python.exe not found` atau `Python was not found`

**Penyebab:** Python tidak terinstall atau tidak di PATH

**Solusi:**

1. **Diagnose dulu:**
   ```batch
   .\check_python.bat
   ```

2. **Install Python 3.10-3.12 (Recommended):**
   - Download: https://www.python.org/downloads/
   - ⚠️ **Centang "Add Python to PATH"** saat installasi
   - Restart terminal/command prompt

3. **Alternatif - Gunakan py launcher:**
   ```batch
   py -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

### ⚠️ Python 3.14 Specific Issues

#### Error: `Could not find a version that satisfies the requirement torch==2.6.0`

**Penyebab:** Python 3.14 terlalu baru, beberapa packages belum ada pre-built wheels

**Solusi untuk Python 3.14:**

```batch
# Method 1: Gunakan Quick Install (Recommended)
rmdir /s /q venv
.\quick_install.bat

# Method 2: Downgrade ke Python 3.11/3.12 (Paling Stabil)
# 1. Uninstall Python 3.14 dari Control Panel
# 2. Install Python 3.11 atau 3.12 dari python.org
# 3. Jalankan: .\fix_dependencies.bat

# Method 3: Manual Install dengan versi fleksibel
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn transformers librosa numpy pandas scikit-learn
```

### 🔍 Common Error Messages

| Error Message | Penyebab | Solusi |
|---------------|----------|--------|
| `No module named 'uvicorn'` | Dependencies tidak lengkap | Hapus `venv/`, jalankan script install ulang |
| `No module named 'torch'` | PyTorch tidak terinstall | Jalankan `fix_dependencies.bat` |
| `No module named 'transformers'` | Transformers tidak terinstall | `pip install transformers` |
| `No module named 'aiofiles'` | Phase 2 dependency kurang | `pip install aiofiles` |
| `ImportError: cannot import name 'ONNX_AVAILABLE'` | ONNX opsional tidak terinstall | `pip install onnx onnxruntime` |
| `SQLite locked` | Database sedang digunakan | Restart server, hapus `data/pipeline.db` jika persist |
| `ModuleNotFoundError: webrtcvad` | VAD library gagal install | Aplikasi tetap jalan dengan energy-based fallback |

### 🔧 Diagnostic Commands

```bash
# Test apakah semua dependencies terinstall dengan benar
python test_build.py

# Check Python version
python --version

# List installed packages
pip list

# Check virtual environment
venv\Scripts\python --version
```

---

## 📚 API Endpoints

### Dataset & Preprocessing
```
POST   /dataset/upload              # Upload dataset ZIP
POST   /dataset/preprocess          # Start preprocessing
```

### Training
```
POST   /training/start              # Start training
GET    /training/results            # Get training results
GET    /training/visualization      # Get comparison chart
```

### Phase 1: Persistence
```
GET    /persistence/logs             # Get persistent logs
GET    /persistence/state            # Get pipeline state
GET    /persistence/training-sessions
GET    /persistence/cache-stats     # Feature cache statistics
```

### Phase 2: Task Queue
```
GET    /tasks/queue                 # Queue status
GET    /tasks/list                  # List background tasks
GET    /tasks/{task_id}             # Task details
```

### Phase 2: Model Registry
```
GET    /registry/models             # List model versions
POST   /registry/models/{name}/{version}/promote  # Promote stage
GET    /registry/statistics          # Registry stats
```

### Phase 2: ONNX
```
POST   /models/export-onnx/{name}   # Export to ONNX
GET    /onnx/benchmark/{name}       # Benchmark ONNX model
```

### Phase 2: A/B Testing
```
POST   /experiments/create           # Create experiment
POST   /experiments/{id}/start       # Start experiment
POST   /experiments/{id}/stop        # Stop & get results
GET    /experiments/list             # List experiments
```

### Prediction
```
POST   /predict                      # Make prediction
```

---

## 🧹 Project Structure

```
TB-DETECTOR/
├── app/
│   ├── main_v3.py                 # FastAPI backend (updated v3.2)
│   ├── model_manager.py             # Model management
│   ├── persistence.py               # SQLite persistence
│   ├── async_utils.py               # Async I/O (Phase 2)
│   ├── task_queue.py                # Background tasks (Phase 2)
│   ├── onnx_inference.py            # ONNX Runtime (Phase 2)
│   ├── model_versioning.py          # Model registry (Phase 2)
│   ├── ab_testing.py                # A/B testing (Phase 2)
│   ├── training/
│   │   ├── batch_trainer.py         # Batch training (Phase 1)
│   │   └── cache_manager.py         # Feature caching (Phase 1)
│   ├── models/
│   │   ├── backbones.py             # HeAR, Wav2Vec2, XLS-R, HuBERT
│   │   ├── classifier.py            # MLP + Transformer
│   │   └── preprocessing.py         # Audio preprocessing
│   └── utils/
│       ├── feature_fusion.py        # Audio + metadata fusion
│       └── metadata_encoder.py      # Clinical metadata encoding
├── data/
│   ├── feature_cache/               # Extracted features cache
│   ├── experiments/                 # A/B testing data
│   └── pipeline.db                  # SQLite database
├── train_multi_backbone.py          # Standalone training
├── export_onnx.py                   # ONNX export utility
├── start_v3.bat                     # Windows launcher
├── fix_dependencies.bat             # Dependency fixer (standard)
├── quick_install.bat                # Quick install (Python 3.14 compatible)
├── check_python.bat                 # Check Python installation
└── README.md                        # This file
```

---

## 💻 Development Info

### Helper Scripts

| Script | Purpose |
|--------|---------|
| `start_v3.bat` | Start server dengan auto-reload |
| `fix_dependencies.bat` | Standard install untuk Python 3.10-3.12 |
| `quick_install.bat` | Quick install untuk Python 3.13-3.14 |
| `check_python.bat` | Diagnose Python installation |
| `test_build.py` | Test semua imports |

### File Structure Key Files

```
app/
├── main_v3.py              # FastAPI entry point
├── model_manager.py        # Model lifecycle management
├── persistence.py          # SQLite database operations
├── async_utils.py          # Async I/O utilities (Phase 2)
├── task_queue.py           # Background task queue (Phase 2)
├── onnx_inference.py       # ONNX runtime inference (Phase 2)
├── model_versioning.py     # Model registry (Phase 2)
├── ab_testing.py           # A/B testing framework (Phase 2)
└── training/               # Training modules
    ├── batch_trainer.py    # Batch training (Phase 1)
    └── cache_manager.py    # Feature caching (Phase 1)
```

---

**TB DETECTOR v3.2.0** | Python 3.9-3.14 | [http://localhost:8000](http://localhost:8000)

**Features:**
- ✅ Phase 1: Performance (Batch Training, Caching, Persistence)
- ✅ Phase 2: Production (Async I/O, Task Queue, ONNX, Versioning, A/B Testing)

## 🧪 Testing

### Build Test
```bash
# Test all imports
python test_build.py
```

### Manual Test
```bash
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

---

<div align="center">

**TB DETECTOR v3.2** - Production-ready TB Detection System

[![Version](https://img.shields.io/badge/Version-3.2.0-blue)]()
[![Status](https://img.shields.io/badge/Status-Stable-brightgreen)]()
[![Last Updated](https://img.shields.io/badge/Last%20Updated-April%202026-orange)]()

*Integrated AI Pipeline for Tuberculosis Detection via Audio Recognition*

</div>
