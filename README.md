# TB DETECTOR v3.2 - Integrated AI Pipeline

> **TB Detection via Audio Recognition** - Alur lengkap: Upload → Preprocess → Train → Visualize → Predict

## ⚡ Quick Start

### 1. Install Dependencies
```bash
# Windows
.\fix_dependencies.bat
```

### 2. Run Server
```bash
.\start_v3.bat
```

### 3. Open Browser
**http://localhost:8000**

API Documentation: **http://localhost:8000/docs**

---

## 📋 Requirements

- **Python:** 3.9 - 3.13
- **RAM:** 4GB minimum (8GB recommended for training)
- **OS:** Windows 10/11 / Linux / macOS
- **Disk:** 2GB untuk models dan cache

### Key Dependencies
```
torch==2.6.0 | fastapi==0.104.1 | transformers==4.35.2 | onnxruntime>=1.16.0
```

---

## 🎯 Features

### Core Pipeline (v3.0)
- **Integrated Pipeline:** 5-step workflow (Upload → Preprocess → Train → Results → Predict)
- **Multi-Backbone:** HeAR, Wav2Vec 2.0, XLS-R, HuBERT
- **Real-time Training:** Live logs and visualization
- **Model Management:** Save, compare, and load models
- **Interactive UI:** Drag & drop, charts, model cards

### Phase 1: Performance Optimizations
- **Batch Training:** 10-20x faster training dengan DataLoader (batch_size up to 64)
- **Feature Caching:** Persistent cache untuk pre-extracted features
- **SQLite Persistence:** State durability dan history tracking

### Phase 2: Production Features
- **Async I/O:** Non-blocking file operations dan preprocessing
- **Task Queue:** Background job scheduling dengan priority
- **ONNX Inference:** 2-3x faster inference dengan ONNX Runtime
- **Model Versioning:** Dev → Staging → Production workflow
- **A/B Testing:** Statistical model comparison dengan traffic allocation

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

### Python Not Found / Build Errors

**Problem:** `The system cannot find the file python.exe`

**Solutions:**

1. **Check Python Installation:**
   ```batch
   .\check_python.bat
   ```

2. **Install Python (jika belum):**
   - Download dari https://www.python.org/downloads/
   - Pilih Python **3.10, 3.11, atau 3.12** (recommended)
   - **Centang "Add Python to PATH"** saat installasi
   - Restart terminal setelah install

3. **Gunakan py launcher (alternatif):**
   - Ganti semua `python` dengan `py` di batch files
   - Atau jalankan: `py -m venv venv`

### Python 3.14 Compatibility Issues

**Problem:** `Could not find a version that satisfies the requirement torch==2.6.0`

**Cause:** Python 3.14 terlalu baru, beberapa packages belum ada pre-built wheels.

**Solutions:**

1. **Gunakan Quick Install (recommended untuk Python 3.14):**
   ```batch
   rmdir /s /q venv  :: Hapus venv yang corrupted
   .\quick_install.bat
   ```

2. **Atau downgrade ke Python 3.11/3.12:**
   - Uninstall Python 3.14
   - Install Python 3.11 atau 3.12 dari python.org
   - Jalankan ulang `fix_dependencies.bat`

3. **Manual install dengan versions fleksibel:**
   ```batch
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install fastapi uvicorn transformers librosa numpy pandas
   ```

### Other Issues

| Issue | Solution |
|-------|----------|
| `torch not found` | Run `.\fix_dependencies.bat` |
| `uvicorn error` | Delete `venv/` folder, run fix script |
| PowerShell error | Use `.\script.bat` (with `.\`) |
| `ImportError: cannot import name 'ONNX_AVAILABLE'` | Install ONNX: `pip install onnx onnxruntime` |
| SQLite locked | Restart server, check `data/pipeline.db` permissions |
| Cache corrupted | Delete `data/feature_cache/` folder |

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
├── fix_dependencies.bat             # Dependency fixer
└── README.md                        # This file
```

---

**TB DETECTOR v3.2.0** | Python 3.9+ | [http://localhost:8000](http://localhost:8000)
├── Phase 1: ✅ Performance (Batch Training, Caching, Persistence)
├── Phase 2: ✅ Production (Async, Task Queue, ONNX, Versioning, A/B Testing)
└── API Docs: /docs (auto-generated)

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
