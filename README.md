# TB DETECTOR v3 - Integrated AI Pipeline

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

---

## 📋 Requirements

- **Python:** 3.9 - 3.13
- **RAM:** 4GB minimum (8GB recommended)
- **OS:** Windows 10/11 / Linux / macOS

### Key Dependencies
```
torch==2.6.0 | fastapi==0.104.1 | transformers==4.35.2
```

---

## 🎯 Features

- **Integrated Pipeline:** 5-step workflow (Upload → Preprocess → Train → Results → Predict)
- **Multi-Backbone:** HeAR, Wav2Vec 2.0, XLS-R, HuBERT
- **Real-time Training:** Live logs and visualization
- **Model Management:** Save, compare, and load models
- **Interactive UI:** Drag & drop, charts, model cards

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

| Issue | Solution |
|-------|----------|
| `torch not found` | Run `.\fix_dependencies.bat` |
| `uvicorn error` | Delete `venv/` folder, run fix script |
| PowerShell error | Use `.\script.bat` (with `.\`) |

---

## 🧹 Cleanup Complete

✅ All v1/v2 files removed. Only v3 remains:
- `app/main_v3.py` - Backend
- `app/static/index_v3.html` - UI
- `train_multi_backbone.py` - Training

---

**TB DETECTOR v3.0.1** | Python 3.9+ | [http://localhost:8000](http://localhost:8000)
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
