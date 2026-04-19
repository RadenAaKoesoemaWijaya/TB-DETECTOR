# TB DETECTOR v3.2

> Sistem deteksi Tuberkulosis (TB) berbasis AI menggunakan analisis suara batuk.

---

## 🚀 Cara Menjalankan (3 Langkah)

### 1. Install Dependensi
```bash
# Untuk Python 3.10-3.12 (Direkomendasikan)
.\fix_dependencies.bat

# Untuk Python 3.13-3.14
.\quick_install.bat
```

### 2. Jalankan Server
```bash
.\start_v3.bat
```

### 3. Buka Browser
- **Web UI:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

---

## 📋 Persyaratan Sistem

| Komponen | Minimum | Direkomendasikan |
|----------|---------|------------------|
| Python | 3.9 | **3.10-3.12** |
| RAM | 4GB | 8GB+ |
| OS | Windows 10/11 / Linux / macOS | Windows 11 |
| Disk | 1GB | 2GB+ |

---

## 📁 Struktur Dataset

Upload dataset dalam format ZIP dengan struktur:

```
dataset.zip
└── dataset/
    ├── audio/        # File audio (.wav / .mp3)
    └── metadata.csv  # Informasi pasien dan label
```

**Format metadata.csv:**
```csv
audio_filename,patient_id,age,gender,tb_label,cough_duration_days,has_fever,...
cough_001.wav,P001,35,L,1,14,1,...
```

---

## 🔧 Penyelesaian Masalah

### Masalah Umum

| Masalah | Solusi |
|---------|--------|
| Module not found | Hapus folder `venv/`, jalankan ulang `.\fix_dependencies.bat` |
| Python not found | Jalankan `.\check_python.bat` untuk cek instalasi |
| Database error | Hapus file `data\pipeline.db` lalu restart |
| Cache error | Hapus folder `data\feature_cache` |

### Error Python 3.14
Jika error saat install di Python 3.14:
```bash
rmdir /s /q venv
.\quick_install.bat
```

---

## 📚 API Endpoint Utama

| Endpoint | Fungsi |
|----------|--------|
| `POST /dataset/upload` | Upload dataset ZIP |
| `POST /dataset/preprocess` | Preprocessing data |
| `POST /training/start` | Mulai training model |
| `GET /training/results` | Lihat hasil training |
| `POST /predict` | Prediksi TB dari audio |
| `POST /models/export-onnx/{name}` | Export model ke ONNX |

---

## 🗂️ Struktur Folder

```
TB-DETECTOR/
├── app/                    # Source code aplikasi
│   ├── main_v3.py          # Entry point FastAPI
│   ├── model_manager.py    # Manajemen model
│   ├── persistence.py      # Database SQLite
│   └── training/           # Modul training
├── data/                   # Data dan cache
│   ├── feature_cache/      # Cache fitur
│   └── pipeline.db           # Database
├── start_v3.bat            # Script jalankan server
├── fix_dependencies.bat    # Install dependensi
└── README.md               # Dokumentasi
```

---

## � Fitur Utama

- ✅ Upload dataset ZIP
- ✅ Preprocessing audio otomatis
- ✅ Training model dengan berbagai backbone (HeAR, Wav2Vec2, XLS-R, HuBERT)
- ✅ Visualisasi hasil training
- ✅ Prediksi TB dari audio batuk
- ✅ Export model ke format ONNX

---

<div align="center">

**TB DETECTOR v3.2** | Python 3.9-3.14

Sistem deteksi TB berbasis AI via analisis suara

</div>
