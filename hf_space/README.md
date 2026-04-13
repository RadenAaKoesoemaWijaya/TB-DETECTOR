# TB DETECTOR - Hugging Face Spaces Demo

🫁 **Tuberculosis Detection via Audio Recognition** - AI-powered demo untuk deteksi risiko TB dari suara batuk dan gejala klinis.

## 🎯 Demo Ini

Versi demo ini adalah **simplified version** dari TB DETECTOR v3 yang dioptimalkan untuk:
- ✅ Deployment gratis di Hugging Face Spaces
- ✅ CPU inference (tanpa GPU)
- ✅ User-friendly Gradio UI
- ✅ Demo untuk presentasi dan edukasi

## ⚡ Quick Start - Deploy ke Hugging Face

### Langkah 1: Buat Repository Hugging Face

1. Buka [huggingface.co/spaces](https://huggingface.co/spaces)
2. Klik **"Create new Space"**
3. Pilih:
   - **Owner:** Your username
   - **Space Name:** `tb-detector-demo`
   - **License:** Apache 2.0
   - **Space SDK:** Gradio
   - **Space Hardware:** CPU (free tier)
4. Klik **"Create Space"**

### Langkah 2: Upload Files

Upload 3 file ini ke repository:

```
tb-detector-demo/
├── app.py              ✅ (file utama)
├── requirements.txt    ✅ (dependencies)
└── README.md           ✅ (dokumentasi)
```

**Cara upload:**
1. Buka repository Hugging Face Anda
2. Klik **"Files and versions"** tab
3. Klik **"Upload files"**
4. Upload ketiga file di atas
5. Commit dengan message: "Initial demo version"

### Langkah 3: Tunggu Build

Hugging Face akan otomatis:
1. Install dependencies dari `requirements.txt`
2. Download model Wav2Vec2 (200MB)
3. Build Gradio interface
4. Deploy ke URL publik

**Status build:** Lihat di tab **"Files and versions"** → **"Container"**

### Langkah 4: Akses Demo

Setelah build selesai (~5-10 menit pertama kali):

```
https://huggingface.co/spaces/[YOUR_USERNAME]/tb-detector-demo
```

Contoh: `https://huggingface.co/spaces/john/tb-detector-demo`

---

## 📋 Cara Menggunakan Demo

### 1. Upload Audio Batuk
- Klik **"Upload Audio"** atau rekam langsung
- Format: WAV atau MP3
- Durasi: 3-5 detik batuk

### 2. Isi Data Pasien
- **Usia:** Slider 1-100 tahun
- **Gender:** Pilih Laki-laki/Perempuan

### 3. Pilih Gejala (Checklist)
- ☑️ Demam > 2 minggu
- ☑️ Batuk > 2 minggu
- ☑️ Berkeringat malam
- ☑️ Penurunan berat badan
- ☑️ Nyeri dada

### 4. Klik "Analisis Risiko TB"
- Tunggu 5-10 detik untuk analisis
- Lihat hasil probabilitas dan rekomendasi

---

## 🔧 Troubleshooting Deploy

### Error: "Build failed"

**Solusi:**
```bash
# Cek logs di Hugging Face
# Tab "Files and versions" → "Container"
# Lihat error message
```

Common issues:
- **Timeout:** Model download lambat, coba deploy ulang
- **Memory:** Upgrade ke CPU upgrade tier ($0/hari)
- **Dependencies:** Pastikan semua package di requirements.txt tersedia

### Error: "Model not found"

**Solusi:**
- Pastikan `transformers` di requirements.txt
- Model akan auto-download saat pertama kali
- Butuh ~200MB disk space

### Slow Inference

**Normal:** CPU inference butuh 5-10 detik per audio.

**Optimasi:**
- Audio pendek (3-5 detik)
- Format WAV lebih cepat dari MP3

---

## 🚀 Upgrade ke GPU (Optional)

Jika inference lambat:

1. Buka **Settings** di Hugging Face Space
2. Pilih **Hardware** → **GPU (Nvidia T4)**
3. Harga: **$0/hari** (free tier juga tersedia dengan GPU)
4. Restart Space

**Kecepatan:**
- CPU: ~10 detik
- GPU: ~2 detik

---

## 📊 Perbedaan Demo vs Full Version

| Fitur | Hugging Face Demo | Full TB DETECTOR v3 |
|-------|-------------------|---------------------|
| **Model** | Simplified Wav2Vec2 | Multi-backbone (HeAR, XLS-R) |
| **Training** | ❌ Tidak ada | ✅ Full training pipeline |
| **Dataset Upload** | ❌ Tidak ada | ✅ ZIP upload & preprocessing |
| **Model Comparison** | ❌ Tidak ada | ✅ Multi-model visualization |
| **Save/Load Model** | ❌ Tidak ada | ✅ Model management |
| **Deployment** | Hugging Face (gratis) | Self-hosted/VPS |
| **Inference Time** | 5-10 detik (CPU) | 1-3 detik (GPU) |
| **Use Case** | Demo/Edukasi | Production/Research |

---

## 🎓 Use Cases Demo Ini

### 1. **Presentasi & Edukasi**
- Demo AI untuk deteksi TB
- Edukasi kesehatan masyarakat
- Workshop machine learning

### 2. **Proof of Concept**
- Validasi ide ke stakeholder
- Pitch ke investor
- Grant proposal

### 3. **Research Sharing**
- Share hasil penelitian
- Kolaborasi dengan tim lain
- Feedback dari komunitas

---

## 🔗 Integrasi dengan Full Version

Demo ini bisa terhubung dengan TB DETECTOR v3 full version:

```
Demo (Hugging Face)  →  Edukasi & Awareness
                              ↓
Full Version (VPS)   →  Production Deployment
                              ↓
Mobile App           →  Field Deployment
```

---

## 📞 Support

**Hugging Face Issues:**
- [Hugging Face Forum](https://discuss.huggingface.co/c/spaces/24)
- [Gradio Docs](https://gradio.app/docs)

**TB DETECTOR Full Version:**
- GitHub: [your-repo-url]
- Email: [your-email]

---

## ⚠️ Disclaimer

**IMPORTANT:**

1. **Demo Version Only** - Ini adalah versi demo dengan model yang disederhanakan
2. **Not for Diagnosis** - Hasil bukan diagnosis medis
3. **Consult Professionals** - Selalu konsultasi dengan dokter
4. **Research Purpose** - Untuk edukasi dan penelitian

**Full Production Version:**
- Lebih akurat dengan multi-backbone training
- Dataset preprocessing lengkap
- Model validation dan testing
- Regulatory compliance (FDA/BPOM)

---

## 🌟 Like This Demo?

Jika bermanfaat, berikan **⭐ Star** pada repository TB DETECTOR!

[![GitHub stars](https://img.shields.io/github/stars/yourusername/tb-detector?style=social)](https://github.com/yourusername/tb-detector)

---

**TB DETECTOR v3** - AI for Tuberculosis Detection  
**Demo Version** - Hugging Face Spaces | **Full Version** - [GitHub](https://github.com/yourusername/tb-detector)
