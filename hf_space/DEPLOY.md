# 🚀 Quick Deploy Guide - TB DETECTOR Hugging Face

## 3 Langkah Deploy (5 menit)

### Step 1: Buat Space
```
1. huggingface.co/spaces
2. Create new Space
3. SDK: Gradio
4. Hardware: CPU (free)
```

### Step 2: Upload 3 Files
```
app.py              → Drag & drop
requirements.txt    → Drag & drop
README.md           → Drag & drop
```

### Step 3: Commit & Wait
```
Commit message: "Initial demo"
Wait: 5-10 menit (first build)
Access: https://huggingface.co/spaces/[USER]/tb-detector-demo
```

---

## 🔗 File yang Diupload

| File | Size | Purpose |
|------|------|---------|
| `app.py` | ~8 KB | Main application |
| `requirements.txt` | ~300 B | Dependencies |
| `README.md` | ~5 KB | Documentation |

**Total: ~14 KB** (model akan auto-download, ~200 MB)

---

## ⚡ Copy-Paste Commands

### Clone repo lokal (optional)
```bash
git clone https://huggingface.co/spaces/[YOUR_USERNAME]/tb-detector-demo
cd tb-detector-demo
```

### Test locally
```bash
pip install -r requirements.txt
python app.py
```

---

## 🎯 Done!

Your TB Detector demo is now live at:
```
https://huggingface.co/spaces/[YOUR_USERNAME]/tb-detector-demo
```

Share this link with anyone! 🎉
