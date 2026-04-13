@echo off
echo ===========================================
echo   TB DETECTOR v3 - Integrated AI Pipeline
echo ===========================================
echo.
echo Complete workflow:
echo   1. Upload Dataset (ZIP)
echo   2. Preprocessing
echo   3. Multi-Model Training
echo   4. Visualization
echo   5. Save Model
echo   6. Prediction
echo.
echo Supported Backbones:
echo   - Google HeAR (1024d)
echo   - Wav2Vec 2.0 Base (768d)
echo   - Wav2Vec 2.0 XLS-R (1024d)
echo   - HuBERT (768d/1024d)
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo.
echo ===========================================
echo Starting TB Detector v3 Server...
echo ===========================================
echo.
echo Server URL: http://localhost:8000
echo API Docs:   http://localhost:8000/docs
echo.
echo Quick Start:
echo   1. Open http://localhost:8000 in browser
echo   2. Upload your CODA TB DREAM dataset ZIP
echo   3. Follow the pipeline steps
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn app.main_v3:app --host 0.0.0.0 --port 8000 --reload

pause
