@echo off
echo ===========================================
echo   TB DETECTOR v3.2 - Integrated AI Pipeline
echo ===========================================
echo.
echo Phase 1 - Performance Optimizations:
echo   - Batch Training (10-20x faster)
echo   - Feature Caching
echo   - SQLite Persistence
echo.
echo Phase 2 - Production Features:
echo   - Async I/O Operations
echo   - Background Task Queue
echo   - ONNX Inference (2-3x faster)
echo   - Model Versioning (Dev/Staging/Prod)
echo   - A/B Testing Framework
echo.
echo Supported Backbones:
echo   - Google HeAR (1024d)
echo   - Wav2Vec 2.0 Base (768d)
echo   - Wav2Vec 2.0 XLS-R (1024d)
echo   - HuBERT (768d/1024d)
echo.

REM Find Python
set "PYTHON_CMD="

REM Try common Python locations
if exist "C:\Python39\python.exe" (
    set "PYTHON_CMD=C:\Python39\python.exe"
) else if exist "C:\Python310\python.exe" (
    set "PYTHON_CMD=C:\Python310\python.exe"
) else if exist "C:\Python311\python.exe" (
    set "PYTHON_CMD=C:\Python311\python.exe"
) else if exist "C:\Python312\python.exe" (
    set "PYTHON_CMD=C:\Python312\python.exe"
) else if exist "C:\Python313\python.exe" (
    set "PYTHON_CMD=C:\Python313\python.exe"
) else (
    REM Try py launcher
    py --version >nul 2>&1
    if %errorlevel% == 0 (
        set "PYTHON_CMD=py"
    ) else (
        echo [ERROR] Python tidak ditemukan!
        echo.
        echo Jalankan fix_dependencies.bat terlebih dahulu
        echo atau install Python dari https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo.
echo ===========================================
echo Starting TB Detector v3.2 Server...
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
echo Phase 2 APIs:
echo   - /tasks/queue          - Background task queue
echo   - /registry/models      - Model versioning
echo   - /experiments/list     - A/B testing
echo.
echo Press Ctrl+C to stop the server
echo.

call venv\Scripts\activate.bat

python -m uvicorn app.main_v3:app --host 0.0.0.0 --port 8000 --reload

pause
