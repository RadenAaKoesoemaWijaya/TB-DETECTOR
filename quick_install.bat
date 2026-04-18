@echo off
chcp 65001 >nul
echo ===========================================
echo   TB Detector v3.2 - Quick Install
===========================================
echo.
echo Mode: Compatible dengan Python 3.14
echo.

REM Find Python
set "PYTHON_CMD="

REM Try py launcher first (most reliable)
py --version >nul 2>&1
if %errorlevel% == 0 (
    set "PYTHON_CMD=py"
) else (
    echo [ERROR] Python (py launcher) tidak ditemukan!
    echo.
    echo Silakan install Python 3.10 - 3.14 dari https://www.python.org/downloads/
    echo Pastikan centang "Add Python to PATH" saat install.
    pause
    exit /b 1
)

echo [INFO] Using: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

REM Check if venv exists
if exist venv (
    echo [INFO] Virtual environment exists, using existing...
) else (
    echo [1/3] Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv!
        pause
        exit /b 1
    )
    echo   [OK] Virtual environment created
)

echo.
echo [2/3] Activating environment...
call venv\Scripts\activate.bat

echo.
echo [3/3] Installing packages (this will take several minutes)...
echo.

REM Install packages one by one untuk better error handling
echo   - Installing core dependencies...
pip install fastapi uvicorn python-multipart pydantic tqdm requests pillow

echo   - Installing ML libraries (PyTorch)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo   [INFO] Trying default PyTorch install...
    pip install torch torchaudio
)

echo   - Installing transformers...
pip install transformers

echo   - Installing audio processing...
pip install librosa soundfile pydub webrtcvad-wheels

echo   - Installing data science libraries...
pip install numpy pandas scikit-learn

echo   - Installing visualization...
pip install matplotlib seaborn tensorboard

echo   - Installing ONNX (optional)...
pip install onnx onnxruntime

echo.
echo ===========================================
echo   Installation Complete!
echo ===========================================
echo.
echo Jalankan server dengan:
echo   .\start_v3.bat
echo.
echo Atau manual:
echo   call venv\Scripts\activate.bat
echo   python -m uvicorn app.main_v3:app --host 0.0.0.0 --port 8000
echo.
pause
