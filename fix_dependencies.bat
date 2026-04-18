@echo off
chcp 65001 >nul
echo ===========================================
echo   TB DETECTOR - Fix Dependencies
echo ===========================================
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
        echo Silakan install Python 3.9 - 3.13 dari:
        echo https://www.python.org/downloads/
        echo.
        echo Atau tambahkan Python ke PATH system environment variable.
        pause
        exit /b 1
    )
)

echo [INFO] Using Python: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

echo [1/4] Removing old virtual environment...
if exist venv (
    rmdir /s /q venv
    echo   [OK] Old venv removed
) else (
    echo   [INFO] No venv to remove
)
echo.

echo [2/4] Creating fresh virtual environment...
%PYTHON_CMD% -m venv venv
if errorlevel 1 (
    echo   [ERROR] Failed to create venv!
    pause
    exit /b 1
)
echo   [OK] Virtual environment created
echo.

echo [3/4] Activating and upgrading pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo   [OK] Pip upgraded
echo.

echo [4/4] Installing dependencies (this may take a few minutes)...
echo   Upgrading setuptools first (for Python 3.13 compatibility)...
pip install --upgrade setuptools wheel

echo   Installing torch 2.6.0 + torchaudio 2.6.0...
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

echo   Installing other packages...
pip install fastapi==0.104.1 uvicorn==0.24.0 python-multipart==0.0.6 transformers==4.35.2 librosa==0.10.1 numpy==1.26.4 scikit-learn==1.3.2 pandas==2.0.3 pydantic==2.5.0 soundfile==0.12.1 onnx==1.15.0 onnxruntime==1.16.3 pydub==0.25.1 webrtcvad==2.0.10 matplotlib==3.8.2 seaborn==0.13.0 tensorboard==2.15.1 tqdm==4.66.1 requests==2.31.0 pillow==10.1.0

if errorlevel 1 (
    echo   [WARNING] Some packages may have issues
) else (
    echo   [OK] All dependencies installed
)
echo.

echo ===========================================
echo   Dependencies Fixed!
echo ===========================================
echo.
echo You can now run the server with:
echo   .\start_v3.bat
echo.
pause
