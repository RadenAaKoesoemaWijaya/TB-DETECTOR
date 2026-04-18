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
echo   Upgrading setuptools first...
pip install --upgrade setuptools wheel

REM Check Python version untuk torch compatibility
for /f "tokens=2" %%I in ('%PYTHON_CMD% --version 2^>^&1') do set PYVER=%%I
echo   Detected Python version: %PYVER%

echo   Installing torch + torchaudio...
REM For Python 3.14, gunakan torch 2.10+ atau latest available
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo   [WARNING] Failed to install torch from index, trying default...
    pip install torch torchaudio
)

echo   Installing numpy (pre-built wheel untuk avoid compilation)...
pip install numpy --only-binary :all:
if errorlevel 1 (
    echo   [WARNING] Trying numpy without binary restriction...
    pip install numpy
)

echo   Installing other packages...
pip install fastapi uvicorn python-multipart transformers librosa scikit-learn pandas pydantic soundfile onnx onnxruntime pydub webrtcvad-wheels matplotlib seaborn tensorboard tqdm requests pillow aiofiles

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
