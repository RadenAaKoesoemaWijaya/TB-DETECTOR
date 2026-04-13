@echo off
chcp 65001 >nul
echo ===========================================
echo   TB DETECTOR - Install Dependencies
echo ===========================================
echo.

set ERRORS=0

call venv\Scripts\activate.bat

echo Step 1: Installing core packages...
pip install fastapi==0.104.1 uvicorn==0.24.0 python-multipart==0.0.6
if errorlevel 1 set /a ERRORS+=1

echo Step 2: Installing ML packages (numpy, scipy, sklearn)...
pip install numpy==1.26.4 scipy scikit-learn==1.3.2
if errorlevel 1 set /a ERRORS+=1

echo Step 3: Installing torch and torchaudio...
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 set /a ERRORS+=1

echo Step 4: Installing transformers and librosa...
pip install transformers==4.35.2 librosa==0.10.1
if errorlevel 1 set /a ERRORS+=1

echo Step 5: Installing other packages...
pip install pandas==2.0.3 pydantic==2.5.0 soundfile==0.12.1 matplotlib==3.8.2 seaborn==0.13.0 tqdm==4.66.1 requests==2.31.0 pillow==10.1.0
if errorlevel 1 set /a ERRORS+=1

echo.
echo ===========================================
if %ERRORS%==0 (
    echo   Installation Complete - No Errors
) else (
    echo   Installation Complete - %ERRORS% error(s)
)
echo ===========================================
echo.
pause
