@echo off
echo ===================================
echo   TB Detector - Model Training
echo ===================================
echo.
echo This will train multiple models:
echo   1. Wav2Vec 2.0 Base
echo   2. Wav2Vec 2.0 XLS-R
echo   3. Google HeAR (Proxy)
echo.
echo Requirements:
echo   - Dataset in data/coda_tb_dataset.csv
echo   - Audio files in data/audio/
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo.
echo Starting multi-backbone training...
echo This may take several hours depending on dataset size.
echo.

python train_multi_backbone.py

echo.
echo Training complete!
echo Models saved in: app/models/weights/
echo.
pause
