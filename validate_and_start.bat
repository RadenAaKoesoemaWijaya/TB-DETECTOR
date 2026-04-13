@echo off
chcp 65001 >nul
echo ===========================================
echo   TB DETECTOR - Validation & Startup
echo ===========================================
echo.

set ERRORS=0
set WARNINGS=0

echo [1/5] Checking directory structure...
if not exist app\models\weights (
    echo   Creating app\models\weights...
    mkdir app\models\weights
)
if not exist data (
    echo   Creating data...
    mkdir data
)
if not exist data\uploaded_dataset (
    echo   Creating data\uploaded_dataset...
    mkdir data\uploaded_dataset
)
echo   [OK] Directories verified
echo.

echo [2/5] Checking required files...
if not exist app\main_v3.py (
    echo   [ERROR] app\main_v3.py not found!
    set /a ERRORS+=1
) else (
    echo   [OK] app\main_v3.py
)

if not exist app\static\index_v3.html (
    echo   [ERROR] app\static\index_v3.html not found!
    set /a ERRORS+=1
) else (
    echo   [OK] app\static\index_v3.html
)

if not exist requirements.txt (
    echo   [ERROR] requirements.txt not found!
    set /a ERRORS+=1
) else (
    echo   [OK] requirements.txt
)
echo.

echo [3/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo   [ERROR] Python not found in PATH!
    set /a ERRORS+=1
) else (
    for /f "tokens=*" %%a in ('python --version 2^>^&1') do echo   [OK] %%a
)
echo.

echo [4/5] Checking virtual environment...
if not exist venv\Scripts\activate.bat (
    echo   [INFO] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo   [ERROR] Failed to create venv!
        set /a ERRORS+=1
    ) else (
        echo   [OK] Virtual environment created
    )
) else (
    echo   [OK] Virtual environment exists
)
echo.

echo [5/5] Installing dependencies...
call venv\Scripts\activate.bat >nul
pip install -q -r requirements.txt
if errorlevel 1 (
    echo   [WARNING] Some dependencies may have issues
    set /a WARNINGS+=1
) else (
    echo   [OK] Dependencies installed
)
echo.

if %ERRORS% GTR 0 (
    echo ===========================================
    echo   VALIDATION FAILED: %ERRORS% error(s)
    echo ===========================================
    echo.
    echo Please fix the errors above before starting.
    pause
    exit /b 1
)

echo ===========================================
echo   VALIDATION PASSED! Starting server...
echo ===========================================
echo   Errors: %ERRORS% ^| Warnings: %WARNINGS%
echo.

python -m uvicorn app.main_v3:app --host 0.0.0.0 --port 8000 --reload

pause
