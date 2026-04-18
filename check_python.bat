@echo off
echo ===========================================
echo   Check Python Installation
echo ===========================================
echo.

echo [1] Checking system PATH...
where python 2>nul
if %errorlevel% == 0 (
    echo   [OK] python command found
    python --version
) else (
    echo   [NOT FOUND] python command not in PATH
)
echo.

echo [2] Checking py launcher...
where py 2>nul
if %errorlevel% == 0 (
    echo   [OK] py launcher found
    py --version
) else (
    echo   [NOT FOUND] py launcher not available
)
echo.

echo [3] Checking common Python locations...
set "FOUND="
for %%P in (39 310 311 312 313) do (
    if exist "C:\Python%%P\python.exe" (
        echo   [FOUND] C:\Python%%P\python.exe
        "C:\Python%%P\python.exe" --version
        set "FOUND=1"
    )
)
if not defined FOUND (
    echo   [NOT FOUND] No Python in C:\PythonXX directories
)
echo.

echo [4] Checking Windows Store Python...
if exist "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" (
    echo   [WARNING] Windows Store Python detected
    echo            This may not work correctly.
    echo            Please install Python from python.org instead.
) else (
    echo   [OK] Windows Store Python not found
)
echo.

echo ===========================================
echo   Recommendations
echo ===========================================
echo.
echo Jika Python tidak ditemukan:
echo.
echo 1. Install Python dari https://www.python.org/downloads/
echo    - Pilih Python 3.10, 3.11, atau 3.12
echo    - Centang "Add Python to PATH" saat install
echo.
echo 2. Atau jalankan perintah ini di PowerShell Administrator:
echo    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
echo.
echo 3. Restart terminal setelah install Python
echo.
pause
