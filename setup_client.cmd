@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo ============================================================
echo  No Helmet Detection - Client Setup
echo ============================================================
echo.

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=py -3.9"
) else (
    set "PYTHON_CMD=python"
)

%PYTHON_CMD% -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 9) else 1)" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python 3.9 is required for this client package.
    echo Install Python 3.9.10 or another Python 3.9.x release, then run this file again.
    echo Download: https://www.python.org/downloads/release/python-3910/
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if %ERRORLEVEL% NEQ 0 goto :fail
)

call "venv\Scripts\activate.bat"
if %ERRORLEVEL% NEQ 0 goto :fail

set PYTHONIOENCODING=utf-8
set PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0
set PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
set YOLO_CONFIG_DIR=%CD%\.yolo_config

echo Upgrading pip tools...
python -m pip install --upgrade pip setuptools wheel
if %ERRORLEVEL% NEQ 0 goto :fail

echo Installing Python dependencies. This can take several minutes...
python -m pip install --prefer-binary -r requirements-client.txt
if %ERRORLEVEL% NEQ 0 goto :fail

if not exist ".env" (
    echo Creating .env from .env.example...
    copy ".env.example" ".env" >nul
)

echo Preparing models, database, and OCR runtime assets...
python "scripts\prepare_client_assets.py"
if %ERRORLEVEL% NEQ 0 goto :fail

echo.
echo Setup completed successfully.
echo Run the application with: run_app.cmd
echo.
pause
exit /b 0

:fail
echo.
echo Setup failed. Check the error above, then run setup_client.cmd again.
echo.
pause
exit /b 1
