@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found.
    echo Run setup_client.cmd first.
    pause
    exit /b 1
)

call "venv\Scripts\activate.bat"
set PYTHONIOENCODING=utf-8
set PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0
set PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
set YOLO_CONFIG_DIR=%CD%\.yolo_config

echo.
echo Starting No Helmet Detection...
echo Open this URL in the browser: http://127.0.0.1:5000/
echo Press CTRL+C in this window to stop the server.
echo.

python -m flask --app app run --host 127.0.0.1 --port 5000
