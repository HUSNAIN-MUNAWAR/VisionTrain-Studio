@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
if not exist backend\logs mkdir backend\logs
where conda >nul 2>nul
if not errorlevel 1 (
  conda env list | findstr /C:"visiontrain" >nul
  if not errorlevel 1 (
    echo [INFO] Starting backend with Conda env visiontrain
    conda run -n visiontrain uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
    pause
    exit /b %errorlevel%
  )
)
if exist .venv\Scripts\uvicorn.exe (
  echo [INFO] Starting backend with .venv
  .venv\Scripts\uvicorn.exe backend.main:app --reload --host 127.0.0.1 --port 8000
  pause
  exit /b %errorlevel%
)
echo [ERROR] No environment found. Run scripts\setup_windows.bat first.
pause
exit /b 1
