@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
where conda >nul 2>nul
if not errorlevel 1 (
  conda env list | findstr /C:"visiontrain" >nul
  if not errorlevel 1 (
    echo [INFO] Opening VisionTrain frontend at http://127.0.0.1:5173
    start http://127.0.0.1:5173
    cd frontend
    conda run -n visiontrain python -m http.server 5173 --bind 127.0.0.1
    pause
    exit /b %errorlevel%
  )
)
if exist .venv\Scripts\python.exe (
  echo [INFO] Opening VisionTrain frontend at http://127.0.0.1:5173
  start http://127.0.0.1:5173
  cd frontend
  ..\.venv\Scripts\python.exe -m http.server 5173 --bind 127.0.0.1
  pause
  exit /b %errorlevel%
)
echo [ERROR] No environment found. Run scripts\setup_windows.bat first.
pause
exit /b 1
