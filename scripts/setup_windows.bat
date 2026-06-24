@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0.."

echo ============================================================
echo VisionTrain Studio - Windows Setup
echo ============================================================
echo This script creates a local environment and installs dependencies.
echo It never installs TensorRT automatically. TensorRT must be installed separately.
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found in PATH.
  echo Install Python 3.10/3.11 from https://www.python.org/downloads/ or install Anaconda/Miniconda.
  pause
  exit /b 1
)
python --version

where conda >nul 2>nul
if errorlevel 1 (
  set HAS_CONDA=0
  echo [INFO] Conda was not found. Using Python venv fallback.
) else (
  set HAS_CONDA=1
  echo [INFO] Conda detected.
)

echo.
echo Choose installation mode:
echo 1. CPU only
echo 2. NVIDIA GPU / CUDA
echo.
set /p INSTALL_MODE="Enter choice [1/2]: "
if "%INSTALL_MODE%"=="2" (
  set TORCH_MODE=gpu
) else (
  set TORCH_MODE=cpu
)

echo.
echo [INFO] Creating required folders...
if not exist backend\runs mkdir backend\runs
if not exist backend\runs\exports mkdir backend\runs\exports
if not exist backend\datasets mkdir backend\datasets
if not exist backend\logs mkdir backend\logs
if not exist backend\configs\saved mkdir backend\configs\saved
if not exist docs\screenshots mkdir docs\screenshots

if "%HAS_CONDA%"=="1" (
  echo [INFO] Creating/updating Conda environment: visiontrain
  conda env list | findstr /C:"visiontrain" >nul
  if errorlevel 1 (
    conda create -y -n visiontrain python=3.10
    if errorlevel 1 goto fail
  ) else (
    echo [INFO] Conda environment visiontrain already exists.
  )
  set PIP_CMD=conda run -n visiontrain python -m pip
  set PY_CMD=conda run -n visiontrain python
) else (
  if not exist .venv\Scripts\python.exe (
    echo [INFO] Creating .venv fallback...
    python -m venv .venv
    if errorlevel 1 goto fail
  )
  set PIP_CMD=.venv\Scripts\python.exe -m pip
  set PY_CMD=.venv\Scripts\python.exe
)

echo [INFO] Upgrading pip...
%PIP_CMD% install --upgrade pip setuptools wheel
if errorlevel 1 goto fail

echo [INFO] Installing PyTorch based on selected mode: %TORCH_MODE%
if "%TORCH_MODE%"=="gpu" (
  echo [INFO] Installing CUDA PyTorch from official PyTorch CUDA 12.1 index.
  echo [INFO] If your driver does not support this build, rerun setup and choose CPU or adjust the index URL.
  %PIP_CMD% install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  if errorlevel 1 goto fail
) else (
  echo [INFO] Installing CPU PyTorch from official CPU index.
  %PIP_CMD% install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 goto fail
)

echo [INFO] Installing application dependencies...
%PIP_CMD% install fastapi "uvicorn[standard]" python-multipart pydantic ultralytics timm opencv-python pillow numpy pandas pyyaml psutil tqdm scikit-learn onnx onnxruntime
if errorlevel 1 goto fail

echo [INFO] Running dependency check...
set CHECK_FILE=backend\logs\dependency_check.py
> %CHECK_FILE% echo import importlib, sys
>> %CHECK_FILE% echo required = ['fastapi','uvicorn','pydantic','torch','torchvision','ultralytics','timm','cv2','PIL','numpy','pandas','yaml','psutil','tqdm','sklearn']
>> %CHECK_FILE% echo failed=[]
>> %CHECK_FILE% echo for name in required:
>> %CHECK_FILE% echo     try: importlib.import_module(name)
>> %CHECK_FILE% echo     except Exception as exc: failed.append((name, str(exc)))
>> %CHECK_FILE% echo if failed:
>> %CHECK_FILE% echo     print('[ERROR] Missing imports:')
>> %CHECK_FILE% echo     [print(' -', n, e) for n,e in failed]
>> %CHECK_FILE% echo     sys.exit(1)
>> %CHECK_FILE% echo import torch
>> %CHECK_FILE% echo print('[OK] PyTorch:', torch.__version__)
>> %CHECK_FILE% echo print('[OK] CUDA available:', torch.cuda.is_available())
%PY_CMD% %CHECK_FILE%
if errorlevel 1 goto fail

if "%TORCH_MODE%"=="gpu" call :gpu_check
if errorlevel 1 goto fail

echo.
echo ============================================================
echo [SUCCESS] VisionTrain Studio setup completed.
echo Run: powershell -ExecutionPolicy Bypass -File scripts\start_visiontrain.ps1
echo ============================================================
pause
exit /b 0

:gpu_check
echo [INFO] Running GPU validation test...
set GPU_CHECK_FILE=backend\logs\gpu_setup_check.py
> %GPU_CHECK_FILE% echo import sys, torch
>> %GPU_CHECK_FILE% echo print('CUDA available:', torch.cuda.is_available())
>> %GPU_CHECK_FILE% echo if not torch.cuda.is_available():
>> %GPU_CHECK_FILE% echo     print('[WARNING] GPU mode was selected, but CUDA is not available.')
>> %GPU_CHECK_FILE% echo     print('Suggested fixes: update NVIDIA driver, install matching CUDA PyTorch, or rerun setup in CPU mode.')
>> %GPU_CHECK_FILE% echo     sys.exit(0)
>> %GPU_CHECK_FILE% echo print('GPU count:', torch.cuda.device_count())
>> %GPU_CHECK_FILE% echo print('GPU name:', torch.cuda.get_device_name(0))
>> %GPU_CHECK_FILE% echo x=torch.randn(64,64,device='cuda'); y=x@x
>> %GPU_CHECK_FILE% echo print('[OK] GPU tensor test:', float(y.mean().detach().cpu()))
%PY_CMD% %GPU_CHECK_FILE%
if errorlevel 1 exit /b 1
exit /b 0

:fail
echo.
echo ============================================================
echo [ERROR] Setup failed. Read the message above.
echo Common fixes:
echo - Ensure Python/Conda is installed and available in PATH.
echo - Rerun terminal as Administrator only if folder permissions fail.
echo - For GPU, install/update NVIDIA driver and rerun setup.
echo - For CPU-only usage, rerun setup and choose option 1.
echo ============================================================
pause
exit /b 1
