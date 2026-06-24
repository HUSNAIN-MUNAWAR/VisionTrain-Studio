@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
where conda >nul 2>nul
if not errorlevel 1 (
  conda env list | findstr /C:"visiontrain" >nul
  if not errorlevel 1 (
    set PY_CMD=conda run -n visiontrain python
    goto runcheck
  )
)
if exist .venv\Scripts\python.exe (
  set PY_CMD=.venv\Scripts\python.exe
  goto runcheck
)
set PY_CMD=python

:runcheck
if not exist backend\logs mkdir backend\logs
set CHECK_FILE=backend\logs\gpu_check_runtime.py
> %CHECK_FILE% echo import sys, platform, torch
>> %CHECK_FILE% echo print('Python path:', sys.executable)
>> %CHECK_FILE% echo print('Python version:', platform.python_version())
>> %CHECK_FILE% echo print('PyTorch version:', torch.__version__)
>> %CHECK_FILE% echo print('CUDA available:', torch.cuda.is_available())
>> %CHECK_FILE% echo print('CUDA version:', torch.version.cuda)
>> %CHECK_FILE% echo print('cuDNN version:', torch.backends.cudnn.version())
>> %CHECK_FILE% echo print('GPU count:', torch.cuda.device_count())
>> %CHECK_FILE% echo if torch.cuda.is_available():
>> %CHECK_FILE% echo     idx=0
>> %CHECK_FILE% echo     props=torch.cuda.get_device_properties(idx)
>> %CHECK_FILE% echo     print('GPU name:', props.name)
>> %CHECK_FILE% echo     print('VRAM GB:', round(props.total_memory/1024**3, 2))
>> %CHECK_FILE% echo     x=torch.randn(128,128,device='cuda'); y=x@x
>> %CHECK_FILE% echo     print('Tensor test OK:', float(y.mean().detach().cpu()))
>> %CHECK_FILE% echo else:
>> %CHECK_FILE% echo     print('No CUDA GPU available to PyTorch.')
%PY_CMD% %CHECK_FILE%
pause
