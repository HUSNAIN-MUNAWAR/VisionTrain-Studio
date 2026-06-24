@echo off
cd /d "%~dp0.."
echo This will delete local training runs, exports, and logs.
set /p CONFIRM="Type DELETE to continue: "
if /I not "%CONFIRM%"=="DELETE" (
  echo Cancelled.
  pause
  exit /b 0
)
if exist backend\runs rmdir /s /q backend\runs
if exist backend\logs rmdir /s /q backend\logs
mkdir backend\runs
mkdir backend\runs\exports
mkdir backend\logs
echo Clean complete.
pause
