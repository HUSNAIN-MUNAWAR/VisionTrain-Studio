$ErrorActionPreference = "Stop"

function Write-Stage($Message) {
    Write-Host "[VisionTrain] $Message" -ForegroundColor Cyan
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

$BackendLog = Join-Path $ProjectRoot "backend\logs\backend_runtime.log"
$FrontendLog = Join-Path $ProjectRoot "backend\logs\frontend_runtime.log"
New-Item -ItemType Directory -Force -Path "backend\runs", "backend\datasets", "backend\logs", "backend\configs\saved" | Out-Null

function Get-CondaBase {
    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $condaCmd) { return $null }
    try { return (& conda info --base) } catch { return $null }
}

$UseConda = $false
$UseVenv = $false
$CondaBase = Get-CondaBase
if ($CondaBase) {
    $envs = & conda env list | Out-String
    if ($envs -match "visiontrain") { $UseConda = $true }
}
if (-not $UseConda -and (Test-Path ".venv\Scripts\python.exe")) { $UseVenv = $true }

if (-not $UseConda -and -not $UseVenv) {
    Write-Host "No VisionTrain environment found." -ForegroundColor Red
    Write-Host "Run scripts\setup_windows.bat first. It will create Conda env 'visiontrain' or .venv fallback." -ForegroundColor Yellow
    exit 1
}

if ($UseConda) {
    Write-Stage "Using Conda environment: visiontrain"
    $PythonCmd = "conda run -n visiontrain python"
    $UvicornCmd = "conda run -n visiontrain uvicorn"
} else {
    Write-Stage "Using .venv fallback"
    $PythonCmd = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    $UvicornCmd = Join-Path $ProjectRoot ".venv\Scripts\uvicorn.exe"
}

Write-Stage "Starting FastAPI backend on http://127.0.0.1:8000"
if ($UseConda) {
    $backendArgs = @("/c", "conda run -n visiontrain uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000 1> `"$BackendLog`" 2>&1")
    $BackendProcess = Start-Process -FilePath "cmd.exe" -ArgumentList $backendArgs -PassThru -WindowStyle Hidden
} else {
    $backendArgs = @("/c", "`"$UvicornCmd`" backend.main:app --reload --host 127.0.0.1 --port 8000 1> `"$BackendLog`" 2>&1")
    $BackendProcess = Start-Process -FilePath "cmd.exe" -ArgumentList $backendArgs -PassThru -WindowStyle Hidden
}

Start-Sleep -Seconds 2
Write-Stage "Starting frontend static server on http://127.0.0.1:5173"
if ($UseConda) {
    $frontendArgs = @("/c", "cd /d `"$ProjectRoot\frontend`" && conda run -n visiontrain python -m http.server 5173 --bind 127.0.0.1 1> `"$FrontendLog`" 2>&1")
    $FrontendProcess = Start-Process -FilePath "cmd.exe" -ArgumentList $frontendArgs -PassThru -WindowStyle Hidden
} else {
    $frontendArgs = @("/c", "cd /d `"$ProjectRoot\frontend`" && `"$PythonCmd`" -m http.server 5173 --bind 127.0.0.1 1> `"$FrontendLog`" 2>&1")
    $FrontendProcess = Start-Process -FilePath "cmd.exe" -ArgumentList $frontendArgs -PassThru -WindowStyle Hidden
}

$pidFile = Join-Path $ProjectRoot "backend\logs\visiontrain_processes.json"
@{
    backend_pid = $BackendProcess.Id
    frontend_pid = $FrontendProcess.Id
    started_at = (Get-Date).ToString("s")
} | ConvertTo-Json | Set-Content -Encoding UTF8 $pidFile

Start-Sleep -Seconds 1
Start-Process "http://127.0.0.1:5173"
Write-Stage "Application is running. Browser opened."
Write-Host "Backend log:  $BackendLog" -ForegroundColor DarkGray
Write-Host "Frontend log: $FrontendLog" -ForegroundColor DarkGray
Write-Host "Press Ctrl+C to stop child processes." -ForegroundColor Yellow

try {
    while ($true) {
        if ($BackendProcess.HasExited) { throw "Backend process exited. Check $BackendLog" }
        if ($FrontendProcess.HasExited) { throw "Frontend process exited. Check $FrontendLog" }
        Start-Sleep -Seconds 2
    }
}
finally {
    Write-Stage "Stopping VisionTrain child processes..."
    foreach ($p in @($BackendProcess, $FrontendProcess)) {
        try {
            if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
        } catch {}
    }
    Write-Stage "Stopped."
}
