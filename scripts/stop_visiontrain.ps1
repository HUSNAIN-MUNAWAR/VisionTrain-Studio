$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$pidFile = Join-Path $ProjectRoot "backend\logs\visiontrain_processes.json"

if (-not (Test-Path $pidFile)) {
    Write-Host "No VisionTrain PID file found. Trying to stop common ports 8000 and 5173..." -ForegroundColor Yellow
} else {
    $data = Get-Content $pidFile -Raw | ConvertFrom-Json
    foreach ($pid in @($data.backend_pid, $data.frontend_pid)) {
        if ($pid) {
            try {
                Stop-Process -Id $pid -Force -ErrorAction Stop
                Write-Host "Stopped process $pid" -ForegroundColor Green
            } catch {
                Write-Host "Process $pid was not running or could not be stopped." -ForegroundColor DarkYellow
            }
        }
    }
}

foreach ($port in @(8000, 5173)) {
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in $connections) {
        try {
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
            Write-Host "Stopped process on port $port" -ForegroundColor Green
        } catch {}
    }
}
