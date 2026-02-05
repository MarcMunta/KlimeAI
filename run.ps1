param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
  Write-Host ("[run] " + $Message)
}

function Fail([string]$Message, [int]$Code = 1) {
  Write-Host ("[run] ERROR: " + $Message) -ForegroundColor Red
  exit $Code
}

function Has-Command([string]$Name) {
  return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Ensure-Dir([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
  }
}

function Start-LoggedProcess(
  [string]$Name,
  [string]$WorkingDir,
  [string]$CommandLine,
  [string]$LogPath,
  [string]$PidPath
) {
  Ensure-Dir (Split-Path -Parent $LogPath)
  Ensure-Dir (Split-Path -Parent $PidPath)
  if (Test-Path -LiteralPath $LogPath) { Remove-Item -Force -LiteralPath $LogPath -ErrorAction SilentlyContinue }

  $cmd = "cd /d `"$WorkingDir`" && $CommandLine > `"$LogPath`" 2>&1"
  $proc = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", $cmd) -WindowStyle Hidden -PassThru
  Set-Content -LiteralPath $PidPath -Value $proc.Id -Encoding ascii
  Write-Step "$Name started (pid=$($proc.Id))"
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Defaults
$runBack = $true
$runFront = $true
$runSelfTrain = (($env:ENABLE_SELF_TRAIN -as [string]) -eq "1")
$runAutoEdits = (($env:ENABLE_AUTO_EDITS -as [string]) -eq "1")

foreach ($arg in ($Args | ForEach-Object { $_.Trim() })) {
  switch -Regex ($arg) {
    '^--all$' { }
    '^--front-only$' { $runBack = $false; $runFront = $true }
    '^--back-only$' { $runBack = $true; $runFront = $false }
    '^--no-self-train$' { $runSelfTrain = $false }
    '^--no-auto-edits$' { $runAutoEdits = $false }
    '^--help$' {
      @"
Vortex one-command runner (Windows)

Usage:
  .\run.bat [--all] [--front-only|--back-only] [--no-self-train] [--no-auto-edits]

Env:
  C3RNT2_PROFILE=dev_small
  VORTEX_BACKEND_PORT=8000
  VORTEX_FRONTEND_PORT=5173
  ENABLE_SELF_TRAIN=1
  ENABLE_AUTO_EDITS=1
"@ | Write-Host
      exit 0
    }
    default { if ($arg) { Fail "Unknown arg: $arg" } }
  }
}

$profile = ($env:C3RNT2_PROFILE | ForEach-Object { $_.Trim() }) 
if (-not $profile) { $profile = "dev_small" }

$backendPort = $env:VORTEX_BACKEND_PORT
if (-not $backendPort) { $backendPort = $env:BACKEND_PORT }
if (-not $backendPort) { $backendPort = "8000" }

$frontendPort = $env:VORTEX_FRONTEND_PORT
if (-not $frontendPort) { $frontendPort = $env:FRONTEND_PORT }
if (-not $frontendPort) { $frontendPort = "5173" }

$logsDir = Join-Path $root "logs"
$pidsDir = Join-Path $root ".pids"
Ensure-Dir $logsDir
Ensure-Dir $pidsDir

$needPython = $runBack -or $runSelfTrain -or $runAutoEdits
$needNode = $runFront

if ($needPython -and -not (Has-Command "python")) { Fail "Python not found in PATH." }
if ($needNode -and (-not (Has-Command "node") -or -not (Has-Command "npm"))) { Fail "Node.js/npm not found in PATH." }

# Python venv
$py = Join-Path $root ".venv\\Scripts\\python.exe"
if ($needPython) {
  if (-not (Test-Path -LiteralPath $py)) {
    Write-Step "Creating venv (.venv)..."
    python -m venv .venv
  }
  Write-Step "Checking Python deps..."
  & $py -c "import importlib.util as u; import sys; mods=['c3rnt2','fastapi','uvicorn']; miss=[m for m in mods if u.find_spec(m) is None]; sys.exit(0 if not miss else 1)" 2>$null | Out-Null
  if ($LASTEXITCODE -ne 0) {
    Write-Step "Installing backend deps (editable + api extras)..."
    & $py -m pip install -U pip
    & $py -m pip install -e "c3_rnt2_ai[api]"
  }
}

# Frontend deps
if ($runFront) {
  $frontendDir = Join-Path $root "frontend"
  if (-not (Test-Path -LiteralPath (Join-Path $frontendDir "node_modules"))) {
    Write-Step "Installing frontend deps (npm i)..."
    Push-Location $frontendDir
    try { npm i } finally { Pop-Location }
  }
  $frontendEnv = Join-Path $frontendDir ".env"
  if (-not (Test-Path -LiteralPath $frontendEnv)) {
    Copy-Item -Force -LiteralPath (Join-Path $frontendDir ".env.example") -Destination $frontendEnv
  }
}

# Start services
if ($runBack) {
  $backendDir = Join-Path $root "c3_rnt2_ai"
  Start-LoggedProcess `
    -Name "backend" `
    -WorkingDir $backendDir `
    -CommandLine "`"$py`" -m vortex serve --profile $profile --host 0.0.0.0 --port $backendPort" `
    -LogPath (Join-Path $logsDir "backend.log") `
    -PidPath (Join-Path $pidsDir "backend.pid")
}

if ($runFront) {
  $frontendDir = Join-Path $root "frontend"
  Start-LoggedProcess `
    -Name "frontend" `
    -WorkingDir $frontendDir `
    -CommandLine "npm run dev -- --host 0.0.0.0 --port $frontendPort" `
    -LogPath (Join-Path $logsDir "frontend.log") `
    -PidPath (Join-Path $pidsDir "frontend.pid")
}

if ($runSelfTrain) {
  $backendDir = Join-Path $root "c3_rnt2_ai"
  $intervalMin = $env:SELF_TRAIN_INTERVAL_MINUTES
  if (-not $intervalMin) { $intervalMin = "30" }
  Start-LoggedProcess `
    -Name "self-train" `
    -WorkingDir $backendDir `
    -CommandLine "set C3RNT2_NO_NET=1 && `"$py`" -m vortex self-train --profile $profile --interval-minutes $intervalMin" `
    -LogPath (Join-Path $logsDir "self-train.log") `
    -PidPath (Join-Path $pidsDir "self-train.pid")
}

if ($runAutoEdits) {
  $backendDir = Join-Path $root "c3_rnt2_ai"
  Start-LoggedProcess `
    -Name "auto-edits" `
    -WorkingDir $backendDir `
    -CommandLine "set C3RNT2_NO_NET=1 && set AUTO_EDITS_CREATE_DEMO=1 && `"$py`" scripts\\auto_edits_watcher.py --profile $profile --create-demo-on-start" `
    -LogPath (Join-Path $logsDir "auto-edits.log") `
    -PidPath (Join-Path $pidsDir "auto-edits.pid")
}

Write-Host ""
Write-Step "Summary"
if ($runBack) { Write-Host ("  Backend:  http://localhost:" + $backendPort) }
if ($runFront) { Write-Host ("  Frontend: http://localhost:" + $frontendPort) }
Write-Host ("  Logs:     " + $logsDir)
Write-Host ("  PIDs:     " + $pidsDir)
Write-Host ""
Write-Host "Use .\\status.bat to check, .\\logs.bat backend|frontend|self-train|auto-edits to tail, .\\stop.bat to stop."

