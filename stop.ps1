$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
  Write-Host ("[stop] " + $Message)
}

function Stop-Tree([int]$Pid) {
  & taskkill /PID $Pid /T >$null 2>&1
  Start-Sleep -Milliseconds 300
  & taskkill /PID $Pid /T /F >$null 2>&1
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidsDir = Join-Path $root ".pids"
if (-not (Test-Path -LiteralPath $pidsDir)) {
  Write-Step "No .pids/ directory."
  exit 0
}

$pidFiles = Get-ChildItem -LiteralPath $pidsDir -Filter *.pid -ErrorAction SilentlyContinue
if (-not $pidFiles) {
  Write-Step "No PID files found."
  exit 0
}

foreach ($pf in $pidFiles) {
  $name = $pf.BaseName
  $raw = (Get-Content -LiteralPath $pf.FullName -ErrorAction SilentlyContinue | Select-Object -First 1)
  $procId = 0
  [int]::TryParse(($raw -as [string]), [ref]$procId) | Out-Null
  if ($procId -gt 0) {
    Write-Step "Stopping $name (pid=$procId)..."
    Stop-Tree -Pid $procId
  }
  Remove-Item -Force -LiteralPath $pf.FullName -ErrorAction SilentlyContinue
}

Write-Step "Done."
