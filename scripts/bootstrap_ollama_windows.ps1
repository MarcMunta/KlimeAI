param(
    [switch]$StartOllama,
    [switch]$PullModels
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pullScript = Join-Path $repoRoot "scripts\pull_local_lab_models.ps1"
$candidatePaths = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"),
    "C:\Program Files\Ollama\ollama.exe"
)

$ollamaBinary = $null
foreach ($candidate in $candidatePaths) {
    if ($candidate -and (Test-Path $candidate)) {
        $ollamaBinary = $candidate
        break
    }
}

if (-not $ollamaBinary) {
    throw "Ollama was not found on this Windows host."
}

if ($StartOllama) {
    $listening = $false
    try {
        $probe = Test-NetConnection -ComputerName 127.0.0.1 -Port 11434 -WarningAction SilentlyContinue
        $listening = [bool]$probe.TcpTestSucceeded
    } catch {
        $listening = $false
    }

    if (-not $listening) {
        Start-Process -FilePath $ollamaBinary
        Start-Sleep -Seconds 5
    }
}

if ($PullModels) {
    & powershell -ExecutionPolicy Bypass -File $pullScript
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to pull Ollama models on Windows."
    }
}

Write-Output "Windows Ollama is available at: $ollamaBinary"
