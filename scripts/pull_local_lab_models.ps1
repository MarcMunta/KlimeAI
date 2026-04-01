param(
    [string]$Distro = "Ubuntu",
    [switch]$IncludeDeepCoder
)

$models = @(
    "qwen2.5-coder:14b-instruct-q4_K_S",
    "qwen3:14b",
    "nomic-embed-text"
)

if ($IncludeDeepCoder) {
    $models += "qwen3-coder:30b"
}

$ollamaBinary = $null
$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollamaCmd) {
    $ollamaBinary = $ollamaCmd.Source
} else {
    $candidatePaths = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"),
        "C:\Program Files\Ollama\ollama.exe"
    )
    foreach ($candidate in $candidatePaths) {
        if ($candidate -and (Test-Path $candidate)) {
            $ollamaBinary = $candidate
            break
        }
    }
}

if ($ollamaBinary) {
    foreach ($model in $models) {
        & $ollamaBinary pull $model
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to pull model with local Ollama: $model"
        }
    }
    Write-Output "Models pulled with local Ollama."
    exit 0
}

$wslList = wsl -l -v 2>&1
if ($LASTEXITCODE -ne 0 -or $wslList -notmatch $Distro) {
    throw "Ollama is not available locally and WSL distro '$Distro' was not found."
}

foreach ($model in $models) {
    wsl -d $Distro -- bash -lc "ollama pull '$model'"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to pull model in WSL: $model"
    }
}

Write-Output "Models pulled through WSL Ollama."
