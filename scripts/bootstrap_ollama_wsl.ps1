param(
    [switch]$InstallUbuntu,
    [switch]$InstallOllama,
    [switch]$StartOllama,
    [switch]$PullModels,
    [string]$Distro = "Ubuntu"
)

function Invoke-InWsl {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command
    )

    wsl -d $Distro -- bash -lc $Command
    if ($LASTEXITCODE -ne 0) {
        throw "WSL command failed: $Command"
    }
}

$wslList = wsl -l -v 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "WSL is not available on this machine."
}

if ($wslList -notmatch $Distro) {
    if (-not $InstallUbuntu) {
        Write-Output "$Distro is not installed in WSL. Re-run with -InstallUbuntu to install it."
        exit 1
    }
    wsl --install -d $Distro
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install $Distro in WSL."
    }
}

Write-Output "$Distro is present in WSL."

if ($InstallOllama) {
    Invoke-InWsl "curl -fsSL https://ollama.com/install.sh | sh"
    Write-Output "Ollama installed inside $Distro."
}

if ($StartOllama) {
    Invoke-InWsl "nohup env OLLAMA_HOST=127.0.0.1:11434 ollama serve > ~/ollama.log 2>&1 < /dev/null &"
    Start-Sleep -Seconds 2
    Write-Output "Ollama serve started in $Distro with host 127.0.0.1:11434."
}

if ($PullModels) {
    $models = @(
        "qwen2.5-coder:14b-instruct-q4_K_S",
        "qwen3:14b",
        "nomic-embed-text"
    )
    foreach ($model in $models) {
        Invoke-InWsl "ollama pull '$model'"
    }
    Write-Output "Required local-lab models pulled."
}

Write-Output "Recommended next steps:"
Write-Output "1. powershell -ExecutionPolicy Bypass -File scripts\\pull_local_lab_models.ps1"
Write-Output "2. Start Docker Desktop"
Write-Output "3. powershell -ExecutionPolicy Bypass -File scripts\\start_local_stack.ps1"
