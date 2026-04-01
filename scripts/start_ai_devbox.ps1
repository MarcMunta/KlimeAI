param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectPath,
    [string]$Image = "vortex-ai-devbox:latest"
)

$resolvedProject = (Resolve-Path $ProjectPath).Path
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$dockerfileDir = Join-Path $repoRoot "infra\local-lab\ai-devbox"

docker image inspect $Image *> $null
if ($LASTEXITCODE -ne 0) {
    docker build -t $Image $dockerfileDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to build ai-devbox image"
    }
}

docker run --rm -it `
    --network none `
    --mount "type=bind,source=$resolvedProject,target=/workspace" `
    --workdir /workspace `
    $Image
