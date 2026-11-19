# Build and export the workspace-v2 SDK image (1.5).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\_export_image_15.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\_export_image_15.ps1 -SkipBuild
#
# ASCII-only output on purpose: Windows PowerShell 5.1 decodes this file with the
# system code page, so non-ASCII text would be garbled.
[CmdletBinding()]
param(
    [string]$Tag = "zhixun-harness:1.5",
    [string]$Tar = "",
    [string]$Platform = "linux/amd64",
    [switch]$SkipBuild,
    [switch]$SkipSave,
    [switch]$NoPull
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

if (-not $Tar) { $Tar = "submission\01-image\" + ($Tag -replace ":", "-") + ".tar" }
$Version = $Tag.Split(":")[1]

function Step($n, $text) { Write-Host ("[" + $n + "] " + $text) -ForegroundColor Cyan }
function Fail($text) { Write-Host ("FAILED: " + $text) -ForegroundColor Red; exit 1 }

# --- [0/6] docker engine -----------------------------------------------------
Step "0/6" "docker engine"
$server = (docker version --format "{{.Server.Version}}" 2>$null)
if ($LASTEXITCODE -ne 0 -or -not $server) {
    Write-Host "Docker engine is not reachable. Start Docker Desktop, then run:" -ForegroundColor Yellow
    Write-Host "  docker context use desktop-linux   (or: wsl --shutdown)"
    exit 1
}
Write-Host ("      server " + $server)

# --- [1/6] base image --------------------------------------------------------
Step "1/6" "base image python:3.10-slim"
docker image inspect python:3.10-slim *> $null
if ($LASTEXITCODE -ne 0) {
    if ($NoPull) { Fail "base image missing and -NoPull was given" }
    docker pull --platform $Platform python:3.10-slim
    if ($LASTEXITCODE -ne 0) {
        Fail "cannot pull python:3.10-slim. Add a registry mirror or pull it manually."
    }
}
$basePlatform = (docker image inspect --format "{{.Os}}/{{.Architecture}}" python:3.10-slim)
Write-Host ("      base " + $basePlatform)

# --- [2/6] build -------------------------------------------------------------
if (-not $SkipBuild) {
    Step "2/6" ("build " + $Tag)
    $sha = (git rev-parse --short HEAD 2>$null)
    if (-not $sha) { $sha = "nogit" }
    docker build --platform $Platform --build-arg ZHIXUN_VERSION=$Version --build-arg ZHIXUN_BUILD_SHA=$sha -t $Tag .
    if ($LASTEXITCODE -ne 0) { Fail "docker build failed (see log above)" }
} else {
    Step "2/6" "build skipped"
}
$imagePlatform = (docker image inspect --format "{{.Os}}/{{.Architecture}}" $Tag)
Write-Host ("      image " + $imagePlatform)
if ($imagePlatform -ne $Platform) { Fail ("image platform " + $imagePlatform + " != " + $Platform) }

# --- [3/6] describe ----------------------------------------------------------
Step "3/6" "describe contract"
$describe = (docker run --rm $Tag describe 2>$null)
if ($LASTEXITCODE -ne 0 -or -not $describe) { Fail "describe produced no JSON" }
$parsed = $describe | ConvertFrom-Json
if ($parsed.protocol -ne "contest.harness.workspace.v2") { Fail ("unexpected protocol: " + $parsed.protocol) }
Write-Host ("      protocol " + $parsed.protocol + " tasks " + ($parsed.tasks -join ","))

# --- [4/6] predict -----------------------------------------------------------
Step "4/6" "predict contract (one regression sample)"
$sample = & python "_scratch\sample_line.py" B
if ($LASTEXITCODE -ne 0 -or -not $sample) { Fail "cannot build a sample prediction line" }
$out = ($sample | docker run --rm -i -v "${here}\artifacts\B:/model:ro" -e "CONTEST_ARTIFACT_PATH=/model/checkpoint.npz" -e "CONTEST_DATASET_KEY=B" -e "CONTEST_TASK=regression" $Tag predict)
if ($LASTEXITCODE -ne 0) { Fail "predict exited non-zero" }
$prediction = $out | ConvertFrom-Json
if (-not $prediction.outputs.value) { Fail ("unexpected prediction output: " + $out) }
Write-Host ("      value " + $prediction.outputs.value)

# --- [5/6] import sanity -----------------------------------------------------
Step "5/6" "module import inside the image"
docker run --rm --entrypoint python3 $Tag -c "import contest_submission, rulkit.model, contest_harness; print('ok')"
if ($LASTEXITCODE -ne 0) { Fail "import check failed" }

# --- [6/6] save --------------------------------------------------------------
if (-not $SkipSave) {
    Step "6/6" ("save " + $Tar)
    $dir = Split-Path -Parent $Tar
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
    docker save -o $Tar $Tag
    if ($LASTEXITCODE -ne 0) { Fail "docker save failed" }
    $size = [math]::Round((Get-Item $Tar).Length / 1MB, 2)
    Write-Host ("      " + (Resolve-Path $Tar) + "  " + $size + " MiB")
} else {
    Step "6/6" "save skipped"
}

Write-Host ""
Write-Host ("DONE: " + $Tag)
Write-Host "Upload the tar in the submission page. Local re-check:"
Write-Host ("  docker load -i " + $Tar)
Write-Host ("  docker run --rm " + $Tag + " describe")