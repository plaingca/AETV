param(
    [ValidateSet("cpu", "cuda")]
    [string]$Runtime = "cpu",
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$BuildRoot = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot ".build\windows-$Runtime"))
$DistRoot = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot "dist\windows-$Runtime"))
if (-not $BuildRoot.StartsWith($RepoRoot) -or -not $DistRoot.StartsWith($RepoRoot)) {
    throw "Refusing to build outside the repository"
}

$RequiredModels = @(
    (Join-Path $RepoRoot "models\v8-hf3k-face-gan.pt"),
    (Join-Path $RepoRoot "models\v8-flex8k-ota-rxfix.pt")
)

New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
uv venv (Join-Path $BuildRoot "venv") --python 3.12 --clear
$Python = Join-Path $BuildRoot "venv\Scripts\python.exe"
$TorchIndex = if ($Runtime -eq "cuda") {
    "https://download.pytorch.org/whl/cu128"
} else {
    "https://download.pytorch.org/whl/cpu"
}
uv pip install --python $Python torch --index-url $TorchIndex
uv pip install --python $Python "$RepoRoot[gui]" pyinstaller
& $Python (Join-Path $RepoRoot "scripts\fetch_release_models.py") `
    --output (Join-Path $RepoRoot "models")
foreach ($Model in $RequiredModels) {
    if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
        throw "Missing release model after verified download: $Model"
    }
}

if (Test-Path -LiteralPath $DistRoot) {
    Remove-Item -LiteralPath $DistRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null
$WorkPath = Join-Path $BuildRoot "pyinstaller"
$SpecPath = Join-Path $BuildRoot "spec"

$Common = @(
    "--noconfirm", "--clean", "--onedir",
    "--workpath", $WorkPath,
    "--specpath", $SpecPath,
    "--distpath", $DistRoot,
    "--add-data", "$($RequiredModels[0]);models",
    "--add-data", "$($RequiredModels[1]);models",
    "--add-data", "$(Join-Path $RepoRoot 'aetv\assets');aetv/assets"
)

& $Python -m PyInstaller @Common --windowed --name AETV `
    --icon (Join-Path $RepoRoot "aetv\assets\aetv.ico") `
    (Join-Path $RepoRoot "aetv\gui\app.py")

& $Python -m PyInstaller @Common --console --name AETV-Benchmark `
    (Join-Path $RepoRoot "scripts\benchmark_inference.py")

$AppDir = Join-Path $DistRoot "AETV"
Copy-Item -LiteralPath (Join-Path $DistRoot "AETV-Benchmark\AETV-Benchmark.exe") -Destination $AppDir
Copy-Item -LiteralPath (Join-Path $RepoRoot "README.md") -Destination $AppDir
Copy-Item -LiteralPath (Join-Path $RepoRoot "LICENSE") -Destination $AppDir
Copy-Item -LiteralPath (Join-Path $RepoRoot "NOTICE") -Destination $AppDir

$PreviousOffline = $env:AETV_OFFLINE
try {
    $env:AETV_OFFLINE = "1"
    Push-Location $AppDir
    try {
        $Smoke = & ".\AETV-Benchmark.exe" --mode V8 --device cpu --warmup 0 --repeats 1
        if ($LASTEXITCODE -ne 0) {
            throw "Packaged benchmark smoke test failed"
        }
    } finally {
        Pop-Location
    }
} finally {
    $env:AETV_OFFLINE = $PreviousOffline
}
$Smoke | Set-Content -LiteralPath (Join-Path $AppDir "build-smoke.json") -Encoding utf8

if (-not $NoZip) {
    $Zip = Join-Path $DistRoot "AETV-windows-x64-$Runtime.zip"
    if (Test-Path -LiteralPath $Zip) {
        Remove-Item -LiteralPath $Zip -Force
    }
    tar.exe -a -c -f $Zip -C $DistRoot AETV
}

Write-Host "Portable AETV $Runtime build: $AppDir"
