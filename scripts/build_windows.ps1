param(
    [ValidateSet("cpu", "gpu", "cuda")]
    [string]$Runtime = "cpu",
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$PackageRuntime = if ($Runtime -eq "cuda") { "gpu" } else { $Runtime }
$BuildRoot = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot ".build\windows-$PackageRuntime"))
$DistRoot = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot "dist\windows-$PackageRuntime"))
if (-not $BuildRoot.StartsWith($RepoRoot) -or -not $DistRoot.StartsWith($RepoRoot)) {
    throw "Refusing to build outside the repository"
}

New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
$HamlibDir = Join-Path $BuildRoot "hamlib"
& (Join-Path $RepoRoot "scripts\fetch_hamlib_windows.ps1") -Output $HamlibDir

uv venv (Join-Path $BuildRoot "runtime-venv") --python 3.12 --clear
$Python = Join-Path $BuildRoot "runtime-venv\Scripts\python.exe"
uv pip install --python $Python "$RepoRoot[gui]" pyinstaller
if ($PackageRuntime -eq "gpu") {
    uv pip uninstall --python $Python onnxruntime
    uv pip install --python $Python onnxruntime-directml
}
$RuntimeModelDir = Join-Path $BuildRoot "models"
& $Python (Join-Path $RepoRoot "scripts\fetch_release_runtime.py") `
    --output $RuntimeModelDir
if ($LASTEXITCODE -ne 0) { throw "Release runtime model download failed" }
$RuntimeModels = Get-ChildItem -LiteralPath $RuntimeModelDir -File

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
    "--exclude-module", "torch",
    "--exclude-module", "torchvision",
    "--exclude-module", "aetv.models",
    "--exclude-module", "aetv.channel",
    "--exclude-module", "aetv.data",
    "--exclude-module", "aetv.video_backbone",
    "--add-data", "$(Join-Path $RepoRoot 'aetv\assets');aetv/assets",
    "--add-data", "$HamlibDir;aetv/bin"
)
foreach ($Model in $RuntimeModels) {
    $Common += @("--add-data", "$($Model.FullName);models")
}

& $Python -m PyInstaller @Common --windowed --name AETV `
    --icon (Join-Path $RepoRoot "aetv\assets\aetv.ico") `
    (Join-Path $RepoRoot "aetv\gui\app.py")

& $Python -m PyInstaller @Common --console --name AETV-Benchmark `
    (Join-Path $RepoRoot "scripts\benchmark_inference.py")

& $Python -m PyInstaller --noconfirm --clean --onefile --console `
    --workpath $WorkPath --specpath $SpecPath --distpath $DistRoot `
    --hidden-import soundcard --name AETV-Audio `
    (Join-Path $RepoRoot "scripts\audio_helper.py")

$AppDir = Join-Path $DistRoot "AETV"
Copy-Item -LiteralPath (Join-Path $DistRoot "AETV-Benchmark\AETV-Benchmark.exe") -Destination $AppDir
$AudioHelperDir = Join-Path $AppDir "audio-helper"
New-Item -ItemType Directory -Force -Path $AudioHelperDir | Out-Null
Copy-Item -LiteralPath (Join-Path $DistRoot "AETV-Audio.exe") -Destination $AudioHelperDir
Copy-Item -LiteralPath (Join-Path $RepoRoot "README.md") -Destination $AppDir
Copy-Item -LiteralPath (Join-Path $RepoRoot "LICENSE") -Destination $AppDir
Copy-Item -LiteralPath (Join-Path $RepoRoot "NOTICE") -Destination $AppDir

$PreviousOffline = $env:AETV_OFFLINE
$PreviousQtPlatform = $env:QT_QPA_PLATFORM
$PreviousAppData = $env:APPDATA
$PreviousLocalAppData = $env:LOCALAPPDATA
try {
    $env:AETV_OFFLINE = "1"
    $env:QT_QPA_PLATFORM = "offscreen"
    $env:APPDATA = Join-Path $BuildRoot "smoke-config"
    $env:LOCALAPPDATA = Join-Path $BuildRoot "smoke-cache"
    Push-Location $AppDir
    try {
        $Smoke = & ".\AETV-Benchmark.exe" --mode V8 --device cpu --warmup 0 --repeats 1
        if ($LASTEXITCODE -ne 0) {
            throw "Packaged benchmark smoke test failed"
        }
        $GuiSmoke = Start-Process -FilePath ".\AETV.exe" -ArgumentList "--smoke-test" `
            -PassThru -WindowStyle Hidden
        if (-not $GuiSmoke.WaitForExit(180000)) {
            Stop-Process -Id $GuiSmoke.Id -Force -ErrorAction SilentlyContinue
            throw "Packaged GUI smoke test timed out"
        }
        if ($GuiSmoke.ExitCode -ne 0) {
            throw "Packaged GUI smoke test failed with exit code $($GuiSmoke.ExitCode)"
        }
    } finally {
        Pop-Location
    }
} finally {
    $env:AETV_OFFLINE = $PreviousOffline
    $env:QT_QPA_PLATFORM = $PreviousQtPlatform
    $env:APPDATA = $PreviousAppData
    $env:LOCALAPPDATA = $PreviousLocalAppData
}
$Smoke | Set-Content -LiteralPath (Join-Path $AppDir "build-smoke.json") -Encoding utf8
$PackagedModels = [System.IO.Path]::GetFullPath((Join-Path $AppDir "_internal\models"))
if (-not $PackagedModels.StartsWith([System.IO.Path]::GetFullPath($AppDir))) {
    throw "Refusing to remove models outside the packaged app"
}
if (Test-Path -LiteralPath $PackagedModels) {
    Remove-Item -LiteralPath $PackagedModels -Recurse -Force
}

if (-not $NoZip) {
    $Zip = Join-Path $DistRoot "AETV-windows-x64-$PackageRuntime.zip"
    if (Test-Path -LiteralPath $Zip) {
        Remove-Item -LiteralPath $Zip -Force
    }
    tar.exe -a -c -f $Zip -C $DistRoot AETV
}

Write-Host "Portable AETV $PackageRuntime build: $AppDir"
