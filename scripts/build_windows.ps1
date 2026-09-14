param(
    [ValidateSet("cpu", "gpu", "cuda")]
    [string]$Runtime = "cpu",
    [switch]$TestDirectML,
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$PackageRuntime = if ($Runtime -eq "cuda") { "gpu" } else { $Runtime }
if ($TestDirectML -and $PackageRuntime -ne "gpu") {
    throw "-TestDirectML requires -Runtime gpu"
}
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
$SdrDir = Join-Path $BuildRoot "sdr"
if (Test-Path -LiteralPath $SdrDir) { Remove-Item -LiteralPath $SdrDir -Recurse -Force }
& $Python (Join-Path $RepoRoot "scripts\fetch_sdr_windows.py") --output $SdrDir
if ($LASTEXITCODE -ne 0) { throw "SDR runtime download failed" }
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
    "--paths", $RepoRoot,
    "--runtime-hook", (Join-Path $RepoRoot "scripts\pyi_rth_sdr.py"),
    "--exclude-module", "torch",
    "--exclude-module", "torchvision",
    "--exclude-module", "aetv.models",
    "--exclude-module", "aetv.channel",
    "--exclude-module", "aetv.data",
    "--exclude-module", "aetv.video_backbone",
    "--exclude-module", "imageio_ffmpeg",
    "--add-data", "$(Join-Path $RepoRoot 'aetv\assets');aetv/assets",
    "--add-data", "$HamlibDir;aetv/bin"
)
foreach ($Backend in @('rtlsdr', 'pluto')) {
    foreach ($Binary in (Get-ChildItem -LiteralPath (Join-Path $SdrDir "runtime\$Backend") -File)) {
        $Common += @("--add-binary", "$($Binary.FullName);aetv/bin/$Backend")
    }
}
foreach ($Model in $RuntimeModels) {
    $Common += @("--add-data", "$($Model.FullName);models")
}

& $Python -m PyInstaller @Common --windowed --name AETV `
    --icon (Join-Path $RepoRoot "aetv\assets\aetv.ico") `
    (Join-Path $RepoRoot "aetv\gui\app.py")
if ($LASTEXITCODE -ne 0) { throw "GUI packaging failed" }

& $Python -m PyInstaller @Common --console --name AETV-Benchmark `
    (Join-Path $RepoRoot "scripts\benchmark_inference.py")
if ($LASTEXITCODE -ne 0) { throw "Benchmark packaging failed" }

& $Python -m PyInstaller --noconfirm --clean --onefile --console `
    --workpath $WorkPath --specpath $SpecPath --distpath $DistRoot `
    --hidden-import soundcard --name AETV-Audio `
    (Join-Path $RepoRoot "scripts\audio_helper.py")

$AppDir = Join-Path $DistRoot "AETV"
# Qt uses the Windows ICU compatibility layer. PyInstaller can discover and
# bundle an unrelated third-party icuuc.dll from the build host, which then
# shadows the compatible Windows DLL and prevents PySide6.QtCore from loading.
$ForeignIcu = Join-Path $AppDir "_internal\icuuc.dll"
if (Test-Path -LiteralPath $ForeignIcu -PathType Leaf) {
    Remove-Item -LiteralPath $ForeignIcu -Force
}
$FfmpegSource = (& $Python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())").Trim()
if (-not (Test-Path -LiteralPath $FfmpegSource -PathType Leaf)) {
    throw "imageio-ffmpeg did not provide a Windows executable: $FfmpegSource"
}
Copy-Item -LiteralPath $FfmpegSource -Destination (Join-Path $AppDir "ffmpeg.exe")
Copy-Item -LiteralPath (Join-Path $DistRoot "AETV-Benchmark\AETV-Benchmark.exe") -Destination $AppDir
$AudioHelperDir = Join-Path $AppDir "audio-helper"
New-Item -ItemType Directory -Force -Path $AudioHelperDir | Out-Null
Copy-Item -LiteralPath (Join-Path $DistRoot "AETV-Audio.exe") -Destination $AudioHelperDir
Copy-Item -LiteralPath (Join-Path $RepoRoot "README.md") -Destination $AppDir
Copy-Item -LiteralPath (Join-Path $RepoRoot "LICENSE") -Destination $AppDir
Copy-Item -LiteralPath (Join-Path $RepoRoot "NOTICE") -Destination $AppDir
Copy-Item -LiteralPath (Join-Path $RepoRoot "FFMPEG-NOTICE.txt") -Destination $AppDir
New-Item -ItemType Directory -Force -Path (Join-Path $AppDir "docs") | Out-Null
Copy-Item -LiteralPath (Join-Path $RepoRoot "docs\ac16-gui-and-sdr.md") -Destination (Join-Path $AppDir "docs")
Copy-Item -LiteralPath (Join-Path $RepoRoot "docs\sdr-portable-setup.md") -Destination (Join-Path $AppDir "docs")
Copy-Item -LiteralPath (Join-Path $RepoRoot "SDR-NOTICE.txt") -Destination $AppDir
Copy-Item -LiteralPath (Join-Path $SdrDir 'drivers') -Destination $AppDir -Recurse
foreach ($Folder in @('licenses', 'sources')) {
    Copy-Item -LiteralPath (Join-Path $SdrDir $Folder) -Destination (Join-Path $AppDir "drivers\$Folder") -Recurse
}
Copy-Item -LiteralPath (Join-Path $SdrDir 'dependencies.json') -Destination (Join-Path $AppDir 'drivers')

# Test the actual frozen GUI and console entry points with no PATH-installed
# rtl_sdr or libiio. This requires no SDR and performs no RF operations.
$PreviousPath = $env:PATH
try {
    $env:PATH = ''
    $SdrReport = Join-Path $AppDir 'sdr-gui-smoke.json'
    $SdrProcess = Start-Process -FilePath (Join-Path $AppDir 'AETV.exe') `
        -ArgumentList @('--sdr-smoke', "`"$SdrReport`"") -PassThru -WindowStyle Hidden
    if (-not $SdrProcess.WaitForExit(60000)) {
        Stop-Process -Id $SdrProcess.Id -Force -ErrorAction SilentlyContinue
        throw 'Packaged GUI SDR runtime check timed out'
    }
    if ($SdrProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $SdrReport)) {
        throw 'Packaged GUI SDR runtime check failed'
    }
    & (Join-Path $AppDir 'AETV-Benchmark.exe') --sdr-smoke --json (Join-Path $AppDir 'sdr-smoke.json')
    if ($LASTEXITCODE -ne 0) { throw 'Packaged SDR runtime check failed' }
} finally {
    $env:PATH = $PreviousPath
}

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
        $VideoSmoke = Join-Path $BuildRoot "saved-video-smoke.mp4"
        if (Test-Path -LiteralPath $VideoSmoke) {
            Remove-Item -LiteralPath $VideoSmoke -Force
        }
        & ".\AETV-Benchmark.exe" --video-save-smoke $VideoSmoke | Out-Null
        if ($LASTEXITCODE -ne 0 -or
            -not (Test-Path -LiteralPath $VideoSmoke -PathType Leaf) -or
            (Get-Item -LiteralPath $VideoSmoke).Length -lt 32) {
            throw "Packaged Save video smoke test did not create an MP4"
        }
        $BenchmarkDevice = if ($TestDirectML) { "dml" } else { "cpu" }
        $Smoke = & ".\AETV-Benchmark.exe" --mode V8 --device $BenchmarkDevice --warmup 0 --repeats 1
        if ($LASTEXITCODE -ne 0) {
            throw "Packaged benchmark smoke test failed"
        }
        $SmokeResult = ($Smoke -join "`n") | ConvertFrom-Json
        if ($TestDirectML -and $SmokeResult.device -ne "DirectML") {
            throw "Packaged GPU benchmark did not select DirectML"
        }
        $GuiSmoke = Start-Process -FilePath ".\AETV.exe" `
            -ArgumentList @("--smoke-test", "--video-smoke-output", $VideoSmoke) `
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
