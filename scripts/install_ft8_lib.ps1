param(
    [string]$InstallRoot = "$env:LOCALAPPDATA\AETV\ft8_lib"
)

$ErrorActionPreference = "Stop"
$commit = "9fec6ca39886edbf96f4f5e71edc76da5074e871"
$install = [System.IO.Path]::GetFullPath($InstallRoot)
$allowed = [System.IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "AETV"))
if (-not $install.StartsWith($allowed, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "InstallRoot must be under $allowed"
}
$source = Join-Path $install "source"
$build = Join-Path $install "build"
$exe = Join-Path $install "aetv_gen_ft8.exe"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "Visual Studio Build Tools were not found"
}
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) {
    throw "Visual Studio C++ tools were not found"
}
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
New-Item -ItemType Directory -Path $install -Force | Out-Null
New-Item -ItemType Directory -Path $build -Force | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $source ".git"))) {
    git clone https://github.com/kgoba/ft8_lib.git $source
}
git -C $source fetch --depth 1 origin $commit
git -C $source checkout --detach $commit
$wrapper = Join-Path $PSScriptRoot "ft8_gen_main.c"
$sources = @(
    $wrapper,
    (Join-Path $source "common\wave.c"),
    (Join-Path $source "ft8\encode.c"),
    (Join-Path $source "ft8\crc.c"),
    (Join-Path $source "ft8\constants.c"),
    (Join-Path $source "ft8\message.c"),
    (Join-Path $source "ft8\text.c")
)
$quotedSources = ($sources | ForEach-Object { '"' + $_ + '"' }) -join ' '
$compat = Join-Path $PSScriptRoot "ft8_msvc_compat.h"
$command = 'call "' + $vcvars + '" >nul && cl /nologo /O2 /std:c11 /D_CRT_SECURE_NO_WARNINGS /FI"' + $compat + '" /I"' + $source + '" ' + $quotedSources + ' /Fe:"' + $exe + '"'
Push-Location $build
try {
    cmd.exe /d /s /c $command
}
finally {
    Pop-Location
}
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $exe)) {
    throw "ft8_lib build failed"
}
Write-Output "Installed $exe from ft8_lib $commit"
