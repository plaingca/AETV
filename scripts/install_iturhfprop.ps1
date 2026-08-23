param(
    [ValidateRange(1, 12)]
    [int]$Month = (Get-Date).Month,
    [switch]$AllMonths
)

$ErrorActionPreference = 'Stop'
$commit = '82017594a1c6cacfaa7e86954c4ae7b3a5825a3d'
$archiveUrl = "https://github.com/ITU-R-Study-Group-3/ITU-R-HF/archive/$commit.zip"
$target = Join-Path $env:LOCALAPPDATA 'AETV\iturhfprop'
$scratch = Join-Path ([System.IO.Path]::GetTempPath()) ('aetv-iturhfprop-' + [guid]::NewGuid().ToString('N'))
$archive = Join-Path $scratch 'source.zip'
$source = Join-Path $scratch "ITU-R-HF-$commit"

try {
    New-Item -ItemType Directory -Path $scratch,$target,(Join-Path $target 'Data') -Force | Out-Null
    Invoke-WebRequest -Uri $archiveUrl -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath $scratch

    $vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) {
        throw 'Visual Studio 2022 C++ build tools are required to build ITURHFProp.'
    }
    $visualStudio = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath
    $msbuild = Join-Path $visualStudio 'MSBuild\Current\Bin\MSBuild.exe'
    & $msbuild (Join-Path $source 'ITURHFProp\Win32\Combined\Combined.sln') /m /p:Configuration=Release /p:Platform=x64 /verbosity:minimal
    if ($LASTEXITCODE -ne 0) {
        throw "ITURHFProp build failed with exit code $LASTEXITCODE"
    }

    $built = Join-Path $source 'ITURHFProp\Win32\Combined\x64\Release'
    Copy-Item -LiteralPath (Join-Path $built 'ITURHFProp_x64.exe') -Destination (Join-Path $target 'ITURHFProp.exe')
    Copy-Item -LiteralPath (Join-Path $built 'P533_x64.dll') -Destination (Join-Path $target 'P533.dll')
    Copy-Item -LiteralPath (Join-Path $built 'P372_x64.dll') -Destination (Join-Path $target 'P372.dll')
    Copy-Item -LiteralPath (Join-Path $source 'ITURHFProp\Data\P1239-3 Decile Factors.txt') -Destination (Join-Path $target 'Data')

    $months = if ($AllMonths) { 1..12 } else { @($Month) }
    foreach ($item in $months) {
        $suffix = $item.ToString('00')
        Copy-Item -LiteralPath (Join-Path $source "ITURHFProp\Data\ionos$suffix.bin") -Destination (Join-Path $target 'Data')
        Copy-Item -LiteralPath (Join-Path $source "ITURHFProp\Data\COEFF${suffix}W.txt") -Destination (Join-Path $target 'Data')
    }
    Write-Host "Installed ITURHFProp for month(s) $months in $target"
}
finally {
    if (Test-Path -LiteralPath $scratch) {
        Get-ChildItem -LiteralPath $scratch -Recurse -Force | ForEach-Object {
            if ($_ -is [System.IO.FileInfo]) { $_.IsReadOnly = $false }
        }
        [System.IO.Directory]::Delete($scratch, $true)
    }
}
