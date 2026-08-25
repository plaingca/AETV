param(
    [Parameter(Mandatory = $true)]
    [string]$Output
)

$ErrorActionPreference = 'Stop'
$Version = '4.7.2'
$ExpectedSha256 = '8553bc6c5c6032e8debf99c017e98f58fed7e07e7c25d04815dc3e8bbe3304c7'
$Url = "https://github.com/Hamlib/Hamlib/releases/download/$Version/hamlib-w64-$Version.zip"
$Output = [System.IO.Path]::GetFullPath($Output)
$Scratch = Join-Path ([System.IO.Path]::GetTempPath()) ('aetv-hamlib-' + [guid]::NewGuid().ToString('N'))
$Archive = Join-Path $Scratch 'hamlib.zip'
$Expanded = Join-Path $Scratch 'expanded'

try {
    New-Item -ItemType Directory -Force -Path $Scratch,$Expanded,$Output | Out-Null
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Archive
    $Actual = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $ExpectedSha256) {
        throw "Hamlib archive checksum mismatch: $Actual"
    }
    Expand-Archive -LiteralPath $Archive -DestinationPath $Expanded
    $Root = Join-Path $Expanded "hamlib-w64-$Version"
    foreach ($Name in @(
        'rigctl.exe',
        'libhamlib-4.dll',
        'libusb-1.0.dll',
        'libgcc_s_seh-1.dll',
        'libwinpthread-1.dll'
    )) {
        Copy-Item -LiteralPath (Join-Path $Root "bin\$Name") -Destination $Output
    }
    Copy-Item -LiteralPath (Join-Path $Root 'COPYING.LIB.txt') -Destination $Output
    Copy-Item -LiteralPath (Join-Path $Root 'COPYING.txt') -Destination $Output
    Copy-Item -LiteralPath (Join-Path $Root 'LICENSE.txt') -Destination $Output
    Copy-Item -LiteralPath (Join-Path $Root 'README.w64-bin.txt') -Destination $Output
    @"
Hamlib $Version source and corresponding source archive:
https://github.com/Hamlib/Hamlib/releases/tag/$Version
https://github.com/Hamlib/Hamlib/releases/download/$Version/hamlib-$Version.tar.gz

The bundled DLLs and rigctl utility are unmodified official Hamlib release
binaries. AETV dynamically loads the LGPL library and invokes the GPL rigctl
utility as a separate process for model discovery. You may replace them with
compatible builds.
"@ | Set-Content -LiteralPath (Join-Path $Output 'HAMLIB-SOURCE.txt') -Encoding utf8
}
finally {
    if (Test-Path -LiteralPath $Scratch) {
        [System.IO.Directory]::Delete($Scratch, $true)
    }
}
