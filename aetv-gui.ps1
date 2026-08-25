# Launch the AETV ham-station GUI from a checkout.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is not on PATH. Install it from https://docs.astral.sh/uv/ and re-run."
}

if (-not (Test-Path ".\.venv")) {
    Write-Host "Creating the station environment (Qt, ONNX Runtime, sounddevice)..."
    uv sync --extra gui
}

Write-Host "Release models and their per-user cache are shown under File > Model Manager."

uv run aetv gui
