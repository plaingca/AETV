# Launch the AETV ham-station GUI from a checkout.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is not on PATH. Install it from https://docs.astral.sh/uv/ and re-run."
}

if (-not (Test-Path ".\.venv")) {
    Write-Host "Creating the station environment (PyTorch, Qt, sounddevice)…"
    uv sync --extra gui
}

if (-not (Test-Path ".\models\v7-flex8k.pt")) {
    Write-Host "Missing models\v7-flex8k.pt — the GUI will start but Send/Receive stay disabled until the checkpoint is copied. See models\README.md."
}

uv run aetv gui
