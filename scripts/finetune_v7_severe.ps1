param(
    [string]$BaseCheckpoint = "models\v7-flex8k.pt",
    [string]$CacheDir = "data\openvid_aetv_cache",
    [string]$OutDir = "runs\v7-severe-calibrated-ft",
    [switch]$InstallModels
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    uv run python scripts/train.py `
        --mode V7 `
        --stage 2 `
        --out $OutDir `
        --cache-dir $CacheDir `
        --steps 500 `
        --eval-interval 250 `
        --tb-interval 25 `
        --checkpoint-interval 250 `
        --clean-warmup 0 `
        --channel-ramp 25 `
        --batch 1 `
        --accum 4 `
        --threads 4 `
        --lr 1e-5 `
        --adv-weight 0 `
        --fm-weight 0 `
        --lecam-weight 0 `
        --model-width 128 `
        --latent-channels 3 `
        --snr-min -2 `
        --snr-max 6 `
        --p-fading 0.4 `
        --init-checkpoint $BaseCheckpoint `
        --reset-steps `
        --amp

    if ($InstallModels) {
        $modelDir = Join-Path $repoRoot "models"
        New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
        Copy-Item -LiteralPath (Join-Path $OutDir "checkpoint_step_000250.pt") `
            -Destination (Join-Path $modelDir "v7-flex8k-severe-balanced.pt") -Force
        Copy-Item -LiteralPath (Join-Path $OutDir "checkpoint_step_000500.pt") `
            -Destination (Join-Path $modelDir "v7-flex8k-severe.pt") -Force
    }
}
finally {
    Pop-Location
}
