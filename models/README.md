# Flex-8k checkpoints

The application defaults to the severe-channel fine-tune:

| File | Purpose | SHA-256 |
|---|---|---|
| `v7-flex8k-severe.pt` | Default; strongest 0/-2 dB and MPP recovery | `e900bd7da2f080d23926a64f76d4b2624c413e08ddcdb91fde81c8acdd9a53b4` |
| `v7-flex8k-severe-balanced.pt` | Alternate with slightly more clean-channel fidelity | `18d610a35797f3bffb86f55bd8d9a79182d24b961640b19ffa27304763dcfa03` |
| `v7-flex8k.pt` | Original published stage-2 baseline | `afe476e5c5681210817a8e0598ec38ef40bdbd609485ad10d7a13ae9e6cd460b` |

Leave the checkpoint field empty in the GUI (or omit `--checkpoint`) to use
`models/v7-flex8k-severe.pt`. Set `AETV_CHECKPOINT` or pass `--checkpoint` to
select either alternate.

## Original published checkpoint

Place the inference-only V7 weights here as `v7-flex8k.pt`.

For a tagged release, download the checkpoint asset that accompanies it:

```powershell
Invoke-WebRequest `
  https://github.com/plaingca/AETV/releases/latest/download/v7-flex8k.pt `
  -OutFile models\v7-flex8k.pt
(Get-FileHash models\v7-flex8k.pt -Algorithm SHA256).Hash.ToLower()
```

The result must match the SHA-256 value below before the checkpoint is used.
The `latest` URL becomes available when the first GitHub release is published;
when building an unreleased checkout, export the checkpoint from the research
tree instead.

The file is not stored in git. It is the stage-2 no-GAN Flex-8k run that was
keyed on 40 m and used for the VVC comparison:

| Field | Value |
|---|---|
| Mode | V7 (256x144 @ 12 fps, band U, 24 kHz DAX) |
| Source run | `aetv-v7-flex8k-144p-stage2-nogan` |
| Training step | 10000 |
| Width / latent channels | 128 / 3 |
| Occupied audio | ~800-9200 Hz on a Flex 6600 DIGU slice |
| Inference file | 206 MiB, SHA-256 `afe476e5c5681210817a8e0598ec38ef40bdbd609485ad10d7a13ae9e6cd460b` |

Export from the research tree:

```powershell
.\.venv\Scripts\python.exe scripts\export_inference_checkpoint.py `
  --src C:\path\to\research\runs\aetv-v7-flex8k-144p-stage2-nogan\checkpoint_step_010000.pt `
  --dst models\v7-flex8k.pt
```

The original checkpoint remains useful as a reproducible release baseline,
but is no longer selected by default.
