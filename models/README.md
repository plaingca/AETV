# AETV checkpoints

Release weights are hosted at
[AETV/AETV](https://huggingface.co/AETV/AETV). The app
downloads the selected mode's default into its per-user cache and verifies the
published SHA-256 automatically. Set `AETV_MODEL_DIR` to choose another cache
directory or `AETV_OFFLINE=1` to disable network access.

The application defaults to the receiver-adapted OTA fine-tune:

| File | Purpose | SHA-256 |
|---|---|---|
| `v8-flex8k-ota-rxfix.pt` | Default; adapted to noise-aware MMSE confidence and verified on a 71-GOP OTA sweep | `294987591b8ece1cb6fd6ad10349a160192e4e6fefc26d47bbbefd9cce9a778f` |
| `v8-flex8k-ota-perceptual.pt` | Previous OTA-perceptual default using the legacy equalizer contract | `425f112924693170c61cebb6ab5865bd526714a4afae9aec88a37709441b5d47` |
| `v7-flex8k-severe.pt` | Alternate; strongest 0/-2 dB and MPP recovery | `78f990e34625cfbc6a5d80b673dd8fb0ffea6a565cf9a80796e110c40e0cdf14` |
| `v7-flex8k-severe-balanced.pt` | Alternate with slightly more clean-channel fidelity | `59c0b8338b4a59e10b2ded81e8cecec1b04871d620b66af93be400008089c666` |
| `v7-flex8k.pt` | Original published stage-2 baseline | `afe476e5c5681210817a8e0598ec38ef40bdbd609485ad10d7a13ae9e6cd460b` |

Leave the checkpoint field empty in the GUI (or omit `--checkpoint`) to use the
selected mode's downloaded default. Set `AETV_CHECKPOINT` or pass
`--checkpoint` to select an alternate local file.

See `docs/v8-ota-rxfix.md` for the receiver correction and OTA replay, and
`docs/v8-ota-perceptual.md` for the original 32-clip grid and VVC reference.

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

## HF-3k V8 checkpoints

V8 uses the standard-channel W waveform at 192x108 and 6 fps. These files are
not stored in git; copy the exported inference checkpoints into `models/`:

| File | Purpose | SHA-256 |
|---|---|---|
| `v8-hf3k-face-gan.pt` | V8 default; OpenVid-1M stage-2 model with localized face perceptual/GAN fine-tuning | `f218376af9f9916050c9e345353da0c0970c392f58755efaa81d01e7ded8fc40` |
| `v8-hf3k-perceptual.pt` | Former V8 default; motion-aware perceptual fine-tune | `35fb34a981976070ca7cb6b54e157b3e8ec9f1b44f12f9fc55d26288bc83e707` |
| `v8-hf3k-robust.pt` | Alternate for 0 dB and severe MPP fading | `4620845d282064b2007d1cd620892f96ae3fc8dfc62caf1bb3f244897ebb7cbd` |

See [`docs/v8-hf3k.md`](../docs/v8-hf3k.md) for the waveform contract,
fine-tuning recipe, paired 32-clip results, and operating commands.
