# Published Flex-8k checkpoint

Place the inference-only V7 weights here as `v7-flex8k.pt`.

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
  --src D:\SSTVAE\runs\aetv-v7-flex8k-144p-stage2-nogan\checkpoint_step_010000.pt `
  --dst models\v7-flex8k.pt
```

`aetv send` and `aetv receive` look for `models/v7-flex8k.pt` by default.
