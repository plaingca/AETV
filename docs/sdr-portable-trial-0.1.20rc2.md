# Portable SDR trial 0.1.20rc2

All Windows/Linux CPU and GPU portable variants include RTL-SDR, Pluto libiio and their linked libraries. Windows includes the offline Zadig and Analog Devices USB setup tools, licenses, source archives and pinned dependency hashes. Linux includes udev permission rules.

Binaries come from `35c85db1e31a2300d3d6e84ef86480e6efb877a2`. [All four builds](https://github.com/plaingca/AETV/actions/runs/34791671892) and [Windows/Linux CI](https://github.com/plaingca/AETV/actions/runs/34791673942) passed. Local tests: **348 passed, 2 optional-runtime skips**.

Both frozen entry points in every build were tested with an empty PATH: RTL-SDR help execution, libiio USB/network backend loading and XML parsing passed. Windows ZIPs contain the exact pinned upstream native bytes, required VC runtime, setup tools and source/license files. Sixteen frozen Windows code checks matched the source. Downloaded Linux AppImages passed fresh SDR probes; Linux archives contain their native libraries and build-time probe reports.

Hardware checks used the rebuilt Linux CPU archive at 439 MHz, Pluto TX gain -10 dB, working RTL serial 1001, and the same training-disjoint validation RGB fixture as rc1. The frozen app ran with an empty PATH and resolved its bundled native drivers.

| Trial | RF GOPs recovered | Frames displayed | Minimum latent cosine | Full playback check |
|---|---:|---:|---:|---|
| linux-cpu-rtlsdr | 10/10 | 100/100 | 0.9543 | Pass |
| linux-cpu-pluto | 10/10 | 89/100 | 0.9576 | Fail |
| linux-cpu-pluto-repeat | 10/10 | 100/100 | 0.9574 | Pass |

The successful Pluto repeat used `OPENBLAS_NUM_THREADS=2` and `OMP_NUM_THREADS=8`. The first Pluto attempt is retained: all GOPs were reconstructed in order but playback missed frames. The repeat does not establish the cause or a universal real-time guarantee. All three attempts passed payload order/fidelity; after testing, independent readback confirmed Pluto TX powered down at -89.75 dB and zero RTL capture processes.

Physical Windows USB/RF operation and installer execution were not tested here. A first-time USB driver binding still requires Windows administrator access; the utilities are included. See [portable setup](sdr-portable-setup.md).

[Stable release downloads](https://github.com/plaingca/AETV/releases/tag/v0.1.20). Full machine-readable results are in [the evidence file](evidence/sdr-portable-0.1.20rc2.json).

## Artifact SHA-256

- `AETV-linux-x64-cpu.AppImage` (246553080 bytes): `bc7418354753cc0aa5dac4ae8044df6ec489b597e7a3aec2fecd2b2c2bb10b3d`
- `AETV-linux-x64-cpu.tar.gz` (268891342 bytes): `2a5f3f9809882caff3b73a8ecfe52213f197ed2a7a04d52dbe3c2aad5f2fde22`
- `AETV-linux-x64-gpu.AppImage` (1784498680 bytes): `20c6bf524213d1b31e4c238c6d5b0740f52bbf417b43ae1d4a4a5be3281a7984`
- `AETV-linux-x64-gpu.tar.gz` (1908312207 bytes): `87a4dcb9200c36ce3eaef83a28ac33c1bf5a518f8e281bfdbfb24adb580e0b67`
- `AETV-windows-x64-cpu.zip` (276227642 bytes): `4c72c36caf473eb897960446c2be65e0a84a8de7c8f67a4677c7a068e75ccbf7`
- `AETV-windows-x64-gpu.zip` (287876064 bytes): `c00d449f9a3b43f541f70546be819e922c7ea4e4be41b6b6968451b7fa8ab8f0`
