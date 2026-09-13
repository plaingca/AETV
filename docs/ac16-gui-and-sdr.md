# AC16 and direct SDR operation

AC16 is available in the production GUI's mode selector and Model Manager.
Its selected v4 checkpoint and portable ONNX graphs are published in
[AETV/AETV](https://huggingface.co/AETV/AETV/blob/6f3a4cd4df1e8b261141da75988f3df2750666db/AC16.md).
The application pins that immutable revision and checks each download's size
and SHA-256 before installation. Existing V8 and V7 defaults remain available.

AC16 carries 256×144 RGB at 10 fps, with one I frame and nine P frames per GOP.
The separate 48 ksample/s modem uses 19,200 real coordinates and a guarded
16 kHz waveform. This waveform requires an appropriately wide radio channel;
it is not an ordinary narrow SSB audio mode. Audio/video composite mode is
currently restricted to V8.

## Operating the radio

1. Open **File → Model Manager**, select AC16, and download its runtime.
   Select AC16 in the Transmit mode picker. A CPU package uses ONNX Runtime;
   the Linux GPU package can use CUDA and the Windows GPU package DirectML.
2. Set **Transmit via → PlutoSDR**, enter the Pluto address, and set the large
   **RF center** dial. The default address is `ip:192.168.2.1` and frequency
   439.000000 MHz. Enter the correct callsign in station settings.
3. Choose **PlutoSDR** or **RTL-SDR** in the Receive source picker. For RTL-SDR,
   enter the device's exact serial number. The test host uses working serial
   1001; serial 1002 has a known hardware fault and was excluded from validation.
4. Adjust the separate **Pluto TX gain** and **RX gain** sliders, then click
   **Apply RF settings**. Applying changes restarts an active receiver. RF
   settings are disabled while transmitting; Stop/Cancel remains available.
5. Start receiving, select a webcam, screen, video file or prepared clip, and
   press Send. Receive stays active during Pluto transmission for local
   monitoring with either SDR receiver. Stop ends receive capture; Cancel and
   normal completion both power down Pluto TX and release its finite buffers.

The waterfall displays the actual complex receive IQ, with an RF frequency
axis and a 96 kHz view around the requested center. It does not display a
synthesized spectrum or source waveform. The other radio/audio receive paths
retain their existing audio waterfall.

The RX frequency correction field accounts for tuner/oscillator error. Its
units are Hz relative to the nominal received signal offset. AC16's optional
automatic correction estimates the occupied spectrum from received IQ only;
initial acquisition waits for a stable strong-signal spectrum and can add
startup delay. Disable automatic correction and set a manual correction for
weak signals. The dial denotes RF waveform center, not a soundcard audio
reference. Hardware LOs are offset by 100 kHz to keep the desired waveform
away from their DC spur.

Pluto TX starts at −30 dB hardware gain by default, with the slider covering
−89.75 to 0 dB. Pluto RX uses manual gain; RTL gain is passed to its tuner,
which selects its supported hardware setting. The amplitude control in the
Transmit pane and the hardware gain slider are separate adjustments.

Linux portable packages include `rtl_sdr`, libiio, and their linked runtime
libraries. USB devices still require normal OS device access permissions.
For a source installation, install `aetv[gui]`, libiio and the `rtl-sdr` tools.
Windows SDR operation requires the vendor libiio runtime and `rtl_sdr.exe` on
PATH, plus the device's USB driver; physical Windows SDR operation has not been
validated on this Linux host. Pluto network URIs can be used without a USB
transport to the host.

## Runtime and validation

The native selected checkpoint is `ac16-best-inference.pt`, SHA-256
`ff451787feb2708310eebac4a47151cb0fef654c071af463022d8be2f1c618cb`.
ONNX export preserves its fixed input shapes and receiver confidence inputs.
Four held-out GOPs under clean, noisy, motion-outage and full-outage conditions
matched FP32 native inference within one 8-bit RGB level (mean error below
0.01 levels). The portable app does not require PyTorch.

AC16 defaults to eight inference threads on CPU. `AETV_CPU_THREADS` overrides
this count. CUDA packages include the CUDA runtime/cuDNN dependencies; a
compatible NVIDIA driver is supplied by the host. Actual provider selection
is shown in the status bar. An explicitly selected CUDA provider that fails
to initialize raises an error rather than being reported as working CUDA.

The continuous TX path uses bounded queues and converts IQ on a producer
thread, independently of the hardware DMA consumer. RX captures IQ separately
from its conversion and demodulation. AC16 GUI playback uses two GOPs of
startup buffering and a four-GOP queue cap to accommodate acquisition bursts;
there is no extra cross-GOP image blending. Throughput and startup latency
are separate properties.

The package includes an explicit bounded hardware-validation entry point:

```bash
AETV --ota-validation trial.json
```

The configuration must contain `transmit: true`. It identifies a uint8 NPY
array of AC16 RGB frames, a new output directory, 1–60 GOPs, the receiver,
frequency, gains, and compute device. This option sends actual RF. It feeds
paced RGB frames at the camera boundary, then exercises the ordinary GUI
Send/Receive engines, codec, hardware transports and displayed-frame callback.
It saves timing, source/received latent correspondence, reconstructed frames,
GUI screenshots and a pass/fail report. Camera-driver acquisition itself is
outside that fixture's scope. No source timing or latents are provided to the
receiver; correspondence is checked after reception. Full-GOP recovery alone
is insufficient: displayed frame counts and latent fidelity are also checked.

Trial release evidence and artifact checksums are recorded separately in the
release validation report. No claim is made that every SNR, CPU, GPU provider,
USB topology, or OS has the same throughput.
