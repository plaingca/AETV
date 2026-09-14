# HackRF integration trial

HackRF is an experimental direct SDR transmitter and receiver in the production
GUI. No physical HackRF was available for this integration; this report separates
software and package verification from RF validation.

## Behavior

- Native libhackrf callbacks exchange signed 8-bit interleaved I/Q at 9.6 MS/s.
  RX retains raw IQ for the waterfall, then digitally filters and decimates to
  960 kS/s before the existing frequency correction and modem conversion.
- The signal center uses the existing 100 kHz TX/RX LO offsets. The analog
  baseband filter is 1.75 MHz. RF amplifier and antenna bias power remain off.
- A shared serial selector, TX VGA slider, and separate RX LNA/VGA sliders use
  hardware-supported steps. Empty serial selects the first available device.
- The GUI waits for HackRF RX shutdown before HackRF TX, then resumes reception.
  Pluto/RTL monitoring may remain active while a separate HackRF transmits.
- A bounded TX queue is prefilled before USB streaming starts. Underruns fail
  explicitly. A partial final transfer is submitted before waiting for the native
  flush callback; stop, gain reset, close and library exit run outside callbacks.
- Existing debug waveform and acquisition logs include HackRF reception.

## Dependencies

Linux builds compile official libhackrf 2026.01.3 (library 0.9.2) from a
SHA-256-pinned source ZIP. Windows packages use pinned MSYS2 builds of the same
release plus libusb 1.0.30 and winpthreads. The Windows Pluto and HackRF copies of
libusb are identical: HackRF requires new raw-I/O exports, so an older libusb
with the same DLL basename must not enter the process. Sources, license texts,
USB permission rules and the existing Windows Zadig installer are bundled.

## Software verification

The signed-IQ fixture sends three random AC16 payloads through the production
TX conversion, TX quantization, simulated LO difference, RX quantization,
nonuniform USB-sized chunks, decimation and streaming modem. It recovers all
three GOPs in order; the initial local run's minimum latent cosine was 0.95651.
The frozen GUI and benchmark smoke checks both repeat this fixture and resolve
the actual bundled native APIs without opening USB devices.

Tests also exercise signed-byte order, sample boundaries, partial final TX,
flush completion, underrun, RX overflow, startup/configuration failure,
cancellation, cleanup after stop failure, persisted settings and legal gain
steps, station routing, and GUI half-duplex coordination. Artifact checks and
CI results are recorded in the trial release's validation report.

## Remaining hardware checks

A HackRF owner must verify USB startup, sustained RX/TX, tuning, oscillator
correction, RF level and spectral quality, cancellation, and return to reception.
The software fixture does not model the analog radio, USB scheduling, sample
clock drift or firmware behavior. No HackRF sensitivity, RF compliance, or
physical real-time performance claim is made. The model and AC16 wire contract
are unchanged.

[Setup](sdr-portable-setup.md#hackrf-experimental) ·
[Upstream sampling guidance](https://hackrf.readthedocs.io/en/latest/sampling_rate.html) ·
[Gain controls](https://hackrf.readthedocs.io/en/latest/setting_gain.html) ·
[Native streaming API](https://github.com/greatscottgadgets/hackrf/blob/v2026.01.3/host/libhackrf/src/hackrf.h)
