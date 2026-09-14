# Portable SDR setup

Every Windows and Linux portable CPU/GPU package includes the RTL-SDR receive
utility, Pluto libiio runtime, HackRF native library, Python bindings, and linked native libraries.
Extract the **whole package** and run its AETV executable. No Python, libiio,
rtl-sdr/HackRF installation, or PATH edits are needed. The application loads its own
bundled libraries. These packages target 64-bit Windows and Linux.

## Windows: first USB connection

The application libraries are ready to use. Windows may still need to bind
the correct USB driver to each radio once; the offline setup tools are in the
package's `drivers` folder. Windows prompts for administrator access when
installing a driver.

* **RTL-SDR:** run `drivers/zadig-2.9.exe`, choose **Options → List All Devices**,
  select the RTL-SDR's **Bulk-In, Interface (Interface 0)**, and install
  **WinUSB**. Identify the RTL-SDR before replacing a driver; leave its other
  interfaces and unrelated USB devices alone. A radio already configured for
  another libusb-based SDR application may already be ready.
* **HackRF:** use the bundled Zadig tool to select your **HackRF One** USB device
  and install **WinUSB**. Close other SDR programs first. A PortaPack must be in
  its USB HackRF mode. No firmware flashing is performed by AETV.
* **PlutoSDR:** disconnect the Pluto, run
  `drivers/PlutoSDR-M2k-USB-Drivers.exe`, then reconnect it. This installs the
  vendor's USB network, serial, and WinUSB device support. Start with
  `ip:192.168.2.1` in AETV, or use a libiio `usb:` URI.
* For a Pluto reached over ordinary Ethernet/Wi-Fi via an `ip:` URI, only
  network connectivity to that address is needed on this computer.

Upstream instructions: [RTL-SDR quick start](https://www.rtl-sdr.com/rtl-sdr-quick-start-guide/),
[Zadig](https://zadig.akeo.ie/), and
[Analog Devices USB setup](https://wiki.analog.com/university/tools/pluto/drivers/windows).

## Linux: USB access

The portable libraries need permission to open the USB device. If the radio
already works as your normal user, no system change is needed. Otherwise, on
a distribution using udev and the `plugdev` group, install the supplied rules
from the extracted `AETV` folder:

```sh
sudo cp drivers/udev/*.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
```

Unplug/reconnect the radio and run AETV as your normal user. On systems without
an active desktop session, membership in `plugdev` may be needed. If an RTL-SDR
is claimed by the DVB television driver, stop the TV application and release
that device from the DVB driver before opening it in AETV.

AppImage users can extract the same files with `./AETV-*.AppImage --appimage-extract`;
the application folder is `squashfs-root/usr/lib/AETV`.

## Checking the package

Run `AETV-Benchmark.exe --sdr-smoke` on Windows or
`./AETV-Benchmark --sdr-smoke` on Linux. This starts the bundled RTL-SDR utility,
loads the Pluto libraries and USB/network backends, and exercises libiio's XML
parser, and resolves the HackRF API without opening a device. It also exercises
the 9.6 MS/s signed 8-bit HackRF transport with nonuniform USB-sized chunks
and decodes an offline AC16 late-entry fixture at ±12.5 Hz residual
offset and checks the recovered payload against the transmitted fixture.
It does not open hardware or transmit. Build-time reports are included
as `sdr-smoke.json` and `sdr-gui-smoke.json`; physical USB access depends on the
driver setup above. Source installations still need native SDR dependencies.

## Signal visible, but no video

Select AC16 at both ends. For 439 MHz reception, the RTL driver deliberately
tunes to 439.100 MHz; the application removes that 100 kHz offset. Leave automatic
frequency correction enabled to compensate for the RTL oscillator.

The receive status distinguishes synchronization search, beacon search, and
decoding the first synchronized video GOP. Joining a continuous transmission
after its opening header requires a 12-second beacon observation. Keep the
receiver running during the transmission.

With **Settings → Folders → Save TX/RX waveforms, Kiwi IQ, and modem debug logs**
enabled, direct SDR reception saves `*.audio.wav` at the modem sample rate and
`*.modem.jsonl` under the received-video folder's `debug` directory. The WAV is
the recovered modem signal, not microphone audio or source video. Preserve both
files from the same attempt when reporting a failure. The log includes tuning
messages, acquisition timing, and whether a GOP reached the video decoder.

For AC16, the Windows GPU package checks DirectML color accuracy against CPU
when loading the model. It retries compatible GPU settings if needed, then
selects CPU if the result still differs too much. The model label shows the
device actually in use; the log and model tooltip explain any fallback. For
a diagnostic report, run `AETV-Benchmark.exe --mode AC16 --device auto --json ac16.json`.

## HackRF (experimental)

Choose **Transmit via → HackRF (experimental)** for transmit, or **Receive →
HackRF (experimental)** for receive. The RF controls provide a common frequency
center, optional hexadecimal device serial (empty selects the first device), TX
VGA gain (0–47 dB), RX LNA gain (0–40 dB in 8 dB steps), and RX VGA gain (0–62 dB
in 2 dB steps). Start with TX gain 0, RX LNA 16, and RX VGA 16; reduce receive
gain if the waterfall shows overload. Apply changes before transmitting.

AETV uses 9.6 MS/s at the hardware, the 1.75 MHz analog filter, and digital
filtering/decimation to isolate the modem. The RF amplifier and antenna bias
power stay off. The selected frequency is the signal center; the hardware LO
is 100 kHz below it for TX and 100 kHz above it for RX. The waterfall shows
captured complex IQ. AC16 automatic frequency correction and debug waveform
recording work as with the other direct SDR receivers.

HackRF is half duplex. When the same HackRF path is selected for TX and RX,
AETV stops receiving, transmits, then resumes reception. A separate RTL-SDR or
Pluto can monitor a HackRF transmission. TX uses bounded queues, detects missing
IQ rather than repeating stale buffers, and waits for libhackrf's USB flush
before stopping and closing the device. An underrun is reported as a failed
transmission; use a prepared clip or a faster inference device if live encoding
cannot keep up.

This integration has no physical HackRF validation yet. Software tests cover
signed IQ, modem recovery, callbacks, cancellation, driver errors, and half-duplex
coordination. Portable smoke tests load the actual native libraries. These checks
do not establish RF output level, oscillator accuracy, USB throughput on your
host, receiver sensitivity, or HackRF real-time video performance.

Source installations require libhackrf with the TX-flush API (use 2026.01.3).
Portable packages include it. Hardware and gain references:
[HackRF sampling guidance](https://hackrf.readthedocs.io/en/latest/sampling_rate.html),
[gain controls](https://hackrf.readthedocs.io/en/latest/setting_gain.html), and
[libhackrf streaming API](https://github.com/greatscottgadgets/hackrf/blob/v2026.01.3/host/libhackrf/src/hackrf.h).
