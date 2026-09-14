# Portable SDR setup

Every Windows and Linux portable CPU/GPU package includes the RTL-SDR receive
utility, Pluto libiio runtime, Python bindings, and linked native libraries.
Extract the **whole package** and run its AETV executable. No Python, libiio,
rtl-sdr installation, or PATH edits are needed. The application loads its own
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
parser. It does not open hardware or transmit. Build-time reports are included
as `sdr-smoke.json` and `sdr-gui-smoke.json`; physical USB access depends on the
driver setup above. Source installations still need native SDR dependencies.
