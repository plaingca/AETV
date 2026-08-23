# Kiwi propagation planner

When **Public KiwiSDR** is selected, the normal Receive pane automatically
ranks usable receivers and displays a prominent **Recommended now** card. The
recommended host is selected before receiving starts. **Find best RF path**
compares every configured planning dial and a distance-diverse set of usable
Kiwis, then selects the strongest frequency/receiver pair. **Compare paths…**
opens the next-twelve-hours shortlist for manual
inspection or override. Automatic selection can be disabled beside the card or
persistently in Settings.

The RF search radius expands with frequency so higher-band skip is not clipped
by the local-receiver radius: the defaults rise from 2,500 km on 80 m to 15,000
km on 12/10 m. The user-configured radius remains a floor. Edit **RF planning
dials** in Settings to match the station's licence, regional band plan, antenna,
and AETV occupied bandwidth. Selection is a propagation estimate only: it does
not check whether a channel is occupied and never starts reception or keys PTT.

The detailed **Kiwi path planner** ranks usable public receivers by the
probability that AETV pilot SNR will exceed 9 dB now or during the next twelve
hours. Distance is shown for context but is not the ranking criterion.

The preferred engine is the [ITU-R Study Group 3 reference implementation](https://github.com/ITU-R-Study-Group-3/ITU-R-HF)
of Recommendations [P.533-14](https://www.itu.int/rec/r-rec-p.533/en) and
[P.372](https://www.itu.int/rec/R-REC-P.372-17-202408-I). On Windows, install its current-month data
and native runtime with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_iturhfprop.ps1
```

Use `-AllMonths` to cache all monthly maps (about 135 MB). Without the native
runtime the table remains available using a clearly labelled coarse fallback;
its absolute SNR values should not be treated as P.533 predictions.

## Calibrating a station

1. Enter the station QTH, transmit frequency and Flex transmit power in
   Settings.
2. Accept the Receive pane's automatic recommendation, or open **Compare paths…**
   and select a different receiver.
3. Click **Use selected receiver** and start receiving.
4. Make a normal, operator-controlled AETV transmission.

When the Kiwi decodes a callsign matching the configured station callsign,
AETV records one calibration observation per receiver/frequency/minute. The
observation contains measured pilot SNR, the contemporaneous P.533 prediction,
path bearing and power. It does not record unrelated stations.

When a radio-routed transmission lasts at least three seconds while the
selected Kiwi is receiving, but no matching callsign decodes (including during
a five-second decoder grace period), AETV stores a censored **miss** instead of
inventing an SNR value. Exact receiver/frequency misses immediately reduce that
pair's ranking; successful later probes replace a same-minute miss.

Calibration residuals are direction- and band-weighted and shrunk toward zero
until enough observations exist. This lets the planner learn the combined
effects of the station QTH, transmit antenna pattern and receiver-specific
noise without overfitting one favourable fade. Measurements are stored in
`propagation_measurements.json` under the AETV configuration directory.

For antenna-pattern identification, collect several measurements to multiple
Kiwis in different bearings at similar times. Simultaneous measurements are
best; repeated observations across several days and local-time windows are
needed before treating the correction as stable.

### FT8 / PSK Reporter calibration

The quickest wide-area survey is **Receive → FT8 propagation calibration…**,
also available from the path planner's **Calibrate with FT8…** button. It sends
one standard `CQ CALL GRID` FT8 message on a selected band through configured
Flex direct-VITA audio. Every emission requires its own confirmation; the tool
never sweeps bands unattended. It does not determine whether the frequency is
legal or clear, so listen and verify the dial, 1000 Hz audio offset, bandwidth,
power, and antenna before confirming.

Install the pinned MIT-licensed `ft8_lib` encoder once with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_ft8_lib.ps1
```

After a probe, wait at least five minutes and click **Import PSK Reporter
spots**. The importer observes PSK Reporter's five-minute query interval and
accepts only reports matching a locally recorded transmit slot and RF
frequency. This time/frequency correlation distinguishes a calibration probe
from ordinary FT8 activity without altering the licensed callsign. Reporter
locator and received SNR become station calibration observations; the path
planner then learns the combined antenna, terrain, propagation, and remote
receiver-noise error. Probe runs are retained for 24 hours for importing.

Corrections remain within the measured amateur band; for example, strong 15 m
reports do not inflate an unmeasured 12 m forecast. A completed probe with no
PSK Reporter reception is retained as conservative, band-wide censored
evidence rather than being silently discarded. Import immediately invalidates
the visible receiver ranking and open planner so their displayed corrections
and sample counts reflect the new observations.

The default dials cover common FT8 activity from 80 through 10 metres and are
editable as **FT8 probe dials** in Settings. They are starting defaults only;
regional band plans and operator privileges take precedence.
