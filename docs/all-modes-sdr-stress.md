# All-mode SDR stress testing

The opt-in GUI harness runs the normal codec, modulator, Pluto driver, RTL
capture, receiver, program-audio output and GUI timers. It replaces camera
frames and microphone input with identifiable fixtures. It transmits RF only
when its configuration explicitly includes `transmit: true`.

Released GUI modes are `V8`, `V7`, `AC16`, `V8_AV`, and `AC16_AV`. V0–V6 have
protocol tests but no released GUI model bundles; a protocol test is not a
hardware or reconstructed-video qualification. Model weights and wire formats
are unchanged by the receiver fixes.

## Reproduce a bounded trial

Use a training-disjoint uint8 RGB `.npy` fixture and an existing audio output.
A/V uses a different test tone for each source second. Use an output directory
that does not already exist; local storage avoids perturbing measurement.
The harness isolates its saved station profile from the operator's settings.

```json
{
  "stress": true,
  "transmit": true,
  "mode": "V8_AV",
  "source": "/absolute/path/source.npy",
  "source_fps": 10,
  "output": "/absolute/path/trial-V8_AV",
  "gops": 600,
  "device": "cuda",
  "receiver": "rtlsdr",
  "rtl_serial": "1001",
  "pluto_uri": "ip:192.168.2.1",
  "frequency_mhz": 439,
  "tx_gain": -10,
  "rx_gain": 37.2,
  "auto_correct": true,
  "audio_output": "aetv_stress",
  "gain_schedule": [
    {"at_s": 180, "tx_gain": -47},
    {"at_s": 300, "tx_gain": -89.75},
    {"at_s": 306, "tx_gain": -10},
    {"at_s": 380, "tx_gain": 0},
    {"at_s": 410, "tx_gain": -10}
  ],
  "decode_stalls": [{"at_s": 440, "seconds": 2}],
  "gui_stalls": [{"at_s": 470, "seconds": 0.35}],
  "converter_stalls": [{"at_s": 500, "seconds": 10}]
}
```

```bash
AETV --ota-validation trial.json
```

Times are relative to the first actual IIO write. `join_s` starts a fresh
receiver that many seconds after RF begins. `stress: true` permits 1–900 GOPs;
the normal GUI duration control is unchanged. Camera input remains bounded
and drops oldest frames under an injected encoder stall. Every consumed frame
retains its actual source identity and availability timestamp.

`validation.json` records input/encode/decode/presentation times, source hash,
PortAudio writes and underflow flags, IIO writes, injected faults, received
sample positions, and verified beacon frame counters. `health.jsonl` samples
queue occupancy, high-water marks, processing duration, RSS, drops and overruns.
The raw `rtl.cu8`, sent/received latents, decoded RGB and paired audio permit
independent analysis. All trace queues are bounded and report incomplete
recordings. `completed` means the harness finished; it is not a quality pass.

Pluto is shut down in the transport's `finally` path. Verify its LO powerdown
and minimum gain from a fresh context after each trial. Do not run another
hardware trial while this one owns the radios.

## Replay the same signal

```bash
python scripts/replay_sdr_stress.py trial-V8_AV/rtl.cu8 \
  --mode V8_AV --join 187.137 --limit 45 --output replay/weak-late
```

This starts a fresh production receiver at an arbitrary sample without source
frames, transmitted latents, or transmitter timing. It saves latents, confidence
weights, paired audio, counters and acquisition events. `--correction HZ` is an
explicit manual-tuning control arm; otherwise tuning uses received I/Q only.
Score missing and ambiguous GOPs separately from reconstruction quality. Use
identical captured samples and model/runtime for paired receiver comparisons.

## Receiver changes exercised by this harness

- All released carrier banks support received-only coarse tuning. Boundary
  peaks, narrow interferers and unoccupied banks cannot establish calibration.
- Timing recovery uses analytic cyclic-prefix correlation, avoiding conjugate
  cancellation during oscillator drift. Guarded pilot comparisons repair short
  insertions/deletions. CRC-verified beacon counters correct whole-frame phase
  after larger transport jumps.
- An I/Q queue overflow explicitly discards old samples and resets demodulation
  and A/V clocks. Reception resumes automatically. Exact discarded sample
  counts are logged.
- Both A/V modes associate voice with received GOP positions. Absolute playout
  deadlines avoid accumulating polling jitter. Video is released when its
  paired audio reaches the output worker, with bounded queues and whole-pair
  drops if the endpoint stalls. Silence keeps the audio device clock running
  during RF gaps. A failed speaker no longer stops video reception.
- GUI playback tolerates ordinary GOP-arrival jitter and skips missed display
  instants after a stall. Waterfall pixel processing uses array operations to
  reduce interpreter contention with capture.
- Optional WAV/JSONL recording uses bounded background writes. On overflow or
  disk failure it retains an explicitly incomplete prefix and reports the error
  instead of blocking live RF processing.

## Interpretation limits

RTL's command-line stream has no hardware sample timestamps or reliable USB
loss counter. Queue overrun and read-gap measurements are host observations;
zero queue overruns does not prove lossless USB capture. Pluto starvation
telemetry likewise observes the software producer, not a DAC status register.
A clocked null audio sink exercises PortAudio scheduling but does not measure
physical speaker latency. Frame-availability-to-presentation latency excludes
camera exposure and display scanout, so it is not optical glass-to-glass.

The 2026-09-14 matrix uses Pluto at 439 MHz and RTL serials 1000, 1001, 1003 and
1004. Serial 1002 is excluded because of its known hardware fault. Radios are
approximately ten feet apart. The 60-second fixture cycles 30 training-disjoint
validation clips; they were used for model selection and are not an untouched
confirmation set. Detailed final measurements are retained with the trial
artifacts; Windows and physical HackRF behavior require their own hardware
validation.
