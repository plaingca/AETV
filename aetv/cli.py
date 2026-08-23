"""Ham-facing AETV command line: send, receive, simulate, train, eval."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from .audio_io import list_devices, play_audio, read_wav, record_audio, write_wav
from .codec import AETVCodec, DEFAULT_CHECKPOINT
from .config import AETV_MODES
from .hfchannel import awgn, fading, freq_shift
from .modem import demodulate_gop_stream, modulate_gop_stream
from .source import collect_gops, iter_video_file, iter_webcam, write_mp4, write_side_by_side
from .sync import SyncError


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", default="V7", choices=list(AETV_MODES), help="waveform / video mode")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="inference checkpoint (default: the selected mode's release checkpoint)",
    )
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda or cpu")
    parser.add_argument("--callsign", default="N0CALL", help="station identification carried on the beacon")


def _load_codec(args) -> AETVCodec:
    return AETVCodec(checkpoint=args.checkpoint, device=args.device, mode=args.mode)


def _encode_source(args, codec: AETVCodec) -> tuple[np.ndarray, list[np.ndarray]]:
    mode = codec.mode
    if args.source.lower() in {"webcam", "cam", "camera"}:
        if args.gops < 1:
            raise SystemExit("--gops must be >= 1 for webcam capture")
        frames = collect_gops(iter_webcam(mode, camera=args.camera, duration_s=args.gops), mode)
    else:
        total_frames = args.gops * mode.gop_frames
        frames = iter_video_file(args.source, mode, start_s=args.start, frames=total_frames)
    latents = []
    for index in range(args.gops):
        gop = frames[index * mode.gop_frames : (index + 1) * mode.gop_frames]
        latents.append(codec.encode_gop(gop))
    return frames, latents


def cmd_send(args) -> int:
    codec = _load_codec(args)
    frames, latents = _encode_source(args, codec)
    audio = modulate_gop_stream(latents, mode_name=codec.mode.name, callsign=args.callsign)
    wav_path = write_wav(args.out, codec.mode.geometry.fs, audio, peak=args.peak)
    print(
        f"encoded {args.gops} GOP(s) of {codec.mode.width}x{codec.mode.height} "
        f"@ {codec.mode.fps:g} fps -> {wav_path} "
        f"({len(audio) / codec.mode.geometry.fs:.2f} s @ {codec.mode.geometry.fs} Hz)",
        flush=True,
    )
    if args.play:
        play_audio(audio, codec.mode.geometry.fs, device=args.audio_device)
    if args.flex_host:
        from .flex import send_wav

        report = send_wav(
            wav_path,
            host=args.flex_host,
            device=args.audio_device or "DAX TX (FlexRadio DAX)",
            power=args.power,
            filter_low=int(codec.mode.geometry.tx_bandpass[0]),
            filter_high=int(codec.mode.geometry.tx_bandpass[1]),
            freq_mhz=args.freq_mhz,
            require_mode=args.require_mode,
            audio_only=args.audio_only,
        )
        print(json.dumps({k: v for k, v in report.items() if k != "transcript"}, indent=2))
    if args.source_mp4:
        write_mp4(frames, Path(args.source_mp4), codec.mode.fps)
    return 0


def _decode_audio(codec: AETVCodec, audio: np.ndarray) -> tuple[np.ndarray, dict]:
    result = demodulate_gop_stream(audio, band=codec.mode.band, drift_track="off")
    decoded = []
    for latents, weights in zip(result.gops_latents, result.gops_weights):
        decoded.append(codec.decode_gop(latents, weights))
    if not decoded:
        raise SyncError("demodulator returned no GOPs")
    video = np.concatenate(decoded, axis=0)
    info = {
        "frames_received": result.frames_received,
        "gops": len(result.gops_latents),
        "freq_offset_hz": result.freq_offset,
        "sync_metric": result.sync_metric,
        "pilot_snr_db": result.snr_db,
        "callsign": result.callsign,
        "mode": result.mode.name,
    }
    return video, info


def cmd_receive(args) -> int:
    codec = _load_codec(args)
    if args.wav:
        rate, audio = read_wav(args.wav)
        if rate != codec.mode.geometry.fs:
            from .audio_io import resample_audio

            audio = resample_audio(audio, rate, codec.mode.geometry.fs)
    elif args.record_device is not None or args.duration:
        if args.duration is None:
            raise SystemExit("soundcard receive needs --duration")
        audio = record_audio(args.duration, codec.mode.geometry.fs, device=args.record_device)
        if args.save_wav:
            write_wav(args.save_wav, codec.mode.geometry.fs, audio)
    else:
        raise SystemExit("pass --wav PATH or --duration SECONDS for soundcard capture")

    try:
        video, info = _decode_audio(codec, audio)
    except SyncError as error:
        raise SystemExit(f"receive failed: {error}") from error
    print(json.dumps(info, indent=2), flush=True)
    if args.out:
        write_mp4(video, Path(args.out), codec.mode.fps)
        print(f"wrote {args.out} ({len(video)} frames)", flush=True)
    if args.display:
        _display_video(video, codec.mode.fps)
    return 0


def cmd_simulate(args) -> int:
    codec = _load_codec(args)
    frames, latents = _encode_source(args, codec)
    audio = modulate_gop_stream(latents, mode_name=codec.mode.name, callsign=args.callsign)
    impaired = audio.copy()
    fs = codec.mode.geometry.fs
    if args.fading != "none":
        impaired = fading(impaired, preset=args.fading, seed=args.seed, fs=fs)
    if args.cfo:
        impaired = freq_shift(impaired, args.cfo, fs=fs)
    if args.snr is not None:
        impaired = awgn(impaired, snr_db=args.snr, seed=args.seed, fs=fs)
    video, info = _decode_audio(codec, impaired)
    usable = min(len(frames), len(video))
    mse = np.mean((frames[:usable].astype(np.float32) - video[:usable].astype(np.float32)) ** 2) / (255.0**2)
    psnr = float("inf") if mse <= 0 else -10.0 * math.log10(mse)
    info["psnr_db"] = psnr
    info["snr_db"] = args.snr
    info["fading"] = args.fading
    print(json.dumps(info, indent=2), flush=True)
    if args.out:
        write_side_by_side(frames[:usable], video[:usable], Path(args.out), codec.mode.fps)
        print(f"wrote comparison video {args.out}", flush=True)
    return 0


def cmd_devices(_args) -> int:
    try:
        devices = list_devices()
    except Exception as error:
        raise SystemExit(f"sounddevice unavailable: {error}") from error
    for item in devices:
        print(
            f"{item['index']:3d}  in={item['inputs']} out={item['outputs']}  "
            f"{item['name']}  ({item['default_rate']} Hz)"
        )
    return 0


def cmd_train(args) -> int:
    script = _script_path("train.py")
    return _exec_script(script, args.train_args)


def cmd_gui(_args) -> int:
    from .gui.app import main as gui_main

    return gui_main()


def cmd_kiwi_list(args) -> int:
    from .kiwi import find_receivers

    receivers = find_receivers(args.lat, args.lon, max_km=args.max_km)
    usable = [item for item in receivers if item.usable]
    print(f"reachable within {args.max_km:.0f} km: {len(receivers)}")
    print(f"API enabled with a free channel: {len(usable)}")
    print(f"{'km':>6}  {'api':>3} {'free':>4} {'host':<38} loc")
    for item in receivers:
        flag = "  <-- usable" if item.usable else ""
        print(
            f"{item.km or 0:6.0f}  {item.ext_api:3d} {item.free:4d} "
            f"{item.host:<38} {item.loc}{flag}"
        )
    return 0


def cmd_eval(args) -> int:
    script = _script_path("eval.py")
    forwarded = list(args.eval_args)
    if args.checkpoint and "--checkpoint" not in forwarded:
        forwarded = ["--checkpoint", args.checkpoint, *forwarded]
    return _exec_script(script, forwarded)


def _script_path(name: str) -> Path:
    """Find a research script in either a source checkout or an installed wheel."""
    package_dir = Path(__file__).resolve().parent
    candidates = (package_dir / "_scripts" / name, package_dir.parent / "scripts" / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"AETV installation is missing its {name} script")


def _exec_script(script: Path, extra: list[str]) -> int:
    import runpy

    sys.argv = [str(script), *extra]
    runpy.run_path(str(script), run_name="__main__")
    return 0


def _display_video(frames: np.ndarray, fps: float) -> None:
    try:
        import cv2
    except ImportError as error:
        raise SystemExit("opencv-python is required for --display") from error
    delay = max(1, int(round(1000.0 / fps)))
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        scale = max(1, 480 // frame.shape[0])
        shown = cv2.resize(bgr, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("AETV", shown)
        if cv2.waitKey(delay) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aetv",
        description="Autoencoder Television: analog video over HF OFDM for amateur radio.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    send = sub.add_parser("send", help="encode webcam or file video to an AETV waveform")
    _add_common(send)
    send.add_argument("--source", required=True, help="video file, or 'webcam'")
    send.add_argument("--start", type=float, default=0.0, help="source start time in seconds")
    send.add_argument("--gops", type=int, default=10, help="number of 1-second GOPs to send")
    send.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    send.add_argument("--out", default="aetv_tx.wav", help="output WAV path")
    send.add_argument("--peak", type=float, default=0.7)
    send.add_argument("--play", action="store_true", help="play the waveform on a soundcard")
    send.add_argument("--audio-device", default=None, help="sounddevice output name or index")
    send.add_argument("--source-mp4", default=None, help="optional copy of the captured source frames")
    send.add_argument("--flex-host", default=None, help="FlexRadio IP; keys DAX TX after encoding")
    send.add_argument("--power", type=int, default=5)
    send.add_argument("--freq-mhz", type=float, default=None)
    send.add_argument("--require-mode", default="DIGU")
    send.add_argument("--audio-only", action="store_true", help="Flex DAX playback without keying")
    send.set_defaults(func=cmd_send)

    recv = sub.add_parser("receive", help="demodulate a WAV or soundcard capture back to video")
    _add_common(recv)
    recv.add_argument("--wav", default=None, help="captured passband WAV")
    recv.add_argument("--duration", type=float, default=None, help="seconds to record from a soundcard")
    recv.add_argument("--record-device", default=None, help="sounddevice input name or index")
    recv.add_argument("--save-wav", default=None, help="write the raw capture before decode")
    recv.add_argument("--out", default="aetv_rx.mp4", help="decoded video path")
    recv.add_argument("--display", action="store_true", help="preview decoded frames")
    recv.set_defaults(func=cmd_receive)

    sim = sub.add_parser("simulate", help="encode, impair, and decode without a radio")
    _add_common(sim)
    sim.add_argument("--source", required=True)
    sim.add_argument("--start", type=float, default=0.0)
    sim.add_argument("--gops", type=int, default=4)
    sim.add_argument("--camera", type=int, default=0)
    sim.add_argument("--snr", type=float, default=12.0)
    sim.add_argument("--cfo", type=float, default=0.0)
    sim.add_argument("--fading", choices=("none", "mpg", "mpp", "mpd"), default="none")
    sim.add_argument("--seed", type=int, default=42)
    sim.add_argument("--out", default="aetv_sim.mp4")
    sim.set_defaults(func=cmd_simulate)

    devices = sub.add_parser("devices", help="list soundcard inputs and outputs")
    devices.set_defaults(func=cmd_devices)

    train = sub.add_parser("train", help="run the training script")
    train.add_argument("train_args", nargs=argparse.REMAINDER)
    train.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="run the held-out clip evaluation")
    ev.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ev.add_argument("eval_args", nargs=argparse.REMAINDER)
    ev.set_defaults(func=cmd_eval)

    gui = sub.add_parser("gui", help="open the ham-station GUI")
    gui.set_defaults(func=cmd_gui)

    kiwi = sub.add_parser("kiwi-list", help="find public KiwiSDR receivers that allow API capture")
    kiwi.add_argument("--lat", type=float, default=49.26)
    kiwi.add_argument("--lon", type=float, default=-123.11)
    kiwi.add_argument("--max-km", type=float, default=2500.0)
    kiwi.set_defaults(func=cmd_kiwi_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    extra = args.train_args if args.command == "train" else args.eval_args if args.command == "eval" else None
    if extra and extra[:1] == ["--"]:
        extra[:] = extra[1:]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
