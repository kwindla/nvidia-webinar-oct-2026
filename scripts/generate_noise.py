"""Generate reproducible, speech-free Gaussian noise colored by measured profiles."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def synthesize(profile, *, name="all", seconds=60, rms_dbfs=-30, seed=42):
    if not np.isfinite(seconds) or not 1 <= seconds <= 600:
        raise ValueError("Duration must be between 1 and 600 seconds.")
    if not np.isfinite(rms_dbfs) or not -60 <= rms_dbfs <= -12:
        raise ValueError("RMS must be between -60 and -12 dBFS.")
    rate = profile["sample_rate"]
    spectrum = profile["profiles"][name]["relative_power"]
    frequency = profile["frequency_hz"]
    count = round(rate * seconds)
    rng = np.random.default_rng(seed)
    fft_frequency = np.fft.rfftfreq(count, 1 / rate)
    color = np.sqrt(np.interp(fft_frequency, frequency, spectrum))
    color[0] = 0  # No DC component.
    signal = np.fft.irfft(np.fft.rfft(rng.standard_normal(count)) * color, n=count)
    signal = resample_poly(signal, 3, 1)  # 16 kHz source profile -> 48 kHz playback.
    fade = min(2400, len(signal) // 2)
    ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
    signal[:fade] *= ramp
    signal[-fade:] *= ramp[::-1]
    signal *= 10 ** (rms_dbfs / 20) / np.sqrt(np.mean(signal**2))
    if np.max(np.abs(signal)) >= 0.99:
        raise ValueError("Requested RMS would clip; choose a lower --rms-dbfs.")
    return signal.astype("float32"), rate * 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("artifacts/audio-analysis/noise-profiles.json"),
    )
    parser.add_argument(
        "--profile", default="all", help="all or an airport code from the profile file"
    )
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--rms-dbfs", type=float, default=-30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/audio-analysis/noise-all.wav")
    )
    args = parser.parse_args()
    profile = json.loads(args.profiles.read_text())
    if args.profile not in profile["profiles"]:
        parser.error(f"Choose one of: {', '.join(profile['profiles'])}")
    try:
        signal, rate = synthesize(
            profile,
            name=args.profile,
            seconds=args.seconds,
            rms_dbfs=args.rms_dbfs,
            seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, signal, rate, subtype="PCM_16")
    manifest = {
        "profile": args.profile,
        "source_profiles": str(args.profiles),
        "profiles_sha256": hashlib.sha256(args.profiles.read_bytes()).hexdigest(),
        "seed": args.seed,
        "duration_s": len(signal) / rate,
        "sample_rate": rate,
        "rms_dbfs": args.rms_dbfs,
        "peak_dbfs": float(20 * np.log10(np.max(np.abs(signal)))),
        "method": "Seeded Gaussian noise, FFT-colored to candidate-background PSD; 50 ms fades. No source waveform copied.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Wrote {args.output}: {args.seconds:g}s, {rate} Hz, RMS {args.rms_dbfs:g} dBFS"
    )
    print(
        "Playback volume is separate from dBFS. Start with low speaker volume; Ctrl+C stops ffplay."
    )


if __name__ == "__main__":
    main()
