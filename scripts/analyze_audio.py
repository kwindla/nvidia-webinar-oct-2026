"""Analyze local ATCO2 audio and fit candidate-background spectral profiles.

No audio is uploaded. Silero labels are estimates, not human noise annotations.
Run with: uv run --group analysis scripts/analyze_audio.py
"""

import argparse
import csv
import hashlib
import json
from importlib.resources import files
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from pipecat.audio.vad.silero import SileroOnnxModel
from scipy.ndimage import binary_dilation
from scipy.signal import periodogram

RATE, FRAME = 16000, 512
FLOOR = 1e-20


def db(power):
    return 10 * np.log10(np.maximum(power, FLOOR))


def long_runs(mask, minimum=6):
    """Keep only runs of at least 192 ms at this analysis frame size."""
    result = np.zeros_like(mask)
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    for start, end in edges.reshape(-1, 2):
        if end - start >= minimum:
            result[start:end] = True
    return result


def quantiles(values):
    a = np.array([v for v in values if v is not None], dtype=float)
    return (
        dict(zip(["p10", "median", "p90"], np.quantile(a, [0.1, 0.5, 0.9]).tolist()))
        if len(a)
        else None
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/atc02-preview/data/test")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/audio-analysis"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = args.dataset / "metadata.csv"
    rows = list(csv.DictReader(metadata.open()))
    model_path = files("pipecat.audio.vad.data").joinpath("silero_vad.onnx")
    model = SileroOnnxModel(str(model_path), force_onnx_cpu=True)
    results, profiles, regions = [], {}, []
    for index, row in enumerate(rows):
        x, rate = sf.read(args.dataset / row["file_name"], dtype="float32")
        if rate != RATE or x.ndim != 1:
            raise ValueError(f"Expected 16 kHz mono: {row['file_name']}")
        frames = x[: len(x) // FRAME * FRAME].reshape(-1, FRAME)
        model.reset_states()
        probability = np.array([model(frame, RATE).item() for frame in frames])
        power = np.mean(frames.astype(np.float64) ** 2, axis=1)
        speech = probability >= 0.5
        # Reject speech boundaries, model warm-up, digital silence, and short gaps.
        noise = (
            (probability < 0.1)
            & ~binary_dilation(speech, iterations=5)
            & (db(power) > -65)
        )
        noise[:5] = False
        noise = long_runs(noise)
        edges = np.flatnonzero(np.diff(np.r_[False, noise, False]))
        for start, end in edges.reshape(-1, 2):
            regions.append(
                {
                    "file_name": row["file_name"],
                    "airport": row["airport"],
                    "start_s": float(start * FRAME / RATE),
                    "end_s": float(end * FRAME / RATE),
                    "mean_speech_probability": float(probability[start:end].mean()),
                    "rms_dbfs": float(db(np.mean(power[start:end]))),
                }
            )
        freq, spectra = periodogram(
            frames, RATE, window="hann", detrend="constant", axis=-1
        )
        noise_db = float(db(np.mean(power[noise]))) if noise.any() else None
        speech_db = float(db(np.mean(power[speech]))) if speech.sum() >= 6 else None
        full_spectrum = np.mean(spectra, axis=0)
        total = max(full_spectrum.sum(), FLOOR)
        result = {
            "file_name": row["file_name"],
            "airport": row["airport"],
            "duration_s": len(x) / RATE,
            "rms_dbfs": float(db(np.mean(x.astype(float) ** 2))),
            "peak_dbfs": float(db(np.max(x.astype(float) ** 2))),
            "near_full_scale_fraction": float(np.mean(np.abs(x) >= 0.999)),
            "digital_zero_fraction": float(np.mean(x == 0)),
            "speech_fraction": float(speech.mean()),
            "noise_seconds": float(noise.sum() * FRAME / RATE),
            "noise_rms_dbfs": noise_db,
            "speech_rms_dbfs": speech_db,
            "speech_noise_contrast_db": speech_db - noise_db
            if speech_db is not None and noise_db is not None
            else None,
            "power_below_300_fraction": float(full_spectrum[freq < 300].sum() / total),
            "power_above_3400_fraction": float(
                full_spectrum[freq > 3400].sum() / total
            ),
        }
        results.append(result)
        if noise.any():
            shape = np.mean(spectra[noise], axis=0)
            shape /= max(shape.sum(), FLOOR)
            for key in ["all", row["airport"]]:
                profiles.setdefault(key, []).append(shape)
        if (index + 1) % 200 == 0:
            print(f"Analyzed {index + 1}/{len(rows)} files", flush=True)

    with (args.output / "per-file.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (args.output / "candidate-regions.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_name",
                "airport",
                "start_s",
                "end_s",
                "mean_speech_probability",
                "rms_dbfs",
            ],
        )
        writer.writeheader()
        writer.writerows(regions)
    fitted = {}
    for name, shapes in sorted(profiles.items()):
        # Equal weight per clip, robust log-spectrum median, then unit sum.
        shape = np.exp(np.median(np.log(np.maximum(shapes, FLOOR)), axis=0))
        shape /= shape.sum()
        fitted[name] = {"clips": len(shapes), "relative_power": shape.tolist()}
    profile = {
        "sample_rate": RATE,
        "frequency_hz": freq.tolist(),
        "profiles": fitted,
        "method": "Median normalized log PSD across clips; only conservative Silero candidate noise frames.",
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "vad_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    (args.output / "noise-profiles.json").write_text(
        json.dumps(profile, indent=2) + "\n"
    )
    summary = {
        "files": len(results),
        "hours": sum(r["duration_s"] for r in results) / 3600,
        "candidate_noise_seconds": sum(r["noise_seconds"] for r in results),
        "clips_with_candidate_noise": sum(r["noise_seconds"] > 0 for r in results),
        "clips_with_near_full_scale_samples": sum(
            r["near_full_scale_fraction"] > 0 for r in results
        ),
        "metrics": {
            key: quantiles(r[key] for r in results)
            for key in [
                "rms_dbfs",
                "noise_rms_dbfs",
                "speech_noise_contrast_db",
                "power_below_300_fraction",
                "power_above_3400_fraction",
                "speech_fraction",
            ]
        },
        "airports": {
            name: {
                "files": len(group),
                "noise_clips": sum(r["noise_seconds"] > 0 for r in group),
                "candidate_noise_seconds": sum(r["noise_seconds"] for r in group),
                "noise_rms_dbfs": quantiles(r["noise_rms_dbfs"] for r in group),
                "speech_noise_contrast_db": quantiles(
                    r["speech_noise_contrast_db"] for r in group
                ),
            }
            for name in sorted({r["airport"] for r in results})
            for group in [[r for r in results if r["airport"] == name]]
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name, item in fitted.items():
        shape = np.array(item["relative_power"])
        axes[0, 0].plot(
            freq,
            db(shape / shape.max()),
            label=f"{name} ({item['clips']})",
            lw=2 if name == "all" else 1,
        )
    axes[0, 0].set(
        xlabel="Frequency (Hz)",
        ylabel="Relative power (dB)",
        ylim=(-70, 3),
        title="Candidate-background spectral shape (clips)",
    )
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].hist(
        [r["rms_dbfs"] for r in results], bins=45, label="Whole clip", alpha=0.65
    )
    axes[0, 1].hist(
        [r["noise_rms_dbfs"] for r in results if r["noise_rms_dbfs"] is not None],
        bins=45,
        label="Candidate noise",
        alpha=0.65,
    )
    axes[0, 1].set(
        xlabel="RMS dBFS", ylabel="Clips", title="Recorded levels (not speaker SPL)"
    )
    axes[0, 1].legend()
    axes[1, 0].hist(
        [
            r["speech_noise_contrast_db"]
            for r in results
            if r["speech_noise_contrast_db"] is not None
        ],
        bins=40,
    )
    axes[1, 0].set(
        xlabel="Speech-active / candidate-noise power ratio (dB)",
        ylabel="Clips",
        title="Within-clip contrast proxy — not calibrated SNR",
    )
    names = list(summary["airports"])
    axes[1, 1].bar(
        names,
        [
            summary["airports"][n]["noise_clips"]
            / summary["airports"][n]["files"]
            * 100
            for n in names
        ],
    )
    axes[1, 1].set(
        ylabel="Clips with candidate noise (%)",
        ylim=(0, 100),
        title="Noise-profile coverage / selection bias",
    )
    fig.savefig(args.output / "overview.png", dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
