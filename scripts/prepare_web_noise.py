"""Generate only synthetic noise assets for the browser; run before npm build."""

import hashlib
import json
from pathlib import Path

import soundfile as sf
from generate_noise import synthesize

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "artifacts/audio-analysis/noise-profiles.json"
    profile = json.loads(source.read_text())
    output = ROOT / "client/public/noise"
    output.mkdir(parents=True, exist_ok=True)
    for name in ["all", "LKPR", "LKTB", "LSGS", "LSZB", "LSZH", "LZIB", "YSSY"]:
        signal, rate = synthesize(profile, name=name, seconds=60, rms_dbfs=-30, seed=42)
        sf.write(output / f"noise-{name}.wav", signal, rate, subtype="PCM_16")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "profiles_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "seconds": 60,
                "rms_dbfs": -30,
                "seed": 42,
                "description": "Synthetic Gaussian noise; no dataset audio copied.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Prepared 8 synthetic noise tracks in {output}")


if __name__ == "__main__":
    main()
