"""Download the pinned current public base checkpoint and convert to FP16 GGUF."""

import hashlib
import os
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "nvidia/nemotron-3.5-asr-streaming-0.6b"
REVISION = "ea30d66debe3740a08b573244286791d423d6b3e"
FILENAME = "nemotron-3.5-asr-streaming-0.6b.nemo"
SHA256 = "210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74"
ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    destination = ROOT / "models/base"
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(
        hf_hub_download(
            REPO,
            FILENAME,
            revision=REVISION,
            local_dir=destination,
            token=os.getenv("HF_TOKEN") or False,
        )
    )
    with source.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != SHA256:
            raise RuntimeError("Base checkpoint checksum mismatch")
    converter = ROOT / "vendor/NeMo-Speech.cpp"
    commit = subprocess.check_output(
        ["git", "-C", str(converter), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != "6a3ca369370782acee0dd155e34394ba60b920c4":
        raise RuntimeError(
            "Run scripts/setup_nemo_cpp.py for the pinned converter first"
        )
    subprocess.run(
        [
            "uv",
            "run",
            str(converter / "convert_model.py"),
            str(source),
            "--outfile",
            str(destination / "base-f16.gguf"),
            "--outtype",
            "f16",
        ],
        check=True,
    )
