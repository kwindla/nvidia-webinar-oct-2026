"""Download the exact private checkpoint; supply HF_TOKEN in the environment."""
import hashlib
from pathlib import Path
from huggingface_hub import snapshot_download

REPO = 'fciannella/atc02-nemotron-asr-finetuned'
REVISION = '7b78fe63e6db3015ad8aa3df03422074f00dee73'

if __name__ == '__main__':
    directory = Path(snapshot_download(REPO, revision=REVISION, local_dir='models/atc02'))
    checkpoint = directory / 'nemotron-asr-atc-p3.nemo'
    expected = checkpoint.with_suffix('.nemo.sha256').read_text().split()[0]
    with checkpoint.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise RuntimeError(f'Checkpoint checksum mismatch: {actual}')
    print(f'Verified {checkpoint}: {actual}')
