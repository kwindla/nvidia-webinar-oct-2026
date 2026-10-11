"""Install NVIDIA's pinned Linux x86_64 CUDA server and convert our checkpoint."""
import hashlib
import subprocess
import tarfile
import urllib.request
from pathlib import Path

RELEASE = 'v0.2.0'
COMMIT = '6a3ca369370782acee0dd155e34394ba60b920c4'
ARCHIVE = 'nemo-speech-0.2.0-linux-x86_64-cuda.tar.gz'
SHA256 = 'aef573fcc6c2c759401d2ed03e5ca9e401ed23b0e38cb852d917080c1e2d4515'

if __name__ == '__main__':
    Path('.tools').mkdir(exist_ok=True)
    archive = Path('.tools/nemo-speech.tar.gz')
    if not archive.exists():
        urllib.request.urlretrieve(
            f'https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/{RELEASE}/{ARCHIVE}', archive,
        )
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != SHA256:
            raise RuntimeError('NVIDIA release checksum mismatch')
    with tarfile.open(archive) as bundle:
        # Skip identical installed files so setup can be rerun while serving.
        for member in bundle.getmembers():
            target = Path('.tools/nemo-speech') / member.name
            if member.isfile() and target.is_file():
                with target.open('rb') as installed, bundle.extractfile(member) as packaged:
                    if (hashlib.file_digest(installed, 'sha256').digest()
                            == hashlib.file_digest(packaged, 'sha256').digest()):
                        continue
            bundle.extract(member, '.tools/nemo-speech', filter='data')
    source = Path('vendor/NeMo-Speech.cpp')
    if not source.exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', RELEASE,
                        'https://github.com/NVIDIA/NeMo-Speech.cpp.git', str(source)], check=True)
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise RuntimeError(f'Expected converter revision {COMMIT}, found {actual}')
    subprocess.run(['uv', 'run', str(source / 'convert_model.py'),
                    'models/atc02/nemotron-asr-atc-p3.nemo', '--outfile',
                    'models/atc02/atc02-f16.gguf', '--outtype', 'f16'], check=True)
