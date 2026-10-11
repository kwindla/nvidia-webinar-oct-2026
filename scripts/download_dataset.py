"""Download the pinned ATC preview dataset; supply HF_TOKEN in the environment."""

import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = "fciannella/atc02-preview"
REVISION = "973e9352164fa66c445227b442b2fd71b2aad171"
DESTINATION = Path(__file__).resolve().parents[1] / "datasets" / "atc02-preview"


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN to a token with access to the private dataset.")
    info = HfApi(token=token).dataset_info(REPO, revision=REVISION, files_metadata=True)
    snapshot_download(
        REPO,
        repo_type="dataset",
        revision=REVISION,
        local_dir=DESTINATION,
        token=token,
    )
    # Confirm every remote file is present and has its advertised size.
    for file in info.siblings:
        path = DESTINATION / file.rfilename
        if not path.is_file() or path.stat().st_size != file.size:
            raise RuntimeError(f"Missing or incomplete download: {file.rfilename}")
    print(f"Downloaded {len(info.siblings)} files to {DESTINATION}")
    print(f"Revision: {REVISION}")


if __name__ == "__main__":
    main()
