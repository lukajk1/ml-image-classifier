"""Build-time only: download a pinned checkpoint and export its vision tower.

Distribute models/siglip2-base-224 alongside runner.py. Runtime never downloads.
"""
import json
from pathlib import Path
from hashlib import sha256
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, SiglipVisionModel

ROOT = Path(__file__).resolve().parent
REPOSITORY = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
DESTINATION = ROOT / "models" / "siglip2-base-224"

if __name__ == "__main__":
    DESTINATION.mkdir(parents=True, exist_ok=True)
    print("Downloading pinned SigLIP 2 checkpoint for the app bundle…", flush=True)
    model = SiglipVisionModel.from_pretrained(REPOSITORY, revision=REVISION)
    processor = AutoImageProcessor.from_pretrained(REPOSITORY, revision=REVISION, use_fast=False)
    model.save_pretrained(DESTINATION, safe_serialization=True)
    processor.save_pretrained(DESTINATION)
    readme = hf_hub_download(REPOSITORY, "README.md", revision=REVISION)
    (DESTINATION / "MODEL_CARD.md").write_bytes(Path(readme).read_bytes())
    # Attribute the upstream model and retain its Apache-2.0 license.
    import urllib.request
    license_text = urllib.request.urlopen("https://www.apache.org/licenses/LICENSE-2.0.txt").read()
    (DESTINATION / "LICENSE.txt").write_bytes(license_text)
    files = {}
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        digest = sha256()
        with (DESTINATION / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files[name] = digest.hexdigest()
    (DESTINATION / "bundle.json").write_text(json.dumps({
        "modelId": REPOSITORY, "revision": REVISION,
        "preprocessingVersion": "siglip2-224-pil-exif-rgb-v1",
        "dimension": model.config.hidden_size, "files": files,
    }, indent=2), encoding="utf-8")
    print(f"Bundled vision model in {DESTINATION}", flush=True)
