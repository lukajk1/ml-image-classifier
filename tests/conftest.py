"""Shared fixtures. Nothing here needs the gitignored model weights or checkpoints."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent
for path in (REPO, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch):
    # Developer machines may point these at a real runtime pack.
    for name in ("UMBRA_ML_ASSETS", "UMBRA_ML_HOME", "UMBRA_ML_PYTHON"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def png_bytes():
    from PIL import Image
    buffer = BytesIO()
    Image.new("RGB", (32, 24), (200, 30, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def image_file(tmp_path, png_bytes):
    path = tmp_path / "images" / "sample.png"
    path.parent.mkdir()
    path.write_bytes(png_bytes)
    return path


@pytest.fixture
def ml_root(tmp_path):
    """A minimal ML folder whose registry uses the stub adapters in tests/stub_adapters.py."""
    root = tmp_path / "ml"
    root.mkdir()
    write_json(root / "runtime-requirements.json", {"format": 1, "runtimeId": "1", "python": "3.12"})
    write_json(root / "models.json", {
        "defaultModel": "stub-a",
        "models": [
            {"id": "stub-a", "name": "Stub A", "adapter": "stub_adapters:StubClassifier", "checkpoint": "a.pth", "labels": ["cat", "dog"]},
            {"id": "stub-b", "name": "Stub B", "adapter": "stub_adapters:StubClassifier", "checkpoint": "b.pth", "labels": ["x"]},
            {"id": "missing", "name": "Missing", "adapter": "stub_adapters:StubClassifier", "checkpoint": "nope.pth"},
        ],
    })
    (root / "a.pth").write_bytes(b"checkpoint-a")
    (root / "b.pth").write_bytes(b"checkpoint-b")

    vision = root / "models" / "vision"
    write_json(vision / "config.json", {"hidden_size": 4})
    (vision / "model.safetensors").write_bytes(b"fake vision weights")
    write_json(vision / "bundle.json", {
        "modelId": "stub/vision", "revision": "r1", "preprocessingVersion": "stub-v1", "dimension": 4,
        "files": {name: file_hash(vision / name) for name in ("config.json", "model.safetensors")},
    })
    write_json(root / "embedding.json", {"adapter": "stub_adapters:StubEmbedder", "directory": "models/vision"})

    text = root / "models" / "text"
    (text / "model.safetensors").parent.mkdir(parents=True)
    (text / "model.safetensors").write_bytes(b"fake text weights")
    write_json(text / "bundle.json", {
        "modelId": "stub/text", "revision": "r1", "compatibleEmbeddingKey": "vision-key", "dimension": 4,
        "prompt": "a photo of {}.", "files": {"model.safetensors": file_hash(text / "model.safetensors")},
    })
    write_json(root / "tagging.json", {
        "adapter": "stub_adapters:StubTagSuggester", "directory": "models/text",
        "visionDirectory": "models/vision", "vocabulary": "vocab.json",
    })
    write_json(root / "vocab.json", ["sky", "Cat", "tree"])
    return root
