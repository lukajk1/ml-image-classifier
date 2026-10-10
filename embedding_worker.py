"""Embedding protocol; adapters are independent of classification and storage."""
from contextlib import redirect_stdout
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sys
from runtime_paths import asset_root


class EmbeddingWorker:
    def __init__(self, root):
        self.root = Path(root)
        spec = json.loads((self.root / "embedding.json").read_text(encoding="utf-8"))
        self.directory = asset_root(self.root) / spec["directory"]
        manifest = self.directory / "bundle.json"
        if not manifest.is_file():
            raise ValueError("Bundled similarity model is missing. Include models/siglip2-base-224 in the app's ML folder. No runtime download is performed.")
        self.bundle = json.loads(manifest.read_text(encoding="utf-8"))
        self.spec = spec
        self.key = sha256(json.dumps({"bundle": self.bundle, "adapter": spec["adapter"]}, sort_keys=True).encode()).hexdigest()
        self.adapter = None

    def load(self):
        if self.adapter is not None:
            return
        for filename, expected in self.bundle["files"].items():
            path = (self.directory / filename).resolve()
            if not path.is_relative_to(self.directory.resolve()):
                raise ValueError("Invalid bundled model path.")
            digest = sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"Bundled model file is damaged: {filename}")
        module, factory = self.spec["adapter"].split(":", 1)
        with redirect_stdout(sys.stderr):
            self.adapter = getattr(import_module(module), factory)(self.directory)

    def dispatch(self, request):
        metadata = {"key": self.key, "modelId": self.bundle["modelId"],
                    "dimension": self.bundle["dimension"], "revision": self.bundle["revision"],
                    "preprocessing": self.bundle["preprocessingVersion"]}
        if request["method"] == "embedding.info":
            return metadata
        self.load()
        if request["method"] == "embedding.load":
            return metadata
        if request["method"] != "embedding.compute":
            raise ValueError("Unknown embedding method.")
        path = Path(request["imagePath"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError("Image file is missing.")
        if path.stat().st_size > 100 * 1024 * 1024:
            raise ValueError("Image exceeds the 100 MiB input limit.")
        image_bytes = path.read_bytes()
        digest = sha256(image_bytes).hexdigest()
        if digest != request["contentHash"]:
            raise ValueError("Image changed since import; embedding was not saved.")
        with redirect_stdout(sys.stderr):
            vector = self.adapter.embed(image_bytes)
        if len(vector) != self.bundle["dimension"]:
            raise ValueError("Model returned an unexpected embedding dimension.")
        return {**metadata, "assetId": request["assetId"], "contentHash": digest, "vector": vector}
