"""Persistent JSON-lines classifier worker. stdin/stdout only; no network listener.

Adapters return label/confidence scores and never touch the library catalog.
Stdout is reserved for protocol replies; diagnostics go to stderr.
"""
from contextlib import redirect_stdout
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
# Isolated bundled Python does not automatically add the script directory.
# Resolve only our own shipped adapters, independently of the working directory.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from runtime_paths import asset_root


class ModelRegistry:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.assets = asset_root(self.root)
        self.config = json.loads((self.root / "models.json").read_text(encoding="utf-8"))
        self.models = {spec["id"]: spec for spec in self.config["models"]}
        self.loaded = {}
        self.versions = {}
        self.embedding = None
        self.tag_suggester = None

    def get(self, model_id):
        model_id = model_id or self.config["defaultModel"]
        if model_id not in self.models:
            raise ValueError(f"Unknown model: {model_id}")
        spec = self.models[model_id]
        checkpoint = (self.assets / spec["checkpoint"]).resolve()
        # Content identity changes whenever the weights change, even under the same name.
        stat = checkpoint.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self.versions.get(checkpoint)
        if cached is None or cached[0] != signature:
            digest = sha256()
            with checkpoint.open('rb') as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(chunk)
            cached = (signature, digest.hexdigest())
            self.versions[checkpoint] = cached
        version = cached[1]
        key = (model_id, version)
        if key not in self.loaded:
            module, factory = spec["adapter"].split(":", 1)
            with redirect_stdout(sys.stderr):
                adapter = getattr(import_module(module), factory)(checkpoint, spec)
            self.loaded.clear()  # Keep only the selected model resident.
            self.loaded[key] = adapter
        return model_id, version, self.loaded[key]

    def dispatch(self, request):
        method = request["method"]
        if method.startswith("tags."):
            if self.tag_suggester is None:
                from tag_worker import TagWorker
                self.tag_suggester = TagWorker(self.root)
            return self.tag_suggester.dispatch(request)
        if method.startswith("embedding."):
            if self.embedding is None:
                from embedding_worker import EmbeddingWorker
                self.embedding = EmbeddingWorker(self.root)
            return self.embedding.dispatch(request)
        if method == "models":
            return {"defaultModel": self.config["defaultModel"], "models": [
                {"id": spec["id"], "name": spec["name"],
                 "available": (self.assets / spec["checkpoint"]).is_file()}
                for spec in self.models.values()
            ]}
        if method not in ("load", "classify"):
            raise ValueError(f"Unknown method: {method}")
        model_id, version, adapter = self.get(request.get("modelId"))
        metadata = {"modelId": model_id, "modelVersion": version, "labels": adapter.labels}
        if method == "load":
            return metadata
        path = Path(request["imagePath"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError("Image file is missing or its path is not absolute.")
        if path.stat().st_size > 100 * 1024 * 1024:
            raise ValueError("Image exceeds the classifier's 100 MiB input limit.")
        image_bytes = path.read_bytes()
        content_hash = sha256(image_bytes).hexdigest()
        if content_hash != request["contentHash"]:
            raise ValueError("Image content changed since import. Classification was not saved.")
        with redirect_stdout(sys.stderr):
            scores = adapter.predict(image_bytes)
        return {**metadata, "assetId": request["assetId"], "contentHash": content_hash, "scores": scores}


def main():
    registry = ModelRegistry()
    for line in sys.stdin:
        request = {}
        try:
            if len(line) > 1024 * 1024:
                raise ValueError("Request too large")
            request = json.loads(line)
            result = registry.dispatch(request)
            reply = {"id": request["id"], "ok": True, "result": result}
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            reply = {"id": request.get("id"), "ok": False, "error": str(error)}
        print(json.dumps(reply, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
