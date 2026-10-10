"""Consistency checks on the committed configuration and model manifests (weights not required)."""
from hashlib import sha256
from importlib.util import find_spec
import json
import warnings

from conftest import REPO
from embedding_worker import EmbeddingWorker
from runner import ModelRegistry


def load(name):
    return json.loads((REPO / name).read_text(encoding="utf-8"))


def test_registry_entries_are_well_formed():
    config = load("models.json")
    ids = [spec["id"] for spec in config["models"]]
    assert len(ids) == len(set(ids)) and config["defaultModel"] in ids
    for spec in config["models"]:
        module, _, factory = spec["adapter"].partition(":")
        assert factory and find_spec(module) is not None, spec["adapter"]
        assert spec["checkpoint"].endswith(".pth") and "/" not in spec["checkpoint"]


def test_runtime_requirements_shape():
    requirement = load("runtime-requirements.json")
    assert requirement["format"] == 1
    assert requirement["runtimeId"].replace("-", "").replace("_", "").isalnum()


def test_models_method_works_without_weights():
    result = ModelRegistry(REPO).dispatch({"method": "models"})
    assert {m["id"] for m in result["models"]} == {s["id"] for s in load("models.json")["models"]}


def test_text_bundle_matches_vision_bundle():
    embedding, tagging = load("embedding.json"), load("tagging.json")
    vision = json.loads((REPO / embedding["directory"] / "bundle.json").read_text(encoding="utf-8"))
    text = json.loads((REPO / tagging["directory"] / "bundle.json").read_text(encoding="utf-8"))
    assert tagging["visionDirectory"] == embedding["directory"]
    assert text["revision"] == vision["revision"] and text["modelId"] == vision["modelId"]
    key = sha256(json.dumps({"bundle": vision, "adapter": "ml_adapters.siglip:SiglipEmbedder"}, sort_keys=True).encode()).hexdigest()
    assert text["compatibleEmbeddingKey"] == key == EmbeddingWorker(REPO).key
    for bundle in (vision, text):
        assert "model.safetensors" in bundle["files"]
        assert all(len(digest) == 64 for digest in bundle["files"].values())


def test_committed_small_bundle_files_match_their_hashes():
    # Weights (*.safetensors) are gitignored; every other hashed file is committed and must match.
    # Known issue: the manifests were hashed from CRLF files written on Windows, while git stores
    # them normalised to LF. They only verify after a CRLF checkout (core.autocrlf=true, the
    # Windows default, which the release workflow sets explicitly). Accept either form here, but
    # flag LF-only matches so the problem stays visible until a .gitattributes fix lands.
    crlf_only = []
    for config in ("embedding.json", "tagging.json"):
        directory = REPO / load(config)["directory"]
        bundle = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
        for name, digest in bundle["files"].items():
            if name.endswith(".safetensors"):
                continue
            data = (directory / name).read_bytes()
            if sha256(data).hexdigest() == digest:
                continue
            windows = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            assert sha256(windows).hexdigest() == digest, f"{name} does not match bundle.json"
            crlf_only.append(name)
    if crlf_only:
        warnings.warn("Bundle hashes only match with CRLF line endings: " + ", ".join(crlf_only))


def test_tag_vocabulary_is_clean():
    vocabulary = load(load("tagging.json")["vocabulary"])
    assert vocabulary and all(isinstance(v, str) and v.strip() == v and 0 < len(v) <= 120 for v in vocabulary)
    assert len({v.casefold() for v in vocabulary}) == len(vocabulary)
