from hashlib import sha256
import io
import json
import os

import pytest

import runner
from runner import ModelRegistry
from stub_adapters import StubClassifier


def classify_request(image_file, **overrides):
    request = {"method": "classify", "assetId": "asset-1", "imagePath": str(image_file),
               "contentHash": sha256(image_file.read_bytes()).hexdigest()}
    request.update(overrides)
    return request


def test_models_lists_registry_and_availability(ml_root):
    result = ModelRegistry(ml_root).dispatch({"method": "models"})
    assert result["defaultModel"] == "stub-a"
    assert {m["id"]: m["available"] for m in result["models"]} == {"stub-a": True, "stub-b": True, "missing": False}


def test_load_defaults_and_reports_checkpoint_hash(ml_root):
    result = ModelRegistry(ml_root).dispatch({"method": "load"})
    assert result == {"modelId": "stub-a", "modelVersion": sha256(b"checkpoint-a").hexdigest(), "labels": ["cat", "dog"]}


def test_unknown_model_and_method_are_rejected(ml_root):
    registry = ModelRegistry(ml_root)
    with pytest.raises(ValueError, match="Unknown model"):
        registry.dispatch({"method": "load", "modelId": "does-not-exist"})
    with pytest.raises(ValueError, match="Unknown method"):
        registry.dispatch({"method": "explode"})


def test_classify_returns_scores_for_matching_hash(ml_root, image_file):
    result = ModelRegistry(ml_root).dispatch(classify_request(image_file))
    assert result["assetId"] == "asset-1"
    assert result["contentHash"] == sha256(image_file.read_bytes()).hexdigest()
    assert [s["label"] for s in result["scores"]] == ["cat", "dog"]


def test_classify_rejects_changed_content(ml_root, image_file):
    with pytest.raises(ValueError, match="changed since import"):
        ModelRegistry(ml_root).dispatch(classify_request(image_file, contentHash="0" * 64))


@pytest.mark.parametrize("path", ["relative/image.png", None])
def test_classify_rejects_relative_or_missing_paths(ml_root, tmp_path, path):
    image_path = path or str(tmp_path / "absent.png")
    with pytest.raises(ValueError, match="missing"):
        ModelRegistry(ml_root).dispatch({"method": "classify", "assetId": "a", "imagePath": image_path, "contentHash": ""})


def test_only_selected_model_stays_resident(ml_root):
    registry = ModelRegistry(ml_root)
    registry.dispatch({"method": "load", "modelId": "stub-a"})
    registry.dispatch({"method": "load", "modelId": "stub-b"})
    assert len(registry.loaded) == 1
    assert next(iter(registry.loaded))[0] == "stub-b"


def test_checkpoint_change_produces_new_version_and_reload(ml_root):
    registry = ModelRegistry(ml_root)
    first = registry.dispatch({"method": "load"})["modelVersion"]
    before = StubClassifier.instances
    checkpoint = ml_root / "a.pth"
    checkpoint.write_bytes(b"retrained weights, different size")
    stat = checkpoint.stat()
    os.utime(checkpoint, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
    second = registry.dispatch({"method": "load"})["modelVersion"]
    assert second == sha256(b"retrained weights, different size").hexdigest() != first
    assert StubClassifier.instances == before + 1


def test_cached_version_reused_when_checkpoint_unchanged(ml_root):
    registry = ModelRegistry(ml_root)
    registry.dispatch({"method": "load"})
    before = StubClassifier.instances
    registry.dispatch({"method": "load"})
    assert StubClassifier.instances == before


def run_main(monkeypatch, capsys, ml_root, lines):
    monkeypatch.setattr(runner, "ModelRegistry", lambda: ModelRegistry(ml_root))
    monkeypatch.setattr("sys.stdin", io.StringIO("".join(line + "\n" for line in lines)))
    runner.main()
    captured = capsys.readouterr()
    return [json.loads(line) for line in captured.out.splitlines()], captured.err


def test_main_speaks_json_lines_and_keeps_stdout_clean(monkeypatch, capsys, ml_root, image_file):
    replies, stderr = run_main(monkeypatch, capsys, ml_root, [
        json.dumps({"id": 1, "method": "models"}),
        json.dumps({"id": 2, **classify_request(image_file)}),
        json.dumps({"id": 3, "method": "load", "modelId": "nope"}),
        "this is not json",
    ])
    assert [r["id"] for r in replies] == [1, 2, 3, None]
    assert [r["ok"] for r in replies] == [True, True, False, False]
    assert replies[1]["result"]["scores"][0]["label"] == "cat"
    assert "Unknown model" in replies[2]["error"]
    # Adapter prints are diverted to stderr so they never corrupt the protocol.
    assert "adapter chatter" in stderr and "prediction chatter" in stderr


def test_main_rejects_oversized_requests(monkeypatch, capsys, ml_root):
    replies, _ = run_main(monkeypatch, capsys, ml_root, ["x" * (1024 * 1024 + 1)])
    assert replies == [{"id": None, "ok": False, "error": "Request too large"}]


def test_dispatch_routes_embedding_and_tag_methods(ml_root):
    registry = ModelRegistry(ml_root)
    assert registry.dispatch({"method": "embedding.info"})["modelId"] == "stub/vision"
    assert registry.dispatch({"method": "tags.info"})["modelId"] == "stub/text"
