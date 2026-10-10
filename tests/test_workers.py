from hashlib import sha256
import json

import pytest

from conftest import write_json
from embedding_worker import EmbeddingWorker
from stub_adapters import StubEmbedder, StubTagSuggester
from tag_worker import TagWorker, verify_bundle


def compute(worker, method, image_file, **extra):
    return worker.dispatch({"method": method, "assetId": "a1", "imagePath": str(image_file),
                            "contentHash": sha256(image_file.read_bytes()).hexdigest(), **extra})


# Embedding worker -----------------------------------------------------------

def test_embedding_info_does_not_load_or_verify_weights(ml_root):
    (ml_root / "models" / "vision" / "model.safetensors").unlink()
    info = EmbeddingWorker(ml_root).dispatch({"method": "embedding.info"})
    assert info["dimension"] == 4 and info["revision"] == "r1" and len(info["key"]) == 64


def test_embedding_key_tracks_bundle_and_adapter(ml_root):
    bundle = json.loads((ml_root / "models" / "vision" / "bundle.json").read_text(encoding="utf-8"))
    expected = sha256(json.dumps({"bundle": bundle, "adapter": "stub_adapters:StubEmbedder"}, sort_keys=True).encode()).hexdigest()
    assert EmbeddingWorker(ml_root).key == expected


def test_embedding_compute_returns_vector(ml_root, image_file):
    result = compute(EmbeddingWorker(ml_root), "embedding.compute", image_file)
    assert result["vector"] == [0.5] * 4 and result["assetId"] == "a1"


def test_embedding_rejects_damaged_weights(ml_root):
    (ml_root / "models" / "vision" / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="damaged: model.safetensors"):
        EmbeddingWorker(ml_root).dispatch({"method": "embedding.load"})


def test_embedding_rejects_wrong_dimension(ml_root, image_file, monkeypatch):
    monkeypatch.setattr(StubEmbedder, "dimension", 3)
    with pytest.raises(ValueError, match="unexpected embedding dimension"):
        compute(EmbeddingWorker(ml_root), "embedding.compute", image_file)


def test_embedding_rejects_changed_image(ml_root, image_file):
    worker = EmbeddingWorker(ml_root)
    with pytest.raises(ValueError, match="changed since import"):
        worker.dispatch({"method": "embedding.compute", "assetId": "a", "imagePath": str(image_file), "contentHash": "bad"})


def test_embedding_missing_bundle_has_actionable_error(ml_root):
    (ml_root / "models" / "vision" / "bundle.json").unlink()
    with pytest.raises(ValueError, match="No runtime download"):
        EmbeddingWorker(ml_root)


# Tag worker -----------------------------------------------------------------

def test_verify_bundle_rejects_path_escape(tmp_path):
    directory = tmp_path / "bundle"
    directory.mkdir()
    (tmp_path / "secret.txt").write_text("x")
    with pytest.raises(ValueError, match="Invalid bundled tag model path"):
        verify_bundle(directory, {"files": {"../secret.txt": "0"}})


def test_tags_info_and_damage_detection(ml_root):
    worker = TagWorker(ml_root)
    assert worker.dispatch({"method": "tags.info"})["compatibleEmbeddingKey"] == "vision-key"
    (ml_root / "models" / "text" / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Damaged bundled model file"):
        worker.dispatch({"method": "tags.load"})


def test_tag_candidates_are_trimmed_deduplicated_and_user_first(ml_root, image_file):
    compute(TagWorker(ml_root), "tags.suggest", image_file,
            candidates=["  cat ", "CAT", 7, "", "x" * 121, "Sunset"])
    assert StubTagSuggester.last_call["candidates"] == ["cat", "Sunset", "sky", "tree"]


def test_tag_defaults_can_be_excluded_and_list_is_capped(ml_root, image_file):
    compute(TagWorker(ml_root), "tags.suggest", image_file,
            candidates=[f"label {i}" for i in range(600)], includeDefaults=False)
    candidates = StubTagSuggester.last_call["candidates"]
    assert len(candidates) == 512 and candidates[-1] == "label 511"


def test_cached_vector_only_reused_with_matching_embedding_key(ml_root, image_file):
    worker = TagWorker(ml_root)
    compute(worker, "tags.suggest", image_file, vector=[1, 0, 0, 0], embeddingKey="other-key")
    assert StubTagSuggester.last_call["vector"] is None
    compute(worker, "tags.suggest", image_file, vector=[1, 0, 0, 0], embeddingKey="vision-key")
    assert StubTagSuggester.last_call["vector"] == [1, 0, 0, 0]


def test_tag_suggest_rejects_changed_image(ml_root, image_file):
    with pytest.raises(ValueError, match="changed since import"):
        TagWorker(ml_root).dispatch({"method": "tags.suggest", "assetId": "a", "imagePath": str(image_file), "contentHash": "bad"})


def test_embed_text_returns_query_vector_with_model_metadata(ml_root):
    result = TagWorker(ml_root).dispatch({"method": "tags.embedText", "text": "  blue city at night "})
    assert result["vector"] == [0.5, 0.5, 0.5, 0.5]
    assert result["text"] == "blue city at night"
    assert result["compatibleEmbeddingKey"] == "vision-key"
    assert StubTagSuggester.last_call == {"text": "blue city at night"}


@pytest.mark.parametrize("text", ["", "   ", None, "x" * 301])
def test_embed_text_rejects_empty_or_overlong_queries(ml_root, text):
    with pytest.raises(ValueError):
        TagWorker(ml_root).dispatch({"method": "tags.embedText", "text": text})
