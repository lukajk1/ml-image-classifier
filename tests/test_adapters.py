"""Adapter tests with tiny randomly initialised models; no pretrained weights are read or downloaded."""
from io import BytesIO
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def save_mobilenet_checkpoint(path, classes, **extra):
    from torch import nn
    from torchvision import models
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    torch.save({"model_state": model.state_dict(), "classes": classes, **extra}, path)
    return path


def jpeg_with_rotation_tag(width=40, height=20):
    from PIL import Image
    image = Image.new("RGB", (width, height), (10, 120, 240))
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 degrees.
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def test_mobilenet_predicts_sorted_probabilities(tmp_path, png_bytes):
    from ml_adapters.mobilenet import MobileNetClassifier
    checkpoint = save_mobilenet_checkpoint(tmp_path / "m.pth", ["artwork", "photos"],
                                           preprocessing={"size": [64, 64], "exif_transpose": True})
    classifier = MobileNetClassifier(checkpoint, {})
    scores = classifier.predict(png_bytes)
    assert classifier.labels == ["artwork", "photos"]
    assert sorted(s["label"] for s in scores) == ["artwork", "photos"]
    assert scores[0]["confidence"] >= scores[1]["confidence"]
    assert math.isclose(sum(s["confidence"] for s in scores), 1.0, rel_tol=1e-5)
    assert len(classifier.predict(jpeg_with_rotation_tag())) == 2


def test_mobilenet_preprocessing_shape(tmp_path, png_bytes):
    from PIL import Image
    from ml_adapters.mobilenet import MobileNetClassifier
    checkpoint = save_mobilenet_checkpoint(tmp_path / "m.pth", ["a", "b", "c"], preprocessing={"size": [48, 56]})
    classifier = MobileNetClassifier(checkpoint, {})
    tensor = classifier.transform(Image.open(BytesIO(png_bytes)).convert("RGB"))
    assert tuple(tensor.shape) == (3, 48, 56)


def test_mobilenet_rejects_other_architectures_and_bad_labels(tmp_path):
    from ml_adapters.mobilenet import MobileNetClassifier
    with pytest.raises(ValueError, match="different model adapter"):
        MobileNetClassifier(save_mobilenet_checkpoint(tmp_path / "r.pth", ["a"], architecture="resnet18"), {})
    with pytest.raises(ValueError, match="invalid class labels"):
        MobileNetClassifier(save_mobilenet_checkpoint(tmp_path / "l.pth", ["a", " "]), {})


def test_siglip_embedder_returns_unit_vector_of_configured_size(tmp_path, png_bytes):
    pytest.importorskip("transformers")
    from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel
    from ml_adapters.siglip import SiglipEmbedder
    config = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                num_attention_heads=2, image_size=32, patch_size=16)
    torch.manual_seed(0)
    SiglipVisionModel(config).save_pretrained(tmp_path, safe_serialization=True)
    SiglipImageProcessor(size={"height": 32, "width": 32}).save_pretrained(tmp_path)
    embedder = SiglipEmbedder(tmp_path)
    vector = embedder.embed(png_bytes)
    assert len(vector) == 16
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-4)
    from PIL import Image
    inputs = embedder.processor(images=Image.open(BytesIO(png_bytes)).convert("RGB"), return_tensors="pt")
    assert tuple(inputs["pixel_values"].shape) == (1, 3, 32, 32)


class FakeTokenizer:
    def __call__(self, prompts, **kwargs):
        self.prompts = prompts
        return {"prompts": prompts}


class FakeTextModel:
    """Maps each prompt to a fixed direction so ranking is predictable."""
    def __init__(self, directions):
        self.directions = directions
        self.calls = 0

    def __call__(self, prompts):
        self.calls += 1
        rows = [self.directions[p.removeprefix("a photo of ").removesuffix(".")] for p in prompts]
        return type("Output", (), {"pooler_output": torch.tensor(rows, dtype=torch.float32)})()


def make_suggester(directions):
    pytest.importorskip("transformers")
    from collections import OrderedDict
    from ml_adapters.siglip_tags import SiglipTagSuggester
    suggester = SiglipTagSuggester.__new__(SiglipTagSuggester)
    suggester.model = FakeTextModel(directions)
    suggester.tokenizer = FakeTokenizer()
    suggester.bundle = {"dimension": 2, "prompt": "a photo of {}."}
    suggester.spec, suggester.root, suggester.vision = {}, None, None
    suggester.cache = OrderedDict()
    return suggester


def test_tag_suggester_ranks_by_cosine_and_caches_text(png_bytes):
    suggester = make_suggester({"sky": [1, 0], "cat": [0, 1], "dusk": [1, 1]})
    scores = suggester.suggest(png_bytes, ["cat", "sky", "dusk"], [2.0, 0.0])
    assert [s["label"] for s in scores] == ["sky", "dusk", "cat"]
    assert math.isclose(scores[0]["score"], 1.0, rel_tol=1e-5)
    suggester.suggest(png_bytes, ["cat", "sky"], [0.0, 1.0])
    assert suggester.model.calls == 1  # Text vectors are reused.


def test_tag_suggester_returns_at_most_twelve(png_bytes):
    labels = [f"t{i}" for i in range(20)]
    suggester = make_suggester({label: [1, i / 20] for i, label in enumerate(labels)})
    assert len(suggester.suggest(png_bytes, labels, [1.0, 0.0])) == 12


@pytest.mark.parametrize("vector", [[0.0, 0.0], [1.0], [float("nan"), 1.0]])
def test_tag_suggester_rejects_invalid_vectors(png_bytes, vector):
    suggester = make_suggester({"sky": [1, 0]})
    with pytest.raises(ValueError, match="Invalid image embedding"):
        suggester.suggest(png_bytes, ["sky"], vector)
