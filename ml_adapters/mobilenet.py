"""Adapter for the two existing MobileNet V3 Small checkpoints.

Adapter contract: constructor(checkpoint: Path, config: dict), labels: list[str],
predict(image_bytes: bytes) -> list[{label: str, confidence: float}].
"""
from io import BytesIO
import warnings
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, ImageOps


class MobileNetClassifier:
    def __init__(self, checkpoint, config):
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("architecture", "mobilenet_v3_small") != "mobilenet_v3_small":
            raise ValueError("This checkpoint requires a different model adapter.")
        self.labels = saved["classes"]
        if not self.labels or any(not isinstance(label, str) or not label.strip() for label in self.labels):
            raise ValueError("Checkpoint has invalid class labels.")
        self.model = models.mobilenet_v3_small(weights=None)  # Never download pretrained weights.
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, len(self.labels))
        self.model.load_state_dict(saved["model_state"], strict=True)
        self.model.eval()
        preprocessing = saved.get("preprocessing", {})
        self.exif_transpose = preprocessing.get("exif_transpose", False)
        self.transform = transforms.Compose([
            transforms.Resize(tuple(preprocessing.get("size", [224, 224]))),
            transforms.ToTensor(),
            transforms.Normalize(preprocessing.get("mean", [0.485, 0.456, 0.406]),
                                 preprocessing.get("std", [0.229, 0.224, 0.225])),
        ])

    def predict(self, image_bytes):
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as original:
                original.seek(0)
                image = ImageOps.exif_transpose(original) if self.exif_transpose else original
                tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0].tolist()
        return sorted([{"label": label, "confidence": float(probability)}
                       for label, probability in zip(self.labels, probabilities)],
                      key=lambda result: result["confidence"], reverse=True)
