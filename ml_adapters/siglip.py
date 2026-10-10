"""Offline, vision-only SigLIP embedding adapter. No library writes."""
from io import BytesIO
from PIL import Image, ImageOps
import torch
from transformers import AutoImageProcessor, SiglipVisionModel


class SiglipEmbedder:
    def __init__(self, directory):
        torch.set_num_threads(2)
        self.processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True, use_fast=False)
        self.model = SiglipVisionModel.from_pretrained(directory, local_files_only=True).eval()

    def embed(self, image_bytes):
        with Image.open(BytesIO(image_bytes)) as original:
            if original.width * original.height > 80_000_000:
                raise ValueError("Image exceeds the embedding model's 80 megapixel limit.")
            image = ImageOps.exif_transpose(original).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            vector = self.model(**inputs).pooler_output[0].float()
            vector = torch.nn.functional.normalize(vector, dim=0)
        return vector.tolist()
