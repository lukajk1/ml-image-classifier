import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Small_Weights
import torch.nn as nn
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pth")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    classes = checkpoint["classes"]
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE).eval()
    return model, classes


def classify_image(model, classes, image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    top_class = classes[probs.argmax().item()]
    confidence = probs.max().item()
    scores = {cls: f"{probs[i].item():.2%}" for i, cls in enumerate(classes)}
    return top_class, confidence, scores


def classify_folder(folder_path):
    model, classes = load_model()
    files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]
    if not files:
        print(f"No images found in {folder_path}")
        return

    print(f"\nClassifying {len(files)} images in: {folder_path}\n")
    print(f"{'File':<40} {'Prediction':<16} {'Confidence':<12} Scores")
    print("-" * 90)

    for filename in sorted(files):
        path = os.path.join(folder_path, filename)
        try:
            top_class, confidence, scores = classify_image(model, classes, path)
            score_str = "  ".join(f"{k}: {v}" for k, v in scores.items())
            print(f"{filename:<40} {top_class:<16} {confidence:.2%}        {score_str}")
        except Exception as e:
            print(f"{filename:<40} ERROR: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py <folder_path>")
        sys.exit(1)
    classify_folder(sys.argv[1])
