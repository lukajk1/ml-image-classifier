import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 32
EPOCHS_FROZEN = 5
EPOCHS_UNFROZEN = 10
LR_FROZEN = 1e-3
LR_UNFROZEN = 1e-4
VAL_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = os.path.join(DATA_DIR, "model.pth")

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_data():
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    print(f"Classes: {full_dataset.classes}")
    print(f"Total images: {len(full_dataset)}")

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    val_set.dataset = datasets.ImageFolder(DATA_DIR, transform=val_transforms)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, val_loader, full_dataset.classes


def build_model(num_classes):
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train_phase(model, train_loader, val_loader, epochs, lr, label):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    print(f"\n--- {label} ---")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch:02d}  train loss {train_loss:.4f}  acc {train_acc:.3f}  |  val loss {val_loss:.4f}  acc {val_acc:.3f}")


def main():
    train_loader, val_loader, classes = load_data()
    model = build_model(num_classes=len(classes))

    train_phase(model, train_loader, val_loader, EPOCHS_FROZEN, LR_FROZEN, "Phase 1: classifier only")

    for param in model.parameters():
        param.requires_grad = True
    train_phase(model, train_loader, val_loader, EPOCHS_UNFROZEN, LR_UNFROZEN, "Phase 2: full fine-tune")

    torch.save({"model_state": model.state_dict(), "classes": classes}, SAVE_PATH)
    print(f"\nSaved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
