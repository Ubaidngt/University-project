import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = 224
EPOCHS = 5
SEED = 42
BATCH_SIZE = 32


class FabricDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            return self.transform(image), int(self.labels[index])


def find_dataset_root(data_dir):
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    candidates = []
    for folder in [data_dir, *data_dir.rglob("*")]:
        if not folder.is_dir():
            continue
        class_folders = [
            child
            for child in folder.iterdir()
            if child.is_dir()
            and any(
                image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
                for image in child.rglob("*")
            )
        ]
        if len(class_folders) >= 2:
            candidates.append((len(class_folders), len(folder.parts), folder, class_folders))

    if not candidates:
        raise ValueError(f"No class folders containing images found in {data_dir}")

    candidates.sort(key=lambda item: (-item[0], item[1]))
    _, _, root, class_folders = candidates[0]
    return root, sorted(class_folders, key=lambda folder: folder.name.lower())


def load_images(class_folders):
    grouped = defaultdict(list)
    invalid = 0

    for class_id, class_folder in enumerate(class_folders):
        for path in sorted(class_folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                grouped[digest].append((class_id, str(path)))
            except (OSError, UnidentifiedImageError, ValueError):
                invalid += 1

    samples = []
    duplicates = 0
    conflicting_duplicates = 0
    for entries in grouped.values():
        if len({class_id for class_id, _ in entries}) > 1:
            conflicting_duplicates += len(entries)
            continue
        entries.sort(key=lambda entry: entry[1].lower())
        samples.append(entries[0])
        duplicates += len(entries) - 1

    samples.sort(key=lambda sample: sample[1].lower())
    print(
        f"Images: {len(samples)} usable, {invalid} unreadable removed, "
        f"{duplicates} exact duplicates removed, "
        f"{conflicting_duplicates} cross-class duplicates removed."
    )
    return samples, invalid, duplicates, conflicting_duplicates


def make_model(class_count, device, pretrained=True):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, class_count)
    model = model.to(device)
    return model


def train_epoch(model, loader, loss_function, optimizer, device):
    model.train()
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_items += labels.size(0)

    return total_loss / total_items, total_correct / total_items


def evaluate_loss(model, loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += loss_function(logits, labels).item() * labels.size(0)
            total_items += labels.size(0)

    return total_loss / total_items


def predict(model, loader, device):
    model.eval()
    predictions = []

    with torch.inference_mode():
        for images, _ in loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(1).cpu().tolist())

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default=str(Path(__file__).resolve().parent / "dataset")
    )
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).resolve().parent / "outputs")
    )
    parser.add_argument("--resume", help="Continue from a saved textile model")
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of epochs to train; when resuming, this is additional epochs",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be a positive integer.")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    root, class_folders = find_dataset_root(args.data_dir)
    class_names = [folder.name for folder in class_folders]
    samples, invalid, duplicates, conflicts = load_images(class_folders)
    labels = np.array([sample[0] for sample in samples], dtype=np.int64)
    paths = np.array([sample[1] for sample in samples])

    counts = np.bincount(labels, minlength=len(class_names))
    for name, count in zip(class_names, counts):
        print(f"{name}: {count} images")

    expected_classes = {
        "defect free",
        "hole",
        "horizontal",
        "vertical",
        "lines",
        "pinched fabric",
        "needle mark",
        "broken stitch",
        "stain",
    }
    missing = expected_classes - {name.lower() for name in class_names}
    if missing:
        print(f"Not found in this dataset folder: {', '.join(sorted(missing))}")

    if len(samples) == 0 or np.any(counts < 7):
        raise ValueError(
            "Each class needs at least 7 usable images for the 70/15/15 split."
        )

    train_paths, other_paths, train_labels, other_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    valid_paths, test_paths, valid_labels, test_labels = train_test_split(
        other_paths,
        other_labels,
        test_size=0.50,
        random_state=SEED,
        stratify=other_labels,
    )

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(18),
            transforms.ColorJitter(contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_loader = DataLoader(
        FabricDataset(train_paths, train_labels, train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        FabricDataset(valid_paths, valid_labels, eval_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        FabricDataset(test_paths, test_labels, eval_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    class_weights = len(train_labels) / (len(class_names) * np.bincount(train_labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    checkpoint = None
    if args.resume:
        checkpoint_path = Path(args.resume).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        if checkpoint.get("classes") != class_names:
            raise ValueError(
                "Checkpoint classes do not match the dataset class folders."
            )

    model = make_model(len(class_names), device, pretrained=checkpoint is None)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["state_dict"])

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_weights = output_dir / "best_model.pth"
    print(
        f"Training on {len(train_paths)} images, "
        f"validating on {len(valid_paths)}."
    )

    if checkpoint is None:
        best_validation_loss = float("inf")
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
        frozen_epochs = min(2, args.epochs)

        for epoch in range(frozen_epochs):
            train_loss, train_accuracy = train_epoch(
                model, train_loader, loss_function, optimizer, device
            )
            valid_loss = evaluate_loss(model, valid_loader, loss_function, device)
            print(
                f"Epoch {epoch + 1}/{args.epochs} "
                f"train_loss={train_loss:.4f} "
                f"train_accuracy={train_accuracy:.4f} "
                f"val_loss={valid_loss:.4f}"
            )
            if valid_loss < best_validation_loss:
                best_validation_loss = valid_loss
                torch.save(model.state_dict(), best_weights)

        for parameter in model.features.parameters():
            parameter.requires_grad = True
        for parameter in model.features[:-3].parameters():
            parameter.requires_grad = False
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-5,
        )
        fine_tune_start = frozen_epochs
    else:
        for parameter in model.features.parameters():
            parameter.requires_grad = True
        for parameter in model.features[:-3].parameters():
            parameter.requires_grad = False
        valid_loss = evaluate_loss(model, valid_loader, loss_function, device)
        best_validation_loss = valid_loss
        torch.save(model.state_dict(), best_weights)
        print(f"Checkpoint validation loss: {valid_loss:.4f}")
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-5,
        )
        fine_tune_start = 0

    for epoch in range(fine_tune_start, args.epochs):
        train_loss, train_accuracy = train_epoch(
            model, train_loader, loss_function, optimizer, device
        )
        valid_loss = evaluate_loss(model, valid_loader, loss_function, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"train_accuracy={train_accuracy:.4f} "
            f"val_loss={valid_loss:.4f}"
        )
        if valid_loss < best_validation_loss:
            best_validation_loss = valid_loss
            torch.save(model.state_dict(), best_weights)

    model.load_state_dict(torch.load(best_weights, map_location=device))
    predictions = predict(model, test_loader, device)
    metrics = {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "precision_macro": float(
            precision_score(test_labels, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(test_labels, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(test_labels, predictions, average="macro", zero_division=0)
        ),
        "classification_report": classification_report(
            test_labels,
            predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            test_labels, predictions, labels=list(range(len(class_names)))
        ).tolist(),
    }

    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": class_names,
            "image_size": IMAGE_SIZE,
        },
        output_dir / "textile_defect_model.pth",
    )
    (output_dir / "results.json").write_text(
        json.dumps(
            {
                "dataset_root": str(root),
                "classes": class_names,
                "split_images": {
                    "train": len(train_paths),
                    "validation": len(valid_paths),
                    "test": len(test_paths),
                },
                "removed_unreadable": invalid,
                "removed_exact_duplicates": duplicates,
                "removed_cross_class_duplicates": conflicts,
                "epochs": args.epochs,
                "test_metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved model and results to {output_dir}")


if __name__ == "__main__":
    main()
