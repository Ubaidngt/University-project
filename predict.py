import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


IMAGE_SIZE = 224


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--model",
        default=str(
            Path(__file__).resolve().parent
            / "outputs"
            / "textile_defect_model.pth"
        ),
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    class_names = checkpoint["classes"]
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    image_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    with Image.open(image_path) as image:
        image = image_transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities = torch.softmax(model(image), dim=1)[0]
    top_values, top_indices = probabilities.topk(min(3, len(class_names)))

    print(f"Device: {device}")
    print(f"Image: {image_path}")
    print("Predictions:")
    for probability, class_id in zip(top_values.tolist(), top_indices.tolist()):
        print(f"{class_names[class_id]}: {probability:.1%}")


if __name__ == "__main__":
    main()
