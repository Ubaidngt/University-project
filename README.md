# Textile defect baseline

This separate project runs locally with PyTorch. It trains a MobileNetV2 image classifier using a stratified 70/15/15 split, image cleanup, augmentation, class weights, validation, and a held-out test set.

The proposal describes 3,077 images. The dataset currently has 3,050 usable images in nine class folders. The dataset has defect labels only, so the model does not train root-cause prediction.

## Run on Windows

Install the remaining dependencies:

```powershell
cd "E:\Python\5th-sem\moaz"
python -m pip install -r requirements.txt
```

Check that PyTorch can see the GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Start training:

```powershell
python train.py
```

The dataset is read from `dataset` by default. The trained model and test metrics are saved in `outputs`.

## Continue training

To continue from the saved model for five more epochs, run:

```powershell
python train.py --resume "outputs\textile_defect_model.pth" --epochs 5
```

This continues fine-tuning from the saved weights, using a fresh optimizer and the same training/validation split. The updated model and metrics replace the files in `outputs`.

## Predict an image

After training finishes, run:

```powershell
python predict.py --image "dataset\hole\1.jpg"
```

Replace the image path with any `.jpg`, `.jpeg`, or `.png` image. The script prints the three most likely classes and their confidence scores. The training run already tests the saved model on its held-out test split and saves those metrics in `outputs\results.json`.
