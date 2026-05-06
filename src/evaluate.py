from src.model import UNet
from src.dataset import BrainTumorDataset
from src.train import train
import os
import torch    

def dice_coefficient(pred, target, smooth=1e-8):

    pred = torch.argmax(pred, dim=1)  # Get predicted class labels
    dice_scores = []
    for class_idx in range(1,4):  # Assuming class indices are 1, 2, and 3
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()

        intersection = (pred_class * target_class).sum()
        dice = (2. * intersection + smooth) / (pred_class.sum() + target_class.sum() + smooth)
        dice_scores.append(dice.item())

    return sum(dice_scores) / len(dice_scores) if dice_scores else 0.0

def evaluate(model, test_dataset, batch_size=16):
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dice_scores = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            dice_score = dice_coefficient(outputs, labels)
            dice_scores.append(dice_score)

    average_dice_score = sum(dice_scores) / len(dice_scores)
    return average_dice_score