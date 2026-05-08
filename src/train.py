import torch
import torch.nn as nn
import torch.optim as optim
import os

from src.model import UNet

def train(train_dataset, val_dataset, model, num_epochs=10,num_workers=0 ,batch_size=16 ,learning_rate=0.001, save_path=r"D:\Projects\medical-image-segmentation\models\best_model.pth"):
    best_val_loss = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 5.0, 3.0, 4.9]).to(device))  # Adjust weights for class imbalance
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size ,shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    os.makedirs("models", exist_ok=True)
    for epoch in range(num_epochs):
        model.train()

        epoch_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx) % 10 == 0:
                print(f"Epoch: {epoch+1}/{num_epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
        
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels).item()

            average_val_loss = val_loss / len(val_loader)
            print(f"Epoch: {epoch+1}/{num_epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Val Loss: {average_val_loss:.4f}")

        average_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch: {epoch+1}/{num_epochs} | Average Loss: {average_epoch_loss:.4f}")
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            torch.save(model.state_dict(), save_path)

    return model