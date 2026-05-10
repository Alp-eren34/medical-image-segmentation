# src/dataset.py dosyasının YENİ hali

import torch
from torch.utils.data import Dataset

class FastBrainTumorDataset(Dataset):
    def __init__(self, dosya_yollari, transform=None):
        self.dosya_yollari = dosya_yollari
        self.transform = transform

    def __len__(self):
        return len(self.dosya_yollari)

    def __getitem__(self, idx):
        # NIfTI dosyası okumak yok! Hazır PyTorch tensörünü RAM'e ışınlıyoruz.
        image, label = torch.load(self.dosya_yollari[idx], weights_only=True)

        if self.transform:
            image, label = self.transform(image, label)

        return image, label