import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np


class BrainTumorDataset(Dataset):
    def __init__(self,hastalar, transform=None):
        self.ornekler = []
        self.hastalar = hastalar
        self.transform = transform

        start_slice=15
        end_slice=140
        slice_count = 10
        
        secilecek_indeksler = np.linspace(start_slice, end_slice, slice_count, dtype=int) # 15 ile 140 arasında 10 dilim seçmek için
        for hasta in self.hastalar:
            for dilim_index in secilecek_indeksler:
                self.ornekler.append((hasta, dilim_index))
        
    def __len__(self):
        return len(self.ornekler)

    def __getitem__(self, idx):
        hastalar, dilim_index = self.ornekler[idx]
        try:
            # Load the image and label using nibabel
            image = torch.stack([torch.from_numpy(nib.load(hastalar['flair']).get_fdata()),
                                torch.from_numpy(nib.load(hastalar['t1']).get_fdata()),
                                    torch.from_numpy(nib.load(hastalar['t1ce']).get_fdata()),
                                    torch.from_numpy(nib.load(hastalar['t2']).get_fdata())], dim=0)
            label = torch.from_numpy(nib.load(hastalar['seg']).get_fdata())
            label[label ==4 ] = 3  # Map label 4 to 3
            image = image[:, :, :, dilim_index]  # (4, 240, 240,    155)
            label = label[:, :, dilim_index] # (240, 240, 155)
        except Exception as e:
            print(f"Error loading data for index {idx}: {e}")
            return None, None  # Return None for both image and label if there's an error

        # Normalize the image to [0, 1]
        image = (image - image.min() + 1e-8) / (image.max() - image.min() + 1e-8)

        # Convert to torch tensors
        image = image.float()  # Add channel dimension
        label = label.long()  # Assuming labels are integers

        if self.transform:
            image, label = self.transform(image, label)

        return image, label