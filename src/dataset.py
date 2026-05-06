import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np


class BrainTumorDataset(Dataset):
    def __init__(self,hastalar, transform=None):
        self.hastalar = hastalar
        self.transform = transform

    def __len__(self):
        return len(self.hastalar)

    def __getitem__(self, idx):
        hastalar = self.hastalar[idx]
        try:
            # Load the image and label using nibabel
            image = torch.stack([torch.from_numpy(nib.load(hastalar['flair']).get_fdata()),
                                torch.from_numpy(nib.load(hastalar['t1']).get_fdata()),
                                    torch.from_numpy(nib.load(hastalar['t1ce']).get_fdata()),
                                    torch.from_numpy(nib.load(hastalar['t2']).get_fdata())], dim=0)
            label = torch.from_numpy(nib.load(hastalar['seg']).get_fdata())
            label[label ==4 ] = 3  # Map label 4 to 3
            dilim_index = np.random.randint(0, 155)
            image = image[:, :, :, dilim_index]  # (4, 240, 240)
            label = label[:, :, dilim_index]  # (240, 240) 
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