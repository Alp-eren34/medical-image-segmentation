import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

class BrainTumorDataset(Dataset):
    def __init__(self, hastalar, transform=None):
        self.ornekler = []
        self.hastalar = hastalar
        self.transform = transform

        start_slice = 15
        end_slice = 140
        slice_count = 10
        
        # 15 ile 140 arasında 10 dilim seçmek için
        secilecek_indeksler = np.linspace(start_slice, end_slice, slice_count, dtype=int) 
        for hasta in self.hastalar:
            for dilim_index in secilecek_indeksler:
                self.ornekler.append((hasta, dilim_index))
        
    def __len__(self):
        return len(self.ornekler)

    def __getitem__(self, idx):
        hastalar, dilim_index = self.ornekler[idx]
        try:
            # YENİ SİSTEM: get_fdata() yerine dataobj kullanarak
            # TÜM 3D veriyi değil, SADECE istediğimiz 2D kesiti RAM'e alıyoruz.
            
            # np.array() kullanmamızın sebebi dataobj'nin bir "proxy" olmasıdır, 
            # numpy array'e çevirerek veriyi belleğe kesin olarak alıyoruz.
            flair_slice = np.array(nib.load(hastalar['flair']).dataobj[:, :, dilim_index])
            t1_slice    = np.array(nib.load(hastalar['t1']).dataobj[:, :, dilim_index])
            t1ce_slice  = np.array(nib.load(hastalar['t1ce']).dataobj[:, :, dilim_index])
            t2_slice    = np.array(nib.load(hastalar['t2']).dataobj[:, :, dilim_index])
            seg_slice   = np.array(nib.load(hastalar['seg']).dataobj[:, :, dilim_index])

            # 4 kanallı (Modality) görüntüyü birleştiriyoruz (C, H, W)
            image = torch.stack([
                torch.from_numpy(flair_slice),
                torch.from_numpy(t1_slice),
                torch.from_numpy(t1ce_slice),
                torch.from_numpy(t2_slice)
            ], dim=0)

            # Maske (Label) işlemleri
            label = torch.from_numpy(seg_slice)
            label[label == 4] = 3  # Map label 4 to 3

        except Exception as e:
            print(f"Error loading data for index {idx}: {e}")
            return None, None 

        # Normalizasyon: 0-1 arasına çekme
        image = (image - image.min() + 1e-8) / (image.max() - image.min() + 1e-8)

        # Float ve Long tip dönüşümleri
        image = image.float() 
        label = label.long()

        if self.transform:
            image, label = self.transform(image, label)

        return image, label