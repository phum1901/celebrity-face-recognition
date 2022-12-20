from pathlib import Path
import os

import torch
from torchvision import transforms as T
import pytorch_lightning as pl

from PIL import Image

from typing import Callable, Union, Optional

class DataSet(torch.utils.data.Dataset):
    def __init__(self, data_dir: Union[str, Path], mapping, transform: Callable = None):
        super().__init__()
        if not isinstance(data_dir, Path):
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = data_dir
        self.transform = transform 
        self.filenames = list(self.data_dir.glob('*/*'))
        self.mapping = mapping
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        img = Image.open(str(self.filenames[index]))
        label = self.filenames[index].parent.stem   
        label = self.mapping[label]
        if self.transform:
            img = self.transform(img)
        
        return img, label 

class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, mapping, batch_size: int = 32, remove_stranger: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size 
        self.mapping = mapping
        self.remove_stranger = remove_stranger

        self.transforms = T.Compose([
            T.Resize((160, 160)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transforms_train = T.Compose([
            T.Resize((160, 160)),
            T.RandomHorizontalFlip(p=0.5),
            T.TrivialAugmentWide(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            if self.remove_stranger:
                self.data_train = DataSet(os.path.join(self.data_dir, 'train_new'), self.mapping, self.transforms_train)
            else:
                self.data_train = DataSet(os.path.join(self.data_dir, 'train'), self.mapping, self.transforms_train)
            self.data_val = DataSet(os.path.join(self.data_dir, 'val'), self.mapping, self.transforms)
        if stage == 'test' or stage is None:
            self.data_test = DataSet(os.path.join(self.data_dir, 'test'), self.mapping, self.transforms)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data_train, 
            shuffle=True,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data_val, 
            shuffle=False,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data_test, 
            shuffle=False,
            batch_size=self.batch_size,
        )