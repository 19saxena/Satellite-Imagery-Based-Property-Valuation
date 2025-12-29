from PIL import Image
import torch
from torch.utils.data import Dataset
import os
class PropertyImageDataset(Dataset):
    def __init__(self,df,image_dir,transform=None):
        self.df=df.reset_index(drop=True)
        self.image_dir=image_dir
        self.transform=transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        pid=self.df.loc[idx, "id"]
        img_path=os.path.join(self.image_dir,f"{pid}.png")
        image=Image.open(img_path).convert("RGB")
        if self.transform:
            image=self.transform(image)
        return image
