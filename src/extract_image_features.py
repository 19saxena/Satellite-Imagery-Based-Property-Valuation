import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from image_dataset import PropertyImageDataset
import torchvision.models as models
device="cuda" if torch.cuda.is_available() else "cpu"
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])
resnet=models.resnet18(pretrained=True)
resnet.fc=torch.nn.Identity()
resnet=resnet.to(device)
resnet.eval()
def extract(df,img_dir,out_path):
    dataset=PropertyImageDataset(df,img_dir,transform)
    loader=DataLoader(dataset,batch_size=32,shuffle=False)
    features=[]
    with torch.no_grad():
        for imgs in loader:
            imgs=imgs.to(device)
            emb=resnet(imgs)
            features.append(emb.cpu())
    features=torch.cat(features).numpy()
    pd.DataFrame(features).to_csv(out_path,index=False)

