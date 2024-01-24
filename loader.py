import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import albumentations as A

class MRISegmentationDataset(Dataset):
    def __init__(self,df,transform=None):
        super(MRISegmentationDataset,self).__init__()
        self.image_path=df['image_path'].to_list()
        self.mask_path=df['mask_path'].to_list()
        self.transform=transform
        
    def load_transform(self,index):
        image_path=self.image_path[index]
        mask_path=self.mask_path[index]
        
        image,mask=Image.open(image_path),Image.open(mask_path)
        
        image=np.array(image).astype(np.float32)/255
        mask=np.array(mask).astype(np.float32)/255
        
        return image,mask
    
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self,index):
        image,mask=self.load_transform(index)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            return transformed['image'], transformed['mask'].unsqueeze_(0)
        else:
            transformed = ToTensorV2()(image=image, mask=mask)    
            return transformed['image'], transformed['mask'].unsqueeze_(0)
        
if __name__ == '__main__':
   train_transforms = A.Compose([
    A.Resize(224,224, p=1.0),
    A.RandomBrightnessContrast(p=0.2),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    ToTensorV2(),
    ])

   train_df=pd.read_csv('train_data.csv')
   train_ds=MRISegmentationDataset(train_df,train_transforms)
   train_loader=DataLoader(
       train_ds,batch_size=32,num_workers=4,shuffle=True
   )
   print(train_loader)
   