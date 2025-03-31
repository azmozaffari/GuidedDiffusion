import torch
from models.diffusion import DDPM
from utilities.sampler import sampler,  DDIM_inversion, DDIM_generation
from PIL import Image
from torchvision import transforms
import os
from os import listdir
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils




class FaceDataset(Dataset):

    def __init__(self,root_dir, transform=None):
        """
        Args:
            
            root_dir (string): Directory with all the original images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir
        self.transform = transform
        

    def __len__(self):
        folder_dir = self.root_dir
        image_list = os.listdir(folder_dir)
        return len(image_list)

    def __getitem__(self, idx):
        
        img_list = os.listdir(self.root_dir)
       
        
        if len(img_list) == 0:
            print("Dataloader cannot load images. There is no image file in this folder")
            img = []
            img_name = ""
        else:
            img_add = os.path.join(self.root_dir, img_list[idx])  
            img = Image.open(img_add).convert("RGB")
            img = self.transform(img) 
            img_name = img_list[idx]     

        return img, img_name


