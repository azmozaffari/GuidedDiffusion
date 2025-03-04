import torch
from models.diffusion import DDPM
from utilities.sampler import sampler,  DDIM_inversion, DDIM_generation
from PIL import Image
from torchvision import transforms
import os
from os import listdir
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# get the path/directory
# def convertImgtoNoise(model,config, condition="train"):
#     if condition == "train":
#         clean_image_path = config.training.clean_address
#         noisy_image_path = config.training.noisy_address
#     if condition == "test":
#         clean_image_path = config.test.clean_address
#         noisy_image_path = config.test.noisy_address


#     folder_dir = clean_image_path
#     image_list = os.listdir(folder_dir)
    
#     for i in range(len(image_list)):
#         # Read a PIL image
#         image = Image.open(folder_dir +"/"+image_list[i])
#         transform = transforms.Compose([
#             transforms.PILToTensor(),
#             transforms.Resize((config.data.image_size,config.data.image_size))        
#         ]) 
#         x0 = transform(image).to(config.device)
#         x0 = x0.unsqueeze(0).float()
#         x_min, x_max = x0.min(), x0.max()
#         x0 = (x0 - x_min) / (x_max-x_min)
#         tr = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         x0 = tr(x0)



#         # add noise till forward steps
#         xt = DDIM_inversion(model, config, x0)

#         ims = torch.clamp(xt, -1., 1.).detach().cpu()
#         ims = (ims + 1) / 2
#         ims = ims.squeeze(0)    
#         img = transforms.ToPILImage()(ims)

#         if not os.path.exists(noisy_image_path):
#             os.mkdir(noisy_image_path)    
#         img.save(os.path.join(noisy_image_path, image_list[i]))
#         img.close()
#     return xt




class FaceDataset(Dataset):
    """source image, noisy image and target image."""

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
        folder_dir = self.root_dir+"/img"
        image_list = os.listdir(folder_dir)
        return len(image_list)

    def __getitem__(self, idx):
        

        
        img_list = os.listdir(self.root_dir+"/img")
        label_list = os.listdir(self.root_dir+"/label")

        
        if len(img_list) == 0:
            print("Dataloader tries to load images. There is no file in the image folder")
            img = []
            img_name = ""
        else:
            img_add = os.path.join(self.root_dir,'img', img_list[idx])  
            img = Image.open(img_add)
            img = self.transform(img) 
            img_name = img_list[idx]     


        if len(label_list) == 0:
            print("Dataloader tries to load labels. There is no file in the label folder")
            label = []
            label_name = ""
        else:
            label_add = os.path.join(self.root_dir,'label', label_list[idx]) 
            label = Image.open(label_add)
            label = self.transform(label)
            label_name =  label_list[idx]   



        
        
        
        




        return img, img_name, label, label_name
