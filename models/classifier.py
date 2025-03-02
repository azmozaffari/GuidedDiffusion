import torch
import clip
import torch.nn as nn



class loss(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32",device=config.device)
        self.preprocess.transforms.pop(2)
        self.preprocess.transforms.pop(2)
        self.config = config

        for param in self.model.parameters():
            param.requiers_grad = False


    
    def forward(self,img1, img2):
        img1 = img1.to(self.config.device)
        img2 = img2.to(self.config.device)
        
        img1 = self.preprocess(img1)
        img2 = self.preprocess(img2)
        
       
        image_features_1 = self.model.encode_image(img1).float()
        image_features_2 = self.model.encode_image(img2).float()

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
        
        return torch.mean(-cos(image_features_1, image_features_2)) 