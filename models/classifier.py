import torch
import torchvision.transforms as T

import clip
import numpy as np

def loss(config, img1, img2):
    img1 = img1.to(config.device)
    img2 = img2.to(config.device)
    
    model, preprocess = clip.load("ViT-B/32",device=config.device)

    preprocess.transforms.pop(2)
    preprocess.transforms.pop(2)
    img1 = preprocess(img1)
    img2 = preprocess(img2)
    
    with torch.no_grad():
        image_features_1 = model.encode_image(img1).float()
        image_features_2 = model.encode_image(img2).float()

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_1 = cos(image_features_1, image_features_2)


    loss_id
    
    return loss_1