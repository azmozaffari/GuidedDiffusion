import argparse
import traceback
import yaml
import sys
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from models.diffusion import DDPM
from models.classifier import *
from utilities.sampler import sampler,  DDIM_inversion, DDIM_generation
from utilities.training import *
from utilities.test import *
from utilities.load_data import *
import torchvision

import os
torch.cuda.empty_cache() 


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    new_config.device = device

    # ---------------- set random seed -----------------------------------------------
    torch.manual_seed(32)
    np.random.seed(32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(32)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()   


# load pretrained checkpoints for face generation model
    model_pretrained = DDPM(config)
    ckpt =  torch.load(config.checkpoints.pretrained_diffusion_checkpoint, weights_only=True)
    model_pretrained.load_state_dict(ckpt)
    model_pretrained.to(config.device)
    print("Pretrained diffusion model loaded")
    
# ------------------  TRAIN ----------------------------

 
    # define the image transformet
    transform = transforms.Compose([            
            transforms.Resize((config.data.image_size,config.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) 
   
    d_target = FaceDataset(config.training.target_img, transform)
    dataloader_target = DataLoader(d_target, batch_size=config.training.batch_size,
                        shuffle=True, num_workers=1)    
    
    train(model_pretrained, config,dataloader_target)

    return 0


if __name__ == '__main__':
    sys.exit(main())
