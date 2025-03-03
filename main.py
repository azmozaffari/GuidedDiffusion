import argparse
import traceback
import logging
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
from utilities.load_data import *


import os
torch.cuda.empty_cache() 
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<enter-size-here>"


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"




def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='./img', help='Path for saving running related data.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--sample', action='store_true', help='Whether to produce samples from the model')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    # parser.add_argument('--npy_name', type=str, required=True)
    parser.add_argument('--sample_step', type=int, default=3, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=400, help='Sampling noise scale')
    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

   

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    print(">" * 80)
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print("<" * 80)

    try:
        model = DDPM(config)
        # runner.image_editing_sample()
    except Exception:
        logging.error(traceback.format_exc())


    # load pretrained weights to DDPM
    ckpt =  torch.load("./pre_trained/celeba_hq.ckpt", weights_only=True)
    model.load_state_dict(ckpt)
    model.to(config.device)
    # model = torch.nn.DataParallel(model)
    print("Model loaded")



    # convertImgtoNoise(model,config)  ################  prepare training data by adding noise

    d = FaceDataset("./data/training")
    dataloader = DataLoader(d, batch_size=2,
                        shuffle=True, num_workers=1)
    
    
    train(model, config, dataloader)
        
    return 0


if __name__ == '__main__':
    sys.exit(main())
