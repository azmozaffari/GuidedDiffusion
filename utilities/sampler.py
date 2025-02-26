import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from scheduler.linear_noise_scheduler import LinearNoiseScheduler



def sampler(model, config):
    # torch.cuda.empty_cache() 
    scheduler = LinearNoiseScheduler(num_timesteps=config.sampling.num_timesteps,
                                     beta_start=config.diffusion.beta_start,
                                     beta_end=config.diffusion.beta_end)
    model.eval()
    xt = torch.randn((1,
                      config.data.channels,
                      config.data.image_size,
                      config.data.image_size)).to(config.device)
    with torch.no_grad():
        for i in tqdm(reversed(range(config.sampling.num_timesteps))):
            # Get prediction of noise
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(config.device))
            
            # Use scheduler to get x0 and xt-1
            xt_1, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(config.device))
            xt = xt_1
            # Save x0
            ims = torch.clamp(xt, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2
            ims = ims.squeeze(0)
            # grid = make_grid(ims, nrow=train_config['num_grid_rows'])
            # img = torchvision.transforms.ToPILImage()(grid)
            
            img = torchvision.transforms.ToPILImage()(ims)


            if not os.path.exists(os.path.join(config.output.address, 'samples')):
                os.mkdir(os.path.join(config.output.address, 'samples'))
            
            img.save(os.path.join(config.output.address, 'samples', 'x0_{}.png'.format(i)))
            img.close()



