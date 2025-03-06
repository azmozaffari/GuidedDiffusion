import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from scheduler.linear_noise_scheduler import LinearNoiseScheduler,LinearNoiseSchedulerDDIM



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



    
    

def DDIM_inversion(model, config, x0):
    scheduler = LinearNoiseSchedulerDDIM(num_timesteps=config.samplingDDIM.num_timesteps,
                                    beta_start=config.samplingDDIM.beta_start,
                                    beta_end=config.samplingDDIM.beta_end)
    
    
    
    # generate random gaussian noise and add it to the original image
    z = torch.randn((x0.size(0),
                      config.data.channels,
                      config.data.image_size,
                      config.data.image_size)).to(config.device)
    x1 = scheduler.add_noise(x0,z,1)

    # now predict the added noise with the pretrained model
    noise_pred = model(x1, torch.as_tensor(1).unsqueeze(0).to(config.device))
    
    # starting point is x1 that we want to add noise to it and make x2 and so on
    xt_1 = x1
    step = config.samplingDDIM.step_forward
    with torch.no_grad():
        
        for t in ((range(1,int(config.samplingDDIM.forward_timesteps/step)))):  
            xt = scheduler.sample_forward_timestep(xt_1, torch.as_tensor(t).to(config.device), noise_pred, step)            
            noise_pred = model(xt, torch.as_tensor(t*step).unsqueeze(0).to(config.device))
            xt_1 = xt
    return xt








def DDIM_generation(model, config, xt):
   

    scheduler = LinearNoiseSchedulerDDIM(num_timesteps=config.samplingDDIM.num_timesteps,
                                    beta_start=config.samplingDDIM.beta_start,
                                    beta_end=config.samplingDDIM.beta_end)
    step = config.samplingDDIM.step_backward
    for i in (reversed(range(1,int(config.samplingDDIM.forward_timesteps/step)))):        
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i*step).unsqueeze(0).to(config.device))        
        # Use scheduler to get x0 and xt-1
        xt_1, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(config.device), config.samplingDDIM.sigma, step)
        xt = xt_1





        # ims = torch.clamp(xt, -1., 1.).detach().cpu()
        # ims = (ims + 1) / 2
        # ims = ims.squeeze(0)    
        # img = torchvision.transforms.ToPILImage()(ims)





        # if not os.path.exists(config.test.generated_address):
        #     os.mkdir(config.test.generated_address)    
        # img.save(os.path.join(config.test.generated_address, str(i)+".jpg"))
        # img.close()
    
   
    
    return xt




