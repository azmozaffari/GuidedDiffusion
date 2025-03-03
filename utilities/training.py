import torch
from models.classifier import *
from utilities.sampler import sampler,  DDIM_inversion, DDIM_generation
import torch.optim as optim
from utilities.load_data import *
from torch.autograd import Variable 
import torch.nn as nn
from scheduler.linear_noise_scheduler import LinearNoiseScheduler,LinearNoiseSchedulerDDIM
from tqdm import tqdm

def train(model, config, dataloader):
    torch.cuda.empty_cache() 
    n_epochs = config.training.n_epochs
    # batch_size = config.training.batch_size
    scheduler = LinearNoiseSchedulerDDIM(num_timesteps=config.samplingDDIM.num_timesteps,
                                    beta_start=config.samplingDDIM.beta_start,
                                    beta_end=config.samplingDDIM.beta_end)
    lr = config.training.lr
    step = config.samplingDDIM.step_backward
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # loss = nn.MSELoss()
    loss_function = loss(config)
   
    model.train()
    # for param in model.parameters():
    #     param.requiers_grad = False
    
    for epoch in range(n_epochs):
       
        total_loss = 0
        batch_size = 0
        total_loss = [0]*int(config.samplingDDIM.forward_timesteps/step)

        for clean_img, noisy_img, label in dataloader:
            batch_size += clean_img.size(0) 
            # clean_img = clean_img.to(config.device)  ## later it can be used for id loss but in this version we do not need this 
            noisy_img = noisy_img.to(config.device)
            label = label.to(config.device)            
            
            for i in tqdm(reversed(range(int(config.samplingDDIM.forward_timesteps/step)))):      


                # Get prediction of noise
                noise_pred = model(noisy_img, torch.as_tensor(i*step).unsqueeze(0).to(config.device))        
                # Use scheduler to get x0 and xt-1
                xt_1, x0_pred = scheduler.sample_prev_timestep(noisy_img, noise_pred, torch.as_tensor(i).to(config.device), config.samplingDDIM.sigma, step)
                noisy_img = xt_1 
                

              
                l = loss_function(x0_pred, label)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                total_loss[i] = l.data
                noisy_img = torch.tensor(noisy_img)
            
            
        print("Epoch %d: SGD lr=%.4f"% (epoch, optimizer.param_groups[0]["lr"]))
        print("total loss:", torch.tensor(total_loss)/batch_size)
 


    





