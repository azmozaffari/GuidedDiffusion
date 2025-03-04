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
    scheduler = LinearNoiseSchedulerDDIM(num_timesteps=config.samplingDDIM.num_timesteps,
                                    beta_start=config.samplingDDIM.beta_start,
                                    beta_end=config.samplingDDIM.beta_end)
    lr = config.training.lr
    step = config.samplingDDIM.step_backward
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_function = loss(config)
   
    model.train()
    
    for epoch in range(n_epochs):
       
        total_loss = 0
        batch_size = 0
        total_loss = [0]*int(config.samplingDDIM.forward_timesteps/step)

        for img,_,  label,_ in dataloader:
            batch_size += img.size(0) 
            img = img.to(config.device)  
            noisy_img = DDIM_inversion(model, config, img)
            noisy_img = noisy_img.to(config.device)
            label = label.to(config.device)            
            
            for i in tqdm(reversed(range(1,int(config.samplingDDIM.forward_timesteps/step)))):      

                # Get prediction of noise
                noise_pred = model(noisy_img, torch.as_tensor(i*step).unsqueeze(0).to(config.device))        
                # Use scheduler to get x0 and xt-1
                xt_1, x0_pred = scheduler.sample_prev_timestep(noisy_img, noise_pred, torch.as_tensor(i).to(config.device), config.samplingDDIM.sigma, step)
                noisy_img = xt_1 
                

               # loss and backpropagation
                l = loss_function(x0_pred, label)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                total_loss[i] = l.data
                # we break the gradient tree in every cycly to have efficient gpu training version
                noisy_img = noisy_img.clone().detach().requires_grad_(False)
            
            
        print("Epoch %d: SGD lr=%.4f"% (epoch, optimizer.param_groups[0]["lr"]))
        print("total loss:", torch.mean(torch.tensor(total_loss)/batch_size).data)    
        if epoch%5 == 0:
            torch.save(model.state_dict(), config.training.checkpoints+"/ckpt"+str(epoch))
    


    
 


    





