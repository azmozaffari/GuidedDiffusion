import torch
from models.classifier import *
from utilities.sampler import sampler,  DDIM_inversion, DDIM_generation
import torch.optim as optim
from utilities.load_data import *
from torch.autograd import Variable 
import torch.nn as nn
from scheduler.linear_noise_scheduler import LinearNoiseScheduler,LinearNoiseSchedulerDDIM
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import copy

def train(model, config, dataloader_source, dataloader_target):
    torch.cuda.empty_cache() 
    n_epochs = config.training.n_epochs
    scheduler = LinearNoiseSchedulerDDIM(num_timesteps=config.samplingDDIM.DDPM_num_timesteps,
                                    beta_start=config.samplingDDIM.beta_start,
                                    beta_end=config.samplingDDIM.beta_end)
    lr = config.training.lr
    step = config.samplingDDIM.stepsize_backward
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    
    lr_schedul = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.2) #lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1.2, total_iters=50)
    
    max_norm = 0.01  # Define the max gradient value for each parameter

    loss_clip = ClipLoss(config)
    loss_id = IDLoss(config)
    loss_mse = nn.MSELoss()

    model_ = model    
    # model.attribute = list(model.attribute)  # where attribute was dict_keys
    model_ = copy.deepcopy(model)
    
    
    model.train()
    
    for epoch in range(n_epochs):
       
        total_loss = 0
        batch_size = 0
        total_loss = [0]*int(config.samplingDDIM.t/step)
        for img,_,  in dataloader_source:
            batch_size += img.size(0) 
            img = img.to(config.device)  
            noisy_img = DDIM_inversion(model_, config, img)
            noisy_img = noisy_img.to(config.device)
            for i in (reversed(range(1,int(config.samplingDDIM.t/step)))):      

                # Get prediction of noise
                noise_pred = model(noisy_img, torch.as_tensor(i*step).unsqueeze(0).to(config.device))        
                # Use scheduler to get x0 and xt-1
                xt_1, x0_pred = scheduler.sample_prev_timestep(noisy_img, noise_pred, torch.as_tensor(i).to(config.device), config.samplingDDIM.sigma, step)
                noisy_img = xt_1 
                
                # loss and backpropagation
                l_2 = loss_id(x0_pred,img )
                l_3 = loss_mse(x0_pred, img)
                l = l_2 + l_3
                total_loss[i] = l.data 
                optimizer.zero_grad()
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                noisy_img = noisy_img.clone().detach().requires_grad_(False)
                    

        total_loss = 0
        batch_size = 0
        total_loss = [0]*int(config.samplingDDIM.t/step)

        for img,_,  in dataloader_target:
            batch_size += img.size(0) 
            img = img.to(config.device)  
            noisy_img = DDIM_inversion(model_, config, img)
            noisy_img = noisy_img.to(config.device)
            for i in (reversed(range(1,int(config.samplingDDIM.t/step)))):      

                # Get prediction of noise
                noise_pred = model(noisy_img, torch.as_tensor(i*step).unsqueeze(0).to(config.device))        
                # Use scheduler to get x0 and xt-1
                xt_1, x0_pred = scheduler.sample_prev_timestep(noisy_img, noise_pred, torch.as_tensor(i).to(config.device), config.samplingDDIM.sigma, step)
                noisy_img = xt_1 
                       
                # loss and backpropagation
                l_1 = loss_clip(x0_pred, img)
                l_2 = loss_id(x0_pred,img )
                l_3 = loss_mse(x0_pred, img)
                l =  (l_1 + l_2 + l_3)
                total_loss[i] = l.data 
                optimizer.zero_grad()
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
            # we break the gradient tree in every cycly to have efficient gpu training version
                noisy_img = noisy_img.clone().detach().requires_grad_(False)
            ims = torch.clamp(x0_pred, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2        
            for j in range(ims.size(0)):
                torchvision.utils.save_image(ims[j, :, :, :], os.path.join(config.training.output_img,str(epoch),_[j]))              
        lr_schedul.step()




        torch.save(model.state_dict(), config.checkpoints.finetuned_diffusion_checkpoints+"/ckpt"+str(epoch))      
        print("Epoch %d: SGD lr=%.4f"% (epoch, optimizer.param_groups[0]["lr"]))
        print("total loss:", torch.mean(torch.tensor(total_loss)/batch_size).data)    
        


    
 


    





