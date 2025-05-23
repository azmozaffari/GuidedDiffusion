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

def train(model, config, dataloader_target):
    # read ref image
    transform = transforms.Compose([            
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]) 

    

    torch.cuda.empty_cache() 
    n_epochs = config.training.n_epochs
    scheduler = LinearNoiseSchedulerDDIM(num_timesteps=config.samplingDDIM.DDPM_num_timesteps,
                                    beta_start=config.samplingDDIM.beta_start,
                                    beta_end=config.samplingDDIM.beta_end)
    lr = config.training.lr
    step = config.samplingDDIM.stepsize_backward
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    
    
    max_norm = 1  # Define the max gradient value for each parameter

    loss_clip = ClipLoss(config)
    loss_id = IDLoss(config)
    loss_mse = nn.MSELoss()
    loss_emotion = EmoNet(config)
    
    # we need two copies of the model, one for the DDIM forward process to generate noise
    # and one for the backward process as the denoiser that we want to finetune for each sample

    model_ = model    
    model_ = copy.deepcopy(model)
    model.train()

   
    
    for epoch in range(n_epochs):
       
        
        total_loss = 0
        batch_size = 0
        total_loss = [0]*int(config.samplingDDIM.t/step)

        for img,_,  in dataloader_target:
            batch_size += img.size(0) 
            img = img.to(config.device)  
            noisy_img = DDIM_inversion(model_, config, img)
            noisy_img = noisy_img.to(config.device)
            for i in (reversed(range(1,int(config.samplingDDIM.t/step)))):      
                sigma = config.samplingDDIM.sigma * 0.08 * i
                # Get prediction of noise
                noise_pred = model(noisy_img, torch.as_tensor(i*step).unsqueeze(0).to(config.device))        
                # Use scheduler to get x0 and xt-1
                xt_1, x0_pred = scheduler.sample_prev_timestep(noisy_img, noise_pred, torch.as_tensor(i).to(config.device), sigma, step)
                noisy_img = xt_1 
                       
                # loss and backpropagation
                
                if config.classifier == "clip":
                    l_1 = loss_clip( img, x0_pred, config.training.clip_classifier_text_source, config.training.clip_classifier_text_target )
                
                if config.classifier == "emonet":
                    l_1 = loss_emotion(x0_pred,config.training.emonet_emotion)  
                l_2 = loss_id(x0_pred,img )
                l_3 = loss_mse(x0_pred, img)
                l =  (l_1 + l_2 + l_3) #0.1*l_1 for clip
                
                total_loss[i] = l.data 
                optimizer.zero_grad()
                l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
            # we break the gradient tree in every cycly to have efficient gpu training version
                noisy_img = noisy_img.clone().detach().requires_grad_(False)


            if not os.path.exists(os.path.join(config.training.output_img,str(epoch))):
                os.makedirs(os.path.join(config.training.output_img,str(epoch)))

            ims = torch.clamp(x0_pred, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2        
            for j in range(ims.size(0)):
                torchvision.utils.save_image(ims[j, :, :, :], os.path.join(config.training.output_img,str(epoch),_[j]))   
           




        torch.save(model.state_dict(), config.checkpoints.finetuned_diffusion_checkpoints+"/ckpt"+str(epoch))      
        print("Epoch %d: SGD lr=%.4f"% (epoch, optimizer.param_groups[0]["lr"]))
        print("total loss:", torch.mean(torch.tensor(total_loss)/batch_size).item())    
        


    
 


    





