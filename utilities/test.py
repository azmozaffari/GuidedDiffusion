import torch


from models.diffusion import DDPM
from utilities.load_data import *
from scheduler.linear_noise_scheduler import LinearNoiseScheduler,LinearNoiseSchedulerDDIM
import torchvision

def test(finetuned_model, original_model,config, dataloader):
    

    
    
    for img, img_name, label, label_name in dataloader:
        # this line can be replaced with the dataloader
        
       
        x0 = img.to(config.device)

        # add noise till forward steps
        xt = DDIM_inversion(original_model, config, x0)
        # xt = torch.randn((x0.size(0),
        #               config.data.channels,
        #               config.data.image_size,
        #               config.data.image_size)).to(config.device)
        generated_clean_img = DDIM_generation(finetuned_model, config, xt)
        # ims = generated_clean_img
        ims = torch.clamp(generated_clean_img, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2        
        for j in range(ims.size(0)):
           torchvision.utils.save_image(ims[j, :, :, :], os.path.join(config.test.gen_img_address,img_name[j]))
        
    