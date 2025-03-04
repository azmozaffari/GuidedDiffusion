import torch


from models.diffusion import DDPM
from utilities.load_data import *
from scheduler.linear_noise_scheduler import LinearNoiseScheduler,LinearNoiseSchedulerDDIM
import torchvision

def test(finetuned_model, original_model,config):
    

    
    folder_dir = config.test.img_address
    image_list = os.listdir(folder_dir)
    
    if not os.path.exists(config.test.gen_img_address):
        os.mkdir(config.test.gen_img_address)    

    for i in range(len(image_list)):
        # this line can be replaced with the dataloader
        image = Image.open(folder_dir +"/"+image_list[i])

        transform = transforms.Compose([            
            transforms.Resize((config.data.image_size,config.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) 
        x0 = transform(image)
        x0 = x0.to(config.device)


        # add noise till forward steps
        xt = DDIM_inversion(original_model, config, x0)
   
        
        generated_clean_img = DDIM_generation(finetuned_model, config, xt)
        
        ims = torch.clamp(generated_clean_img, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2        
        for j in range(ims.size(0)):
           torchvision.utils.save_image(ims[j, :, :, :], os.path.join(config.test.gen_img_address,image_list[i]))
        
    