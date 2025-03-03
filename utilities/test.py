import torch


from models.diffusion import DDPM
from utilities.load_data import *
from scheduler.linear_noise_scheduler import LinearNoiseScheduler,LinearNoiseSchedulerDDIM


def test(model,config,xt):
    

    
    folder_dir = config.test.noisy_address
    image_list = os.listdir(folder_dir)
    



    ##################################
#########################################
################  IN TRAIN AND TEST FIRST WE SHOULD GENERATE NOISE AND THEN REVERSE IT WHEN I SAVE\

################  NOISE THE VALUES ARE NOT BETWEEN )<! THEN IT CANNOT REVERSE IT CORRECT TEST>PY AND TRAIN>PY FILE  





   
        
    generated_clean_img = DDIM_generation(model, config, xt)
    ims = torch.clamp(generated_clean_img, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2
    ims = ims.squeeze(0)    
    img = transforms.ToPILImage()(ims)





    if not os.path.exists(config.test.generated_address):
        os.mkdir(config.test.generated_address)    
    img.save(os.path.join(config.test.generated_address, "image_list[i].jpg"))
    img.close()




