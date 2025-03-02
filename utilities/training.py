import torch
from models.classifier import *
from utilities.sampler import sampler,  DDIM_inversion, DDIM_generation
import torch.optim as optim
from utilities.load_data import *
from torch.autograd import Variable 
import torch.nn as nn


def train(model, config, dataloader):

    n_epochs = config.training.n_epochs
    # batch_size = config.training.batch_size
    
    lr = config.training.lr
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # loss = nn.MSELoss()
    l = loss(config)
   
    model.train()
    for param in model.parameters():
        param.requiers_grad = True
        # param.requires_grad = False
    for epoch in range(n_epochs):
        total_loss = 0
        batch_size = 0
        for clean_img, noisy_img, label in dataloader:
            batch_size += clean_img.size(0) 
            clean_img = clean_img.to(config.device)
            noisy_img = noisy_img.to(config.device)
            label = label.to(config.device)

            clean_img = DDIM_generation(model, config, noisy_img)
            # l = Variable(loss(config, clean_img, label))
            l_1 = l(clean_img, label)
            optimizer.zero_grad()
            l_1.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            # optimizer.step()
            optimizer.step()
            total_loss += l_1.data
            
        print("Epoch %d: SGD lr=%.4f Total loss= %.4f"% (epoch, optimizer.param_groups[0]["lr"], total_loss/batch_size))
 


    





