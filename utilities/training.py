import torch
from models.classifier import *
from utilities.sampler import sampler,  DDIM_inversion, DDIM_generation
import torch.optim as optim
from utilities.load_data import *
from torch.autograd import Variable 


def train(model, config, dataloader):

    n_epochs = config.training.n_epochs
    # batch_size = config.training.batch_size
    
    lr = config.training.lr
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    for param in model.parameters():
        param.requiers_grad = True
        # param.requires_grad = False
    for epoch in range(n_epochs):
        for clean_img, noisy_img, label in dataloader:
            clean_img = clean_img.to(config.device)
            noisy_img = noisy_img.to(config.device)
            label = label.to(config.device)

            clean_img = DDIM_generation(model, config, noisy_img)
            l = Variable(loss(config, clean_img, label))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
        print("Epoch %d: SGD lr=%.4f" % (epoch, optimizer.param_groups[0]["lr"]))
 


    





