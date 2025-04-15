# GuidedDiffusion

Recently, it has been shown in the literature that using CLIP as a classifier to guide the generative models like GAN and StyleGAN can lead to interesting results. Here, I've implemented [DiffusionClip](https://github.com/gwang-kim/DiffusionCLIP) model as a classifier guided diffusion model.

Here, to generate the samples with guided text and clip classifier,  I only fine-tune the model for 2 epochs and generate the changed samples during the training phase.

I also compare the generative results, in the case of using Emonet and CLIP to guide the diffusion model.

# Quick Start









This is the DDPM code that I have used here https://github.com/explainingai-code/DDPM-Pytorch/tree/main


The pretrained weights for the id_loss is grabbed from https://drive.usercontent.google.com/download?id=1b7d9xyvUm1y2xxMyX_LTy9UgamVf563F&export=download




