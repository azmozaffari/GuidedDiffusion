# GuidedDiffusion

Recently, it has been shown in the literature that using CLIP as a classifier to guide the generative models like GAN and StyleGAN can lead to interesting results. Here, I've implemented [DiffusionClip](https://github.com/gwang-kim/DiffusionCLIP) paper that proposed to use CLIP as a classifier to guide the diffusion model.

Here, to generate the samples with guided text and clip classifier,  I only fine-tune the model for 2 epochs and generate the changed samples during the training phase.

In order to use less GPU resources, the GPU-efficient fine-tunning approach is only implemented.

I also compare the generative results, in the case of using Emonet versus CLIP to guide the diffusion model.

A part of the codes is grabed from [DiffusionClip](https://github.com/gwang-kim/DiffusionCLIP) original implementation. The DDPM implementation is downloaded from [https://github.com/explainingai-code/DDPM-Pytorch/tree/main](https://github.com/explainingai-code/DDPM-Pytorch/tree/main).

# Quick Start

* Create a new conda environment with python 3.8 then run below commands

* git clone https://github.com/azmozaffari/GuidedDiffusion.git

* cd GuidedDiffusion

* pip install -r requirements.txt

* Run the batch file run.sh by ./run.sh

* The target image folder is in ./data/training/target. Drop some CelebA images in this folder

* Run main.py by python3 main.py --config config.yml

* See the results in ./data/trainig/output
   



This is the DDPM code that I have used here https://github.com/explainingai-code/DDPM-Pytorch/tree/main


The pretrained weights for the id_loss is grabbed from https://drive.usercontent.google.com/download?id=1b7d9xyvUm1y2xxMyX_LTy9UgamVf563F&export=download




