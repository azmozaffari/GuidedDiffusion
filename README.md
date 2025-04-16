# GuidedDiffusion

Recently, it has been shown in the literature that using CLIP as a classifier to guide generative models like GAN and StyleGAN can lead to interesting results. Here, I've implemented [DiffusionClip](https://github.com/gwang-kim/DiffusionCLIP) paper that proposed to use CLIP as a classifier to guide the diffusion model.

Here, to generate the samples with guided text and clip classifier,  I only fine-tune the model for 2 epochs and generate the changed samples during the training phase.

To use fewer GPU resources, the GPU-efficient fine-tuning approach is only implemented.

I also compare the generative results, in the case of using Emonet versus CLIP to guide the diffusion model.

A part of the codes is grabed from [DiffusionClip](https://github.com/gwang-kim/DiffusionCLIP) original implementation. The DDPM implementation model is downloaded from [https://github.com/explainingai-code/DDPM-Pytorch/tree/main](https://github.com/explainingai-code/DDPM-Pytorch/tree/main).

# Quick Start

* Create a new conda environment with Python 3.8, then run the following commands

* git clone https://github.com/azmozaffari/GuidedDiffusion.git

* cd GuidedDiffusion

* pip install -r requirements.txt

* Run the batch file run.sh by ./run.sh

* The target image folder is in ./data/training/target. Drop some CelebA images in this folder

* Run main.py by python3 main.py --config celeba.yml

* See the results in ./data/trainig/output
   
# Configuration
* celeba.yml - Allows you to play with different components of the Guided Diffusion model

# Results
* CLIP results:
* original samples from CelebA dataset

  <table>
  <tr>
    <td> <img src="./images/source/00007.jpg"  alt="1" width = 128px height = 128px ></td>
    <td> <img src="./images/source/00008.jpg"  alt="2" width = 128px height = 128px ></td>
    <td> <img src="./images/source/00028.jpg"  alt="3" width = 128px height = 128px ></td>
    <td> <img src="./images/source/00045.jpg"  alt="4" width = 128px height = 128px ></td>
   </tr> 
  </table>

* Results after fine-tuning the diffusion model with CLIP classifier with guided text: "fearful face"
  <table>
     <tr>
    <td> <img src="./images/clipfear/00007.jpg"  alt="1" width = 128px height = 128px ></td>
    <td> <img src="./images/clipfear/00008.jpg"  alt="2" width = 128px height = 128px ></td>
    <td> <img src="./images/clipfear/00028.jpg"  alt="3" width = 128px height = 128px ></td>
    <td> <img src="./images/clipfear/00045.jpg"  alt="4" width = 128px height = 128px ></td>
   </tr>   
</table>

*Results after fine-tuning the diffusion model with CLIP classifier with guided text: "happy face"
  <table>
     <tr>
    <td> <img src="./images/cliphappy/00007.jpg"  alt="1" width = 128px height = 128px ></td>
    <td> <img src="./images/cliphappy/00008.jpg"  alt="2" width = 128px height = 128px ></td>
    <td> <img src="./images/cliphappy/00028.jpg"  alt="3" width = 128px height = 128px ></td>
    <td> <img src="./images/cliphappy/00045.jpg"  alt="4" width = 128px height = 128px ></td>
   </tr> 
</table>



