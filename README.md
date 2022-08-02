# Joint Denoising and Super-resolution for Fluorescence Microscopy using Weakly-supervised Deep Learning
This is the official Pytorch implementation of "Joint Denoising and Super-resolution for Fluorescence Microscopy using Weakly-supervised Deep Learning" (MOVI 2022), written by Colin S. C. Tsang, Tony C. W. Mok and Albert C. S. Chung.

## Prerequisites
This code was tested with `Pytorch 1.10.0` and NVIDIA GeForce RTX 3080 Ti.

## Training and testing scripts
- `train.py`: Train a U-Net model in an <u>weakly-supervised</u> manner.

- `test.py`: Test the model and evaluate it in RMSE and SSIM. 

## Pre-trained model 
Pre-trained model: `model\pretrained_2x_Unet.pth`

## Train or test your own model
Step 1: Download the dataset from https://github.com/IVRL/w2s and place it under the `dataset` folder.

Step 2: Adjust the parameters to your desired value. 

Step 3: Run `train.py` or `test.py`

## Publication
If you find this repository useful, please cite:
- **Joint Denoising and Super-resolution for Fluorescence Microscopy using Weakly-supervised Deep Learning**  
Colin S. C. Tsang, Tony C . W. Mok, and Albert C. S. Chung  
MOVI 2022


## Acknowledgment
Some codes in this repository are modified from https://github.com/BUPTLdy/Pytorch-LapSRN

The SSIM function is provided by https://github.com/jacenfox/pytorch-msssim

###### Keywords
Keywords: Super-resolution, Denoising, Weakly-supervised.
