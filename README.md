# Hypernet-Ensemble Learning of Segmentation Probability for Medical Image Segmentation with Ambiguous Labels

![Human Annotations of Ambiguous Stroke Lesions!](/figures/Git_AmbiguousAnnot.png "Ambiguous Label")

![Estimated Segmentation Probability Map!](/figures/Git_SegProbEst.png "Segmentation Probability Map")

![Segmentation Label Estimation with Different Threshold!](/figures/Git_SegProg_Thres.png "Segmentation Probability Thresholding")

This repository contains the hypernetwork implementation of Hypernet-Ensemble Learning of Segmentation Probability for Medical Image Segmentation with Ambiguous Labels. 

# Hypernetwork Architecture

![Hypernetwork Architecture!](/figures/HyperNet2.png "Hypernetwork")

The overall hypernetwork architecture shown above is implemented in networks/hyper_resunet.py. 

![Hyperconvolution Blocks!](/figures/HyperBlocks.png "Hyperconvolution Blocks")

The hyperconvolution blocks are implemented in blocks/hyper_convolution.py.

# Usage

The codes are implemented with the MONAI framework (<https://monai.io/>), PyTorch (<https://pytorch.org/>), and PyTorch Lightning (<https://www.pytorchlightning.ai/>). 

## Requirements

pip install -r requirements.txt

Example Dependencies
* monai=0.7.0
* torch=1.9.1
* nibabel=3.2.1

## Training 

The hypernetwork and optimizer is wrapped with PyTorch Lightning (lightning_modules/module_hyper_resunet.py). 

Please see train.py for setting up the network parameters and training configurations. 

## Inference 

Please see predict.py for inference.

The network parameters need to be the same with a trained network. 

# Contact

Please contact Sungmin Hong (MGH, HMS <shong20@mgh.harvard.edu>) if you have questions on the codes or the paper. 
