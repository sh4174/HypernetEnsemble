"""
Prediction script for Hyper-ResUNet ensembles. 
"""

# System / Numpy
import os
import sys
import glob

import numpy as np

# MONAI / PyTorch
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    EnsureType,
    EnsureTyped,
    Activations,
    SaveImage,
)

from monai.networks.layers import Norm

from monai.config import print_config

from monai.data import (
    DataLoader,
    CacheDataset,
    decollate_batch,
)

import torch

# Hypernetwork 
# Hypernetwork Lightning Module
from lightning_modules.module_hyper_resunet import Hyper_UNet_Module

# Hypernetwork SW inference
from infer.hyper_sliding_window_inference import hyper_sliding_window_inference

# Set up PyTorch/MONAI Environment and Logger
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device(0)
torch.backends.cudnn.benchmark = True
print_config()

############################
####       Predict      ####
############################
# Model Directory
model_dir = 'models/conf10/'
model_name = 'last.ckpt'

# Validation image directory path
# Test images are not provided in the repo because of the IRB regulation  
val_img_dir = 'DWI/'

# Output directory path 
out_dir = 'outputs/conf10/'
os.makedirs( out_dir, exist_ok=True )
# Segmentation probability map output directory path
val_out_prob_dir = out_dir + "Validation_SegProb/"
os.makedirs( val_out_prob_dir, exist_ok=True )
# Binary segmentation output directory path
val_out_dir = out_dir + "Validation_Seg/"
os.makedirs( val_out_dir, exist_ok=True )

# Trained Network Parameters
# Primary network parameters
spatial_dims=3
in_channels=1
out_channels=2
channels=(32, 32, 64, 64, 128)
strides=(2, 2, 2, 2)
num_res_units=2
norm=Norm.BATCH
# Hypernet mapping network params
hyper_n_invar=2
hyper_n_layers=5
hyper_n_feat=64

# Hyperparameter - alpha for the varying Tversky Loss
lambda_alphas = np.arange( 0.1, 1.0, 0.1 )

# Inference Parameters
roi_size = (192, 192, 16)
sw_batch_size = 8
overlap = 0.8


#######################################################
###                  Validation                     ### 
#######################################################

# Set up module
module = Hyper_UNet_Module()
# Load trained module
module = module.load_from_checkpoint(
        os.path.join(model_dir, model_name ),
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm=norm,
        # Hypernet params
        hyper_n_invar=hyper_n_invar,
        hyper_n_layers=hyper_n_layers,
        hyper_n_feat=hyper_n_feat
    )
module.eval()
module.to(device)

# Input transform
eval_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        ScaleIntensityd(
            keys=["image"],
        ),
        EnsureTyped(keys=["image"]),
    ]
)

# Output transform for the probability estimation
post_prob_transforms = Compose(
    [
        EnsureType(),
        Activations(sigmoid=True),
        EnsureType(),
    ]
)
# Post-Transform for Binary Segmentation - ArgMax
post_binary_transforms = Compose(
    [
        EnsureType(),
        AsDiscrete(argmax=True, threshold_values=True),
        EnsureType(),
    ]
)

# Validation Output Directory
val_saver = SaveImage(output_dir=val_out_dir, output_ext=".nii.gz", output_postfix="seg", separate_folder=False, print_log=False )
val_prob_saver = SaveImage(output_dir=val_out_prob_dir, output_ext=".nii.gz", output_postfix="seg_prob", separate_folder=False, print_log=False )

# Validation Set
val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.nii.gz")))
val_files = [ {"image": img} for img in val_images ]

# Indiviual Image evaluation
val_ds = CacheDataset(
    data=val_files,
    transform=eval_transforms,
    cache_rate=1.0,
    num_workers=1,
)
val_loader = torch.utils.data.DataLoader( val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True )

val_img_name_arr = [] 
val_vol_arr = []
val_pred_vol_arr = []
val_dice_arr = [] 

with torch.no_grad():
    for val_i, val_data in enumerate( val_loader ):
        img_file_prefix_i = os.path.basename( val_data[ 'image_meta_dict' ][ 'filename_or_obj' ][ 0 ][ :-7 ] )

        val_img = val_data[ 'image' ].to( device )
        val_meta = val_data[ 'image_meta_dict' ]

        val_meta = decollate_batch( val_meta )[ 0 ]

        val_output_probs = []
        for i in range( len( lambda_alphas ) ):
            lambda_alpha = lambda_alphas[ i ]
            lambda_beta = 1 - lambda_alpha
            hyper_input = torch.Tensor( [ lambda_alpha, lambda_beta ] ).to(device)

            val_output_i = hyper_sliding_window_inference(
                val_img, hyper_input, roi_size, sw_batch_size, module, overlap=overlap
            )

            val_output_prob_i = post_prob_transforms( decollate_batch( val_output_i )[ 0 ] )
            val_output_probs.append( val_output_prob_i )

        # Stack and reduce mean for probability estimation
        val_output_prob = torch.mean( torch.stack( val_output_probs, dim=0 ), dim=0 )
        val_prob_saver( val_output_prob, val_meta )

        # Binarization
        val_output = post_binary_transforms( val_output_prob )
        val_saver(val_output, val_meta )
