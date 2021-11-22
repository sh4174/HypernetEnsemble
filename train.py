"""
Training script for Hyper-ResUNet ensemble. 
This code is based on the tutorial codes of the MONAI framework in 
https://github.com/Project-MONAI/tutorials/tree/master/3d_segmentation
"""

# System / Misc
import os
import glob 
import pandas as pd 
import time 

# Torch / Numpy
import torch
import numpy as np

# Lightning 
import pytorch_lightning
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

# MONAI
from monai.config import print_config

# Noramlization 
from monai.networks.layers import Norm

# Transforms
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    RandFlipd,
    RandSpatialCropd,
    RandShiftIntensityd,
    ScaleIntensityd,
    RandAdjustContrastd,
    EnsureTyped
)

# Data/Loader/Dataset
from monai.data import (
    DataLoader,
    CacheDataset,
    decollate_batch,
    list_data_collate,
)

# Hypernetwork Module
from lightning_modules.module_hyper_resunet import Hyper_UNet_Module

# MONAI/Torch Environment and Configurations
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device(0)
torch.backends.cudnn.benchmark = True
print_config()


#########################################
###                Main               ###
#########################################

# Training data directories
# Training data are not provided in the repo because of the IRB regulation
# Image directory path (only tested on nifti format)
img_dir = 'DWI/'
# Human annotation directory path (only tested on nifti format) 
seg_dir = 'Lesion/'
# k-fold data separation for training/validation
n_fold_val = 7

# Output directory
model_dir = 'models/hypernet_vTversky/'
# Model checkpoint prefix 
model_ckpt_prefix = "hypernet_vTversky_{epoch:04d}"

# Primaryt network parameters
spatial_dims=3
in_channels=1
out_channels=2
channels=(16, 32, 64, 128, 256)
strides=(2, 2, 2, 2)
num_res_units=2
norm=Norm.BATCH
# Hypernet mapping network params
hyper_n_invar=2
hyper_n_layers=3
hyper_n_feat=16

# Training params
# No. Epochs
max_epochs=2
# Validation Epochs - Every check_val, do validation step for tracking training
check_val=1
# Training batch size 
tr_batch_size = 8
# Validation batch size
val_batch_size = 4


#########################################
###              Training             ###
#########################################
# Output Directory
os.makedirs( model_dir, exist_ok=True )

# Set up hypernetwork module
hyper_module = Hyper_UNet_Module(
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
        hyper_n_feat=hyper_n_feat,
        # Training Params
        max_epochs=max_epochs,
        check_val=check_val,
    )

# Logger
tb_logger = pl_loggers.TensorBoardLogger( model_dir + 'tb_logs/' )

# Set up training data 
images_sorted = sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
segs_sorted = sorted(glob.glob(os.path.join(seg_dir, "*.nii.gz")))

# Data permutation
permuted_idx = np.random.permutation( len( images_sorted ) )

images = [ images_sorted[ idx ] for idx in permuted_idx ]
segs = [ segs_sorted[ idx ] for idx in permuted_idx ]

# Training/Validatiaon data separation
len_val = int( len( images ) / n_fold_val )
train_files = [ {"image": img, "label": seg} for img, seg in zip(images[len_val:], segs[len_val:]) ]
val_files = [ {"image": img, "label": seg} for img, seg in zip(images[:len_val], segs[:len_val]) ]

# Data transformations for loading data and basic augmentation
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        RandAdjustContrastd( keys=["image"], prob=0.2 ),
        ScaleIntensityd(
            keys=["image"],
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=(192, 192, 16),
            random_size=False,
        ),
        EnsureTyped(keys=["image", "label"]),
    ]
)

train_ds = CacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_rate=1.0,
    num_workers=8,
)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=tr_batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    collate_fn=list_data_collate,
)

# Data Transformations for Validation (No augmentation) 
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ScaleIntensityd(
            keys=["image"],
        ),
        EnsureTyped(keys=["image", "label"]),
    ]
)
val_ds = CacheDataset(
    data=val_files,
    transform=val_transforms,
    cache_rate=1.0,
    num_workers=4,
)
val_loader = torch.utils.data.DataLoader( val_ds, batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True )

# Set up checkpoint
checkpoint_callback = ModelCheckpoint(dirpath=model_dir, filename=model_ckpt_prefix, save_last=True, save_top_k=1)

# Set up trainer 
trainer = pytorch_lightning.Trainer(
    gpus=[0],
    max_epochs=max_epochs,
    check_val_every_n_epoch=check_val,
    callbacks=[checkpoint_callback],
    default_root_dir=model_dir,
    log_every_n_steps=1,
    logger=tb_logger,
)

# Train
start_time = time.time()
trainer.fit(hyper_module, train_loader, val_loader)
end_time = time.time()

print( "========================================================" )
print( "Training Time" )
print( f'{end_time - start_time} sec.' )
print( "========================================================" )
