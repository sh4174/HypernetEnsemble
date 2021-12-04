"""
PyTorch Lightning Wrapper for Hyper-ResUNet Module
For more details on training wrappers in the MONAI framework, 
please see https://github.com/Project-MONAI/tutorials/tree/master/3d_segmentation.
"""

# System / Misc
import sys
import os

# Torch 
import torch

# Torch Lightning 
import pytorch_lightning
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

# MONAI
from monai.networks.layers import Norm
from monai.metrics import DiceMetric

from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType,
)

from monai.data import decollate_batch

# Hypernetwork modules
sys.path.append("..") 
from networks.hyper_resunet import HyperUNet

# Hypernetwork SW inference
from infer.hyper_sliding_window_inference import hyper_sliding_window_inference

# Modified Tversky
from loss.modified_tversky import TverskyLoss_Modified

# Setting up environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device(0)
torch.backends.cudnn.benchmark = True

# Lightning Module
class Hyper_UNet_Module(pytorch_lightning.LightningModule):
    def __init__(self,
            spatial_dims=3,                        # Dimension of input images. Default: 3 (3D images).
            in_channels=1,                         # Channel of input images. Default: 1.
            out_channels=2,                        # Channel of output segmentation label maps (Number of labels). Default :2 (Binary segmentation).
            channels=(16, 32, 64, 128, 256),       # Kernel filter depths of the primary network (ResUNet). Default: (16, 32, 64, 128, 256 ). 
            strides=(2, 2, 2, 2),                  # Kernel strides of the primary network (ResUNet). Default: (2,2,2,2).
            num_res_units=2,                       # Number of residual blocks of the primary network (ResUNet). Default: 2 
            norm=Norm.BATCH,                       # Feature normalization function. Default: Norm.BATCH (Batch normalization).
            # Sliding window params
            sw_roi_size=(192, 192, 16),            # Patch size for sliding window. Must be equal to the patch size of images for training. Defulat: (192, 192, 16).
            sw_batch_size = 8,                     # Batch size for sliding window. Default: 8.
            # Hypernet params
            hyper_n_invar=2,                       # Number of hyperparameters. 2 for Tversky loss hyperparameters (alpha, beta). Default: 2.
            hyper_n_layers=3,                      # Number of the layers of the mapping network. Default: 3.
            hyper_n_feat=16,                       # Dimension of the mapped hypervector. Must be hyper_n_invar*2^(hyper_n_layers). Default: 16. 
            # Hypernet Trainin params
            hyper_alpha_min=0.05,           # Minimum alpha of Tversky loss. Should be in [0.0, 1.0] range and smaller than hyper_alpha_min. Default : 0.05.
            hyper_alpha_max=0.95,           # Maximum alpha of Tversky loss. Should be in [0.0, 1.0] range and larger than hyper_alpha_min. Default : 0.95.
            hyper_endp_prob=-1,             # Weights on the maximum/minimum alphas. e.g., 0.1: each 10% of training will be trained with hyper_alpha_min or hyper_alpha_max. Default: -1 (No weights).
            # Training Params
            max_epochs=2000,                # Number of training epochs. Default: 2000.
            check_val=50,                   # Step size for in-training validation. e.g., 50: Run validation at every 50 epochs. Default: 50.
            warmup_epochs=0,                # Deprecated. Warm-up epochs. Default: 0.
            # Optimizer Params
            learning_rate=1e-5,             # Learning rate of Adam optimizer. Default: 1e-5.
            weight_decay=1e-3,              # Weight decay of Adam optimizer. Default: 1e-3.
        ):
        super().__init__()

        # Primary Network
        self._model = HyperUNet(
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
        ).to(device)

        self.max_epochs = max_epochs
        self.check_val = check_val
        self.warmup_epochs = warmup_epochs

        self.loss_function = TverskyLoss_Modified( to_onehot_y=True, softmax=True )
        self.post_pred = Compose( [ EnsureType(), AsDiscrete(argmax=True, to_onehot=True, num_classes=2) ] )
        self.post_label = Compose( [ EnsureType(), AsDiscrete(to_onehot=True, num_classes=2) ] )
        self.dice_metric = DiceMetric( include_background=False, reduction="mean", get_not_nans=False )
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.metric_values = []
        self.epoch_loss_values = []

        # Hypernet training params
        self.hyper_alpha_min = hyper_alpha_min
        self.hyper_alpha_max = hyper_alpha_max
        self.hyper_endp_prob = hyper_endp_prob

        # Optimizer params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Sliding window params 
        self.sw_roi_size = sw_roi_size
        self.sw_batch_size = sw_batch_size


    def forward(self, x, hyper_input):
        return self._model(x, hyper_input)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam( self._model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay )
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = (batch["image"].cuda(), batch["label"].cuda())

        endp_prob = torch.rand( 1 )
        if endp_prob < self.hyper_endp_prob:
            hyper_input_alpha = torch.Tensor( [ self.hyper_alpha_min ] )
        elif endp_prob > self.hyper_endp_prob:
            hyper_input_alpha = torch.Tensor( [ self.hyper_alpha_max ] )
        else:
            hyper_input_alpha = ( self.hyper_alpha_max - self.hyper_alpha_min ) * torch.rand( 1 ) + self.hyper_alpha_min

        hyper_input_beta = 1 - hyper_input_alpha
        hyper_input = torch.cat( ( hyper_input_alpha, hyper_input_beta ) ).to( device ) 

        output = self.forward( images, hyper_input )
        loss = self.loss_function(output, labels, hyper_input[ 0 ], hyper_input[ 1 ] )

        tensorboard_logs = {"train_loss": loss.item()}
        self.log( 'train/loss/step', loss.item() )

        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())
        self.log( 'train/loss/epoch', avg_loss.item() )
        

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = self.sw_roi_size
        sw_batch_size = self.sw_batch_size

        hyper_input_alpha = ( self.hyper_alpha_max - self.hyper_alpha_min ) * torch.rand( 1 ) + self.hyper_alpha_min
        hyper_input_beta = 1 - hyper_input_alpha
        hyper_input = torch.cat( ( hyper_input_alpha, hyper_input_beta ) ).to( device ) 

        outputs = hyper_sliding_window_inference(
            images, hyper_input, roi_size, sw_batch_size, self.forward
        )

        loss = self.loss_function(outputs, labels, hyper_input[ 0 ], hyper_input[ 1 ] ) 
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        return {"val_loss": loss, "val_number": len(outputs)}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.log( 'validation/epoch/dice', mean_val_dice )
        self.log( 'validation/epoch/loss', mean_val_loss )
        

        self.metric_values.append(mean_val_dice)
        return {"log": tensorboard_logs}

