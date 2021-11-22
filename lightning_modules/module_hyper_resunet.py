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
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            # Hypernet params
            hyper_n_invar=2,
            hyper_n_layers=3,
            hyper_n_feat=16,
            # Training Params
            max_epochs=2000,
            check_val=50,
            warmup_epochs=0
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

    def forward(self, x, hyper_input):
        return self._model(x, hyper_input)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam( self._model.parameters(), lr=1e-5, weight_decay=1e-3 )
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = (batch["image"].cuda(), batch["label"].cuda())

        hyper_input_alpha = ( 0.95 - 0.05 ) * torch.rand( 1 ) + 0.05
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
        roi_size = (192, 192, 16)
        sw_batch_size = 8

        hyper_input_alpha = ( 0.95 - 0.05 ) * torch.rand( 1 ) + 0.05
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

