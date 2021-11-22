"""
Hypernetwork ResUNet implementation.
The code is modified to incorporate the hypernetwork structure 
to the enhanced UNet implemented in the MONAI framework (https://monai.io/). 
The original implementation of ResUNet in the MONAI framework can be found in 
https://docs.monai.io/en/latest/_modules/monai/networks/nets/unet.html
"""

import sys

import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np

# Torch / Monai
import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, deprecated_arg, export

from monai.networks.layers.convutils import calculate_out_shape, same_padding

from monai.utils import (
    SkipMode,
    look_up_option,
)

# Hypernetwork modules
sys.path.append("..") 

from blocks.hyper_convolution import HyperConvolution, HyperResidualUnit

class HyperConvResUnit( nn.Module ):
    def __init__( self, hyperconv, hyper_res ):
        super().__init__()

        self.hyperconv = hyperconv
        self.hyper_res = hyper_res

    def forward( self, hyper_tensor, x ):
        x = self.hyperconv( hyper_tensor, x )
        x = self.hyper_res( hyper_tensor, x )

        return x

class HyperSkipConnection(nn.Module):
    """
    Skip Connection with HyperResConv/Hyperupsample blocks
    """
    def __init__(self, submodule, dim: int = 1, mode: Union[str, SkipMode] = "cat") -> None:
        super().__init__()
        self.submodule = submodule
        self.dim = dim
        self.mode = look_up_option(mode, SkipMode).value

    def forward(self, hyper_tensor: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if type( self.submodule ) == nn.ModuleList:
            y = x
            for _subsubmodule in self.submodule:
                y = _subsubmodule(hyper_tensor, y )
        else:
            y = self.submodule( hyper_tensor, x )

        if self.mode == "cat":
            return torch.cat([x, y], dim=self.dim)
        if self.mode == "add":
            return torch.add(x, y)
        if self.mode == "mul":
            return torch.mul(x, y)
        raise NotImplementedError(f"Unsupported mode {self.mode}.")

class HyperUNet(nn.Module):
    """
    Hypernetwork ResUNet implementation.
    The original implementation of ResUNet in MONAI framework can be found in 
    https://docs.monai.io/en/latest/_modules/monai/networks/nets/unet.html

    HyperUNet module takes all arguments from the original ResUNet implementation (see above). 

    ##############################################
    ##  Additional Parameters for Hypernetwork  ## 
    ##############################################
    hyper_n_invar: Number of input hyperparameters. 2 for the varying Tversky loss (alpha, beta). Default : 2.
    hyper_n_layers: Number of layers of the mapping network. Default: 5
    hyper_n_feat: Dimension of the mapped hypervector (output of the mapping network). It is set to hyper_n_invar * 2^hyper_n_laters. Default: 64
    hyper_use_activation: Flag to add ReLU activation layers after dense layers in the mapping netwrok. Default: True
    (Currently not supported) hyper_network: The pretrained mapping network, if available. Default: None
    """

    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        dimensions: Optional[int] = None,
        # Hypernetwork Parameters
        hyper_n_invar: int = 2,
        hyper_n_layers: int = 5,
        hyper_n_feat: int = 64,
        hyper_use_activation: bool = True,
        hyper_network: Optional[ nn.Sequential ] = None, 
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias

        ##########################################################################################
        #######                          Hypernet Mapping Network                          #######
        ##########################################################################################
        if hyper_network is not None:
            self.hyper_network = hyper_network
            # Not implmeneted
        elif hyper_n_layers == 0:
            self.hyper_n_invar = hyper_n_invar
            self.hyper_n_feat = hyper_n_feat
            self.hyper_n_layers = hyper_n_layers
            self.hyper_network = None
            self.hyper_use_activation = False

            print( "Setting no Mapping Network")
        else:
            self.hyper_n_invar = hyper_n_invar
            self.hyper_n_feat = hyper_n_feat
            self.hyper_n_layers = hyper_n_layers
            self.hyper_network = nn.Sequential()
            self.hyper_use_activation = hyper_use_activation

            # Setting up a hypernetwork
            for i in range( hyper_n_layers ):
                # First layer
                if i == 0:
                    self.hyper_network.add_module( f'hypernet_dense_{i+1}', nn.Linear( hyper_n_invar, int( hyper_n_feat / ( 2 ** ( hyper_n_layers - 1 - i ) ) ) ) )
                    if hyper_use_activation:
                        self.hyper_network.add_module( f'hypernet_relu_{i+1}', nn.ReLU( True ) )
                # Last layer
                elif i == hyper_n_layers - 1:
                    self.hyper_network.add_module( f'hypernet_dense_{i+1}', nn.Linear( int( hyper_n_feat / ( 2 ** ( hyper_n_layers - i ) ) ), int( hyper_n_feat / ( 2 ** ( hyper_n_layers - i - 1 ) ) ) ) )
                else:
                    self.hyper_network.add_module( f'hypernet_dense_{i+1}', nn.Linear( int( hyper_n_feat / ( 2 ** ( hyper_n_layers - i ) ) ), int( hyper_n_feat / ( 2 ** ( hyper_n_layers - i - 1 ) ) ) ) )
                    if hyper_use_activation:
                        self.hyper_network.add_module( f'hypernet_relu_{i+1}', nn.ReLU( True ) )

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool, hyper_n_feat: int=128
        ) -> nn.ModuleList:
            """
            Create HyperResConv or HyperUpsample blocks 
            instead of Residual/Upsample blocks in the original implementation 
            (https://docs.monai.io/en/latest/_modules/monai/networks/nets/unet.html).
            """
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False, hyper_n_feat=hyper_n_feat )  
                upc = c * 2
            else:
                subblock = self._get_bottom_layer(c, channels[1], hyper_n_feat = hyper_n_feat)
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top, hyper_n_feat = hyper_n_feat)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top, hyper_n_feat = hyper_n_feat)  # create layer in upsampling path

            return nn.ModuleList( [ down, HyperSkipConnection(subblock), up ] )

        self.model_module_list = _create_block(in_channels, out_channels, self.channels, self.strides, True, hyper_n_feat = self.hyper_n_feat )

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool, hyper_n_feat: int = 128) -> nn.Module:
        # For encoder parts
        # HyperResConv / HyperConvolution blocks
        mod: nn.Module
        if self.num_res_units > 0:

            mod = HyperResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                hyper_n_feat = hyper_n_feat,
            )
            return mod
        mod = HyperConvolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            hyper_n_feat = hyper_n_feat,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int, hyper_n_feat: int = 128) -> nn.Module:
        return self._get_down_layer(in_channels, out_channels, 1, False, hyper_n_feat = hyper_n_feat)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool, hyper_n_feat: int = 128 ) -> nn.Module:
        """
        For the decoder part: HyperConvTrans/HyperUpsample
        """
        conv: Union[HyperConvolution, HyperConvResUnit ]

        conv = HyperConvolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            hyper_n_feat = hyper_n_feat,
        )

        if self.num_res_units > 0:
            ru = HyperResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                hyper_n_feat = hyper_n_feat,
            )
            conv = HyperConvResUnit( conv, ru )

        return conv

    def forward(self, x: torch.Tensor, hyper_input: torch.Tensor ) -> torch.Tensor:
        """
        x : input image
        hyper_input: Input hyperparameters. e.g., alpha and beta for the varying Tversky loss.
        """
        if self.hyper_n_layers == 0:
            hyper_tensor = hyper_input
        else:
            hyper_tensor = self.hyper_network.forward( hyper_input )

        for _module in self.model_module_list:
            if type( _module ) == nn.ModuleList:
                for _submodule in _module:
                    x = _submodule( hyper_tensor, x )
            else: 
                x = _module( hyper_tensor, x )

        return x 