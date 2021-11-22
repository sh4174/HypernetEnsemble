"""
The implementation of HyperConvolution/HyperResConv Blocks

The overall convolution implementation is based on the blocks.convolutions in the MONAI framework (https://monai.io/):
https://docs.monai.io/en/latest/_modules/monai/networks/blocks/convolutions.html
The hypernetwork convolution is adapted from the HyperMorph (http://ftp.nmr.mgh.harvard.edu/pub/docs/hypermorph/):
https://github.com/voxelmorph/voxelmorph
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import ADN
from monai.networks.layers.convutils import same_padding, stride_minus_kernel_padding
from monai.networks.layers.factories import Conv

class HyperSequential(nn.Sequential):
    def forward(self, hyper_tensor, x ):
        for module in self._modules.values():
            hyper_tensor, x = module( hyper_tensor, x )
        return hyper_tensor, x 

class HyperConvolution(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        adn_ordering: str = "NDA",
        act: Optional[Union[Tuple, str]] = "PRELU",
        norm: Optional[Union[Tuple, str]] = "INSTANCE",
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = 1,
        dilation: Union[Sequence[int], int] = 1,
        groups: int = 1,
        bias: bool = True,
        conv_only: bool = False,
        is_transposed: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
        output_padding: Optional[Union[Sequence[int], int]] = None,
        dimensions: Optional[int] = None,
        # Hyper-network parameters 
        hyper_n_feat: int = 128,
    ) -> None:
        super().__init__()
        self.dimensions = spatial_dims if dimensions is None else dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_transposed = is_transposed
        if padding is None:
            padding = same_padding(kernel_size, dilation)
            
        conv_type = Conv[Conv.CONVTRANS if is_transposed else Conv.CONV, self.dimensions]

        # Setting up convolutoin kernel
        conv: nn.Module
        if is_transposed:
            if output_padding is None:
                output_padding = stride_minus_kernel_padding(1, strides)
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                bias=bias,
                dilation=dilation,
            )
        else:
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )

        self.strides = strides
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

        # Hyper-convolution weight. It will be estimated by the conv-level sub-hypernetwork
        self.weight_shape = conv.weight.shape
        self.weight_length = list( torch.flatten( conv.weight ).shape )[ 0 ]
        self.bias_shape = conv.bias.shape
        self.bias_length = list( torch.flatten( conv.bias ).shape )[ 0 ]

        self.weight = torch.Tensor( conv.weight.shape )
        self.bias = torch.Tensor( conv.bias.shape )

        # Set up the conv-level sub-hypernetwork for embedding
        self.sub_hyper_network_weight = nn.Linear( hyper_n_feat, self.weight_length )
        self.sub_hyper_network_bias = nn.Linear( hyper_n_feat, self.bias_length )

        if not conv_only:
            self.adn = ADN(
                        ordering=adn_ordering,
                        in_channels=out_channels,
                        act=act,
                        norm=norm,
                        norm_dim=self.dimensions,
                        dropout=dropout,
                        dropout_dim=dropout_dim,
                    )
        else:
            self.adn = None

    def forward( self, hyper_tensor: torch.Tensor, x: torch.Tensor ):
        if self.is_transposed:
            func_conv = getattr( F, f'conv_transpose{self.dimensions}d' )
        else:
            func_conv = getattr( F, f'conv{self.dimensions}d' )

        # Estimate weight
        weight_flat = self.sub_hyper_network_weight( hyper_tensor )

        # For training hypernetwork : one hypertensor for multiple images
        if weight_flat.shape[ 0 ] == 1 or len( list( weight_flat.shape ) ) == 1: 
            weight = weight_flat.reshape( self.weight_shape )

            # Estimate bias
            if self.use_bias:
                bias_flat = self.sub_hyper_network_bias( hyper_tensor )
                bias = bias_flat.reshape( self.bias_shape )

            # Convolve
            if self.use_bias:
                if self.is_transposed:
                    output_padding = stride_minus_kernel_padding(1, self.strides)
                    x = func_conv( x, weight, bias, stride=self.strides, padding=self.padding, dilation=self.dilation, groups = self.groups, output_padding=output_padding )
                else:
                    x = func_conv( x, weight, bias, stride=self.strides, padding=self.padding, dilation=self.dilation, groups = self.groups )
            else:
                if self.is_transposed:
                    output_padding = stride_minus_kernel_padding(1, self.strides)
                    x = func_conv( x, weight, stride=self.strides, padding=self.padding, dilation=self.dilation, groups = self.groups, output_padding=output_padding ) 
                else:
                    x = func_conv( x, weight, stride=self.strides, padding=self.padding, dilation=self.dilation, groups = self.groups ) 
        # For traing auxiliary network : one hyperparameter per one image
        # Currently not supported
        else:
            weights = weight_flat.reshape( [ weight_flat.shape[ 0 ] ] + list( self.weight_shape ) )

            # Estimate bias
            if self.use_bias:
                bias_flat = self.sub_hyper_network_bias( hyper_tensor )
                biases = bias_flat.reshape( [bias_flat.shape[ 0 ] ] + list( self.bias_shape ) )
            x1 = []

            # Convolve
            if self.use_bias:
                if self.is_transposed:
                    output_padding = stride_minus_kernel_padding(1, self.strides)
                    for i in range( weight_flat.shape[ 0 ] ):
                        x_i = func_conv( x[ i:i+1 ], weights[ i ], biases[ i ], stride=self.strides, padding=self.padding, dilation=self.dilation, groups = self.groups, output_padding=output_padding )
                        x1.append( x_i ) 
                else:
                    for i in range( weight_flat.shape[ 0 ] ):
                        x_i = func_conv( x[ i:i+1 ], weights[ i ], biases[ i ], stride=self.strides, padding=self.padding, dilation=self.dilation, groups = self.groups )
                        x1.append( x_i ) 
            else:
                if self.is_transposed:
                    output_padding = stride_minus_kernel_padding(1, self.strides)
                    for i in range( weight_flat.shape[ 0 ] ):
                        x_i = func_conv( x[ i:i+1 ], weight[ i ], stride=self.strides, padding=self.padding, dilation=self.dilation, groups = self.groups, output_padding=output_padding ) 
                        x1.append( x_i )
                else:
                    for i in range( weight_flat.shape[ 0 ] ):
                        x_i = func_conv( x[ i:i+1 ], weight[ i ], stride=self.strides, padding=self.padding, dilation=self.dilation, groups = self.groups ) 
                        x1.append( x_i )

            x = torch.cat( x1, 0 )

        # ADN 
        if self.adn is not None:
            x = self.adn( x )

        return x

class HyperResidualUnit(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        subunits: int = 2,
        adn_ordering: str = "NDA",
        act: Optional[Union[Tuple, str]] = "PRELU",
        norm: Optional[Union[Tuple, str]] = "INSTANCE",
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = 1,
        dilation: Union[Sequence[int], int] = 1,
        bias: bool = True,
        last_conv_only: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
        dimensions: Optional[int] = None,
        # Hyper-network
        hyper_n_feat: int = 128,
    ) -> None:
        super().__init__()
        self.dimensions = spatial_dims if dimensions is None else dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.ModuleList()
        self.residual = nn.Identity()
        self.is_res_identity = True
        if not padding:
            padding = same_padding(kernel_size, dilation)
        schannels = in_channels
        sstrides = strides
        subunits = max(1, subunits)

        for su in range(subunits):
            conv_only = last_conv_only and su == (subunits - 1)
            unit = HyperConvolution(
                self.dimensions,
                schannels,
                out_channels,
                strides=sstrides,
                kernel_size=kernel_size,
                adn_ordering=adn_ordering,
                act=act,
                norm=norm,
                dropout=dropout,
                dropout_dim=dropout_dim,
                dilation=dilation,
                bias=bias,
                conv_only=conv_only,
                padding=padding,
                hyper_n_feat=hyper_n_feat,
            )

            self.conv.append( unit )

            schannels = out_channels
            sstrides = 1

        if np.prod(strides) != 1 or in_channels != out_channels:
            rkernel_size = kernel_size
            rpadding = padding

            if np.prod(strides) == 1: 
                rkernel_size = 1
                rpadding = 0

            self.residual = HyperConvolution( self.dimensions, in_channels, out_channels, strides, rkernel_size, padding=rpadding, bias=bias, conv_only=True, hyper_n_feat=hyper_n_feat )
            self.is_res_identity = False        

    def forward(self, hyper_tensor: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.is_res_identity:
            res = self.residual( x )
        else:
            res = self.residual(hyper_tensor, x) 
        
        cx = x 
        for _module in self.conv:
            cx = _module( hyper_tensor, cx )

        return cx + res 

