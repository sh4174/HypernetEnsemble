"""
Tversky loss function for hypernetwork with hyperparameters as additional inputs
Modified from losses.tversky implementation from the MONAI framework
https://docs.monai.io/en/latest/_modules/monai/losses/tversky.html

Modification:
Made forward function to take hyperparameters (alpha and beta) as inputs instead of fixing them at the construction.
def forward(self, input: torch.Tensor, target: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor ) -> torch.Tensor:

The rest of the implementation is identical to the original implementation (see the above link)
"""

import warnings
from typing import Callable, List, Optional, Union

import torch
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.utils import LossReduction


class TverskyLoss_Modified(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def forward(self, input: torch.Tensor, target: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor ) -> torch.Tensor:
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        p0 = input
        p1 = 1 - p0
        g0 = target
        g1 = 1 - g0

        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            reduce_axis = [0] + reduce_axis

        tp = torch.sum(p0 * g0, reduce_axis)
        fp = alpha * torch.sum(p0 * g1, reduce_axis)
        fn = beta * torch.sum(p1 * g0, reduce_axis)
        numerator = tp + self.smooth_nr
        denominator = tp + fp + fn + self.smooth_dr

        score: torch.Tensor = 1.0 - numerator / denominator

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(score)  
        if self.reduction == LossReduction.NONE.value:
            return score  
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(score)
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')