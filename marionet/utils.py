"""Loss functions."""
import torch as th
import torch.nn.functional as F
import numpy as np


class GaussianSmoothing(th.nn.Module):
    def __init__(self, channels, padding):
        super().__init__()
        self.padding = padding
        kernel = th.Tensor([[[1, 4, 6, 4, 1],
                             [4, 16, 24, 16, 4],
                             [6, 24, 36, 24, 6],
                             [4, 16, 24, 16, 4],
                             [1, 4, 6, 4, 1]]]).float() / 256.0
        kernel = kernel.expand(channels, 1, 5, 5)
        self.kernel = th.nn.Parameter(kernel, requires_grad=False)
        self.groups = channels

    def forward(self, input):
        input = F.pad(input, [self.padding]*4, mode='reflect')
        return F.conv2d(input, self.kernel, groups=self.groups)


class PyramidLoss(th.nn.Module):
    def __init__(self, num_chans, base_loss="l1"):
        super().__init__()
        self.smoothing = GaussianSmoothing(num_chans, 2)
        if base_loss not in ["l1", "l2"]:
            raise ValueError(f"Unkown base loss: {base_loss}")
        self.base_loss = base_loss

    def forward(self, pred, target):
        canvas_size = pred.shape[-1]
        loss = 0
        n_levels = int(np.log2(canvas_size))
        for idx in range(n_levels):
            if self.base_loss == "l1":
                loss += th.abs(pred-target).sum(1).mean()
            else:
                loss += (pred-target).pow(2).sum(1).mean()

            if idx < n_levels-1:  # prepare for next step
                pred = self.smoothing(pred)
                target = self.smoothing(target)
                pred = F.avg_pool2d(pred, 2)
                target = F.avg_pool2d(target, 2)

        return loss / n_levels


def make_checkers(size):
    checkers = th.ones(1, 3, size // 8, 8, size // 8, 8)
    checkers[:, :, ::2, :, ::2] = 0.8
    checkers[:, :, 1::2, :, 1::2] = 0.8
    return checkers.flatten(2, 3).flatten(3, 4)
