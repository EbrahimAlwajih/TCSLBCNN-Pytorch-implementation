from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class TCSLBCNNInitConfig:
    sparsity: float = 0.5
    thresholds: Sequence[float] = (1.0, 0.5, 0.75, 0.25)
    seed: Optional[int] = None
    # Preserve original behavior: skip channel index 0 and output index 0 in the random init loops.
    skip_zero_indices: bool = True


class ConvTCSLBP(nn.Conv2d):
    """
    A Conv2d layer whose weights are initialized to a sparse / ternary-like pattern and then frozen.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        init: TCSLBCNNInitConfig = TCSLBCNNInitConfig(),
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

        # Initialize and freeze weights
        self._init_weights(init)
        for p in self.parameters():
            p.requires_grad_(False)

    def _init_weights(self, init: TCSLBCNNInitConfig) -> None:
        rng = np.random.default_rng(init.seed)

        with torch.no_grad():
            weights = self.weight  # shape: [out, in, k, k]
            out_ch, in_ch, kH, kW = weights.shape
            assert kH == kW, "Only square kernels are supported in this initializer."

            # Flatten kernel positions to index within k*k.
            # Original code uses indices of a 3x3 neighborhood excluding center (index 4).
            # For 3x3: indices 0..8, center=4.
            kk = kH * kW
            if kk != 9:
                # For now we keep behavior consistent with original 3x3 implementation.
                raise ValueError("This ConvTCSLBP initializer currently supports only kernel_size=3.")

            matrix = torch.zeros((out_ch, in_ch, kk), dtype=weights.dtype, device=weights.device)

            index1 = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.int64)

            in_start = 1 if init.skip_zero_indices else 0
            out_start = 1 if init.skip_zero_indices else 0

            for ic in range(in_start, in_ch):
                rng.shuffle(index1)
                for oc in range(out_start, out_ch):
                    thr = init.thresholds[int(rng.integers(0, len(init.thresholds)))]
                    rand_idx = int(rng.integers(0, len(index1)))  # 0..7
                    pos = int(index1[rand_idx])
                    neg = int(8 - pos)
                    matrix[oc, ic, pos] = float(thr)
                    matrix[oc, ic, neg] = -float(thr)

            matrix = matrix.view(out_ch, in_ch, kH, kW)
            weights.copy_(matrix)


class TCSLBPBlock(nn.Module):
    def __init__(self, num_channels: int, num_weights: int, init: TCSLBCNNInitConfig):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_channels)
        self.tcslb = ConvTCSLBP(num_channels, num_weights, kernel_size=3, init=init)
        self.relu = nn.ReLU()
        self.conv_1x1 = nn.Conv2d(num_weights, num_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.bn(x)
        x = self.tcslb(x)
        x = self.relu(x)
        x = self.conv_1x1(x)
        x = x + residual
        return x


class TCSLBCNN(nn.Module):
    """
    TCS-LBCNN for MNIST by default.
    """

    def __init__(
        self,
        n_input_plane: int = 1,
        num_channels: int = 16,
        num_weights: int = 256,
        full: int = 50,
        depth: int = 2,
        init: TCSLBCNNInitConfig = TCSLBCNNInitConfig(),
        num_classes: int = 10,
    ):
        super().__init__()

        self.preprocess_block = nn.Sequential(
            nn.Conv2d(n_input_plane, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )

        self.chained_blocks = nn.Sequential(
            *[TCSLBPBlock(num_channels, num_weights, init=init) for _ in range(depth)]
        )

        # For MNIST 28x28: after blocks, AvgPool 5/5 gives 5x5 (as in original).
        self.pool = nn.AvgPool2d(kernel_size=5, stride=5)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(num_channels * 5 * 5, full)  # MNIST
        # self.fc1 = nn.Linear(num_channels * 6 * 6, full)  # CIFAR (example)
        self.fc2 = nn.Linear(full, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess_block(x)
        x = self.chained_blocks(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
