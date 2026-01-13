from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

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
        """
        Fast initializer: for each (oc, ic) picks:
        - a threshold from init.thresholds
        - a position from the 8 neighbors (excluding center)
        - sets +thr at pos and -thr at opposite (8-pos)
        Preserves skip_zero_indices behavior.

        Note: This matches the *pattern* of the original initializer.
        It will not match NumPy's exact random sequence bit-for-bit.
        """
        with torch.no_grad():
            weights = self.weight  # [out, in, 3, 3]
            out_ch, in_ch, kH, kW = weights.shape
            if (kH, kW) != (3, 3):
                raise ValueError("This initializer currently supports only kernel_size=3.")

            # Start with all zeros
            weights.zero_()

            # Determine active ranges (preserve original skip behavior)
            in_start = 1 if init.skip_zero_indices else 0
            out_start = 1 if init.skip_zero_indices else 0
            if in_start >= in_ch or out_start >= out_ch:
                return  # nothing to fill

            # Torch RNG (fast, can run on GPU)
            g = torch.Generator(device=weights.device)
            if init.seed is not None:
                g.manual_seed(int(init.seed))

            # Valid neighbor positions in flattened 3x3 excluding center (index 4)
            # indices: 0 1 2 / 3 4 5 / 6 7 8
            index1 = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8], device=weights.device)

            oc_count = out_ch - out_start
            ic_count = in_ch - in_start

            # Choose random threshold index per (oc, ic)
            thr_idx = torch.randint(
                low=0,
                high=len(init.thresholds),
                size=(oc_count, ic_count),
                generator=g,
                device=weights.device,
            )

            thresholds = torch.tensor(
                list(init.thresholds), dtype=weights.dtype, device=weights.device
            )
            thr = thresholds[thr_idx]  # [oc_count, ic_count]

            # Choose random neighbor index (0..7) per (oc, ic)
            r = torch.randint(
                low=0,
                high=len(index1),
                size=(oc_count, ic_count),
                generator=g,
                device=weights.device,
            )
            pos = index1[r]  # [oc_count, ic_count] in {0,1,2,3,5,6,7,8}
            neg = 8 - pos  # opposite position

            # Convert flattened indices to (row, col)
            pos_r, pos_c = pos // 3, pos % 3
            neg_r, neg_c = neg // 3, neg % 3

            # Build index grids for oc, ic
            oc_idx = (
                torch.arange(out_start, out_ch, device=weights.device)
                .view(-1, 1)
                .expand(oc_count, ic_count)
            )
            ic_idx = (
                torch.arange(in_start, in_ch, device=weights.device)
                .view(1, -1)
                .expand(oc_count, ic_count)
            )

            # Scatter +thr and -thr into weights
            weights[oc_idx, ic_idx, pos_r, pos_c] = thr
            weights[oc_idx, ic_idx, neg_r, neg_c] = -thr


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
