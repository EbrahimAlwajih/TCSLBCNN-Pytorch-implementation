from __future__ import annotations

from typing import Optional

import torch
from tqdm import tqdm


@torch.no_grad()
def calc_accuracy(
    model: torch.nn.Module, loader, device: Optional[torch.device] = None, verbose: bool = False
) -> float:
    """
    Compute classification accuracy.

    Notes:
        - This function will NOT move the model to a device. Pass `device` to move inputs/labels.
        - Temporarily switches the model to eval mode and restores the previous mode.

    Args:
        model: A PyTorch model.
        loader: DataLoader yielding (inputs, labels).
        device: torch.device to run on (e.g. torch.device("cuda")). If None, uses inputs as-is.
        verbose: Show tqdm progress bar.

    Returns:
        Accuracy as float in [0, 1].
    """
    was_training = model.training
    model.eval()

    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Eval", total=len(loader), disable=not verbose):
        if device is not None:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.numel()

    if was_training:
        model.train(True)

    return correct / float(total) if total > 0 else 0.0
