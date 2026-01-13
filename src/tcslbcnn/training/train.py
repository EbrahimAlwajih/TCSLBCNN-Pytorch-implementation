from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm

from tcslbcnn.eval import calc_accuracy
from tcslbcnn.models import TCSLBCNN, TCSLBCNNInitConfig


DatasetName = Literal["cifar10", "mnist"]


def _make_loaders(dataset: DatasetName, batch_size: int, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, int]:
    """
    Returns: train_loader, test_loader, n_input_plane
    """
    if dataset == "cifar10":
        n_input_plane = 3
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([28, 28]),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    elif dataset == "mnist":
        n_input_plane = 1
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            ]
        )
        train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader, n_input_plane


def train(
    *,
    dataset: DatasetName = "cifar10",
    n_epochs: int = 50,
    tcslbcnn_depth: int = 2,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    lr_scheduler_step: int = 30,
    run_dir: str = "artifacts/runs",
    checkpoint_path: str = "artifacts/models/tcslbcnn_best.pt",
    seed: Optional[int] = 1337,
    init_cfg: Optional[TCSLBCNNInitConfig] = None,
) -> None:
    """
    Train TCSLBCNN on MNIST or CIFAR10 (resized to 28x28).

    Artifacts:
        - TensorBoard logs: {run_dir}/...
        - Best checkpoint: {checkpoint_path}
    """
    start = time.time()

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, n_input_plane = _make_loaders(dataset, batch_size=batch_size)

    init_cfg = init_cfg or TCSLBCNNInitConfig(seed=seed)

    model = TCSLBCNN(depth=tcslbcnn_depth, n_input_plane=n_input_plane, init=init_cfg).to(device)

    # TensorBoard
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_dir)

    # Add graph (optional; can be slow / sometimes fails on some ops)
    try:
        example_inputs, _ = next(iter(test_loader))
        writer.add_graph(model, example_inputs.to(device))
    except Exception:
        pass

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step)

    best_accuracy = 0.0

    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(n_epochs):
        model.train(True)
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        acc_train = calc_accuracy(model, loader=train_loader, device=device, verbose=False)
        acc_test = calc_accuracy(model, loader=test_loader, device=device, verbose=False)

        writer.add_scalar("acc/train", acc_train, epoch)
        writer.add_scalar("acc/test", acc_test, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        print(f"Epoch {epoch+1} accuracy: train={acc_train:.4f}, test={acc_test:.4f}")

        if acc_test > best_accuracy:
            best_accuracy = acc_test
            torch.save(
                {
                    "depth": tcslbcnn_depth,
                    "n_input_plane": n_input_plane,
                    "dataset": dataset,
                    "init_cfg": asdict(init_cfg),
                    "state_dict": model.state_dict(),
                },
                ckpt_path,
            )

        scheduler.step()

    writer.close()

    train_duration_sec = int(time.time() - start)
    print(f"Finished Training. Total training time: {train_duration_sec} sec")


@torch.no_grad()
def test(checkpoint_path: str = "artifacts/models/tcslbcnn_best.pt", dataset: Optional[DatasetName] = None) -> None:
    """
    Evaluate a saved checkpoint. If dataset is not provided, use the dataset stored in checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    depth = int(ckpt["depth"])
    n_input_plane = int(ckpt["n_input_plane"])
    ds = dataset or ckpt.get("dataset", "cifar10")

    init_cfg = TCSLBCNNInitConfig(**ckpt.get("init_cfg", {}))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, _ = _make_loaders(ds, batch_size=256)

    model = TCSLBCNN(depth=depth, n_input_plane=n_input_plane, init=init_cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    acc = calc_accuracy(model, loader=test_loader, device=device, verbose=True)
    print(f"{ds.upper()} test accuracy: {acc:.4f}")
