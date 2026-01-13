"""Temporary backwards-compatible shim.

Keeps legacy imports working while code lives under the `tcslbcnn` package.
"""

from tcslbcnn.data import get_mnist_loader
from tcslbcnn.eval import calc_accuracy

__all__ = ["calc_accuracy", "get_mnist_loader"]
