import torch

from tcslbcnn.models import TCSLBCNN, TCSLBCNNInitConfig


def test_forward_shape():
    model = TCSLBCNN(depth=2, n_input_plane=1, init=TCSLBCNNInitConfig(seed=123))
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    assert y.shape == (4, 10)
