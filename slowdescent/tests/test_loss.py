import torch
import numpy as np

from slowdescent.layers import loss


def test_mse_loss():
    input  = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    
    got      = loss.MSE()(input, target)
    expected = torch.nn.MSELoss()(input, target).data

    np.testing.assert_array_equal(got, expected)
