from unittest import TestCase
import numpy as np
import unittest
import layers
import torch


class TestReLU(TestCase):

    def setUp(self):
        self.input = layers.Input(5, True)
        self.input.set(torch.tensor([3,-3,0,-.0001,1213324.34232], dtype=torch.float64))
        self.relu = layers.ReLU(self.input)
        self.relu.set_grad(torch.ones(5))

    def test_forward(self):
        self.relu.forward()
        np.testing.assert_allclose(self.relu.output.detach().numpy(), np.array([3,0,0,0,1213324.34232]))

    def test_backward(self):
        self.relu.backward()
        np.testing.assert_allclose(self.input.grad.detach().numpy(), np.array([1,0,0,0,1]))


if __name__ == '__main__':
    unittest.main()
