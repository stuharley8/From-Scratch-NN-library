from unittest import TestCase
import numpy as np
import unittest
import layers
import torch


class TestFlatten(TestCase):

    def setUp(self):
        self.input = layers.Input([2,3], True)
        self.input.set(torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32))
        self.flatten = layers.Flatten(self.input)
        self.flatten.set_grad(torch.ones(6, dtype=torch.float32))

    def test_forward(self):
        self.flatten.forward()
        np.testing.assert_allclose(self.flatten.output.detach().numpy(), np.array([1,2,3,4,5,6]))

    def test_backward(self):
        self.flatten.backward()
        np.testing.assert_allclose(self.input.grad.detach().numpy(), np.ones([2,3]))


if __name__ == '__main__':
    unittest.main()
