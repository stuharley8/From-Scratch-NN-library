from unittest import TestCase
import numpy as np
import unittest
import layers
import torch


class TestSoftmaxCrossEntropy(TestCase):

    def setUp(self):
        self.pred = layers.Input(4, True)
        self.pred.set(torch.tensor([1,-2,0,3], dtype=torch.float64))
        self.true = layers.Input(4, True)
        self.true.set(torch.tensor([0,0,0,1], dtype=torch.float64))
        self.softmaxce = layers.SoftmaxCrossEntropy(self.pred, self.true)
        self.softmaxce.set_grad(torch.ones(1))

    def test_forward(self):
        self.softmaxce.forward()
        np.testing.assert_allclose(self.softmaxce.output.detach().numpy(), np.array([.1755153626]))

    def test_backward(self):
        self.softmaxce.backward()
        np.testing.assert_allclose(self.pred.grad.detach().numpy(), np.array([.1135496194, .0056533027, .0417725705, -.1609754925]))
        np.testing.assert_allclose(self.true.grad.detach().numpy(), np.zeros([4]))


if __name__ == '__main__':
    unittest.main()
