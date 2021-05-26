from unittest import TestCase
import numpy as np
import unittest
import layers
import torch


class TestL2Loss(TestCase):

    def setUp(self):
        self.pred = layers.Input(2, True)
        self.pred.set(torch.tensor([3,5],dtype=torch.float64))
        self.true = layers.Input(2, True)
        self.true.set(torch.tensor([3.4,4.9],dtype=torch.float64))
        self.l2loss = layers.L2Loss(self.pred, self.true)
        self.l2loss.set_grad(torch.ones(1))

    def test_forward(self):
        self.l2loss.forward()
        np.testing.assert_allclose(self.l2loss.output.detach().numpy(), np.array([.2061552813]))

    def test_backward(self):
        self.l2loss.backward()
        np.testing.assert_allclose(self.pred.grad.detach().numpy(), np.array([-.4,.1]))
        np.testing.assert_allclose(self.true.grad.detach().numpy(), np.array([-.4,.1]))


if __name__ == '__main__':
    unittest.main()
