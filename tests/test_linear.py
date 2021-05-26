from unittest import TestCase
import numpy as np
import unittest
import layers
import torch


class TestLinear(TestCase):

    def setUp(self):
        self.x = layers.Input(3, False)
        self.x.set(torch.tensor([1,2,3], dtype=torch.float64))
        self.W = layers.Input([2,3], True)
        self.W.set(torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float64))
        self.b = layers.Input(2, True)
        self.b.set(torch.tensor([100,1000], dtype=torch.float64))
        self.linear = layers.Linear(self.x, self.W, self.b)
        self.linear.set_grad(torch.ones(2, dtype=torch.float64))

    def test_forward(self):
        self.linear.forward()
        np.testing.assert_allclose(self.linear.output.detach().numpy(), np.array([114, 1032]))

    def test_accumulate_grad(self):
        # accumulate_grad is defined in Layer so it only needs to be tested once
        self.x.accumulate_grad(torch.tensor([.1, .2, .3]))
        np.testing.assert_allclose(self.x.grad.detach().numpy(), np.array([.1, .2, .3]))

    def test_backward(self):
        self.linear.backward()
        np.testing.assert_allclose(self.x.grad.detach().numpy(), np.array([5,7,9]))
        np.testing.assert_allclose(self.W.grad.detach().numpy(), np.array([[1,2,3],[1,2,3]]))
        np.testing.assert_allclose(self.b.grad.detach().numpy(), np.ones(2))

    def test_step(self):
        self.linear.forward()
        self.linear.backward()
        self.linear.step(10)  # should not have any affect on linear.output
        self.x.step(.1)
        self.W.step(.2)
        self.b.step(.3)
        np.testing.assert_allclose(self.linear.output.detach().numpy(), np.array([114, 1032]))
        np.testing.assert_allclose(self.x.output.detach().numpy(), np.array([1,2,3]))
        np.testing.assert_allclose(self.W.output.detach().numpy(), np.array([[.8,1.6,2.4],[3.8,4.6,5.4]]))
        np.testing.assert_allclose(self.b.output.detach().numpy(), np.array([99.7,999.7]))


if __name__ == '__main__':
    unittest.main()
