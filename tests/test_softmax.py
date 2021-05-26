from unittest import TestCase
import numpy as np
import unittest
import layers
import torch


class TestSoftmax(TestCase):

    def setUp(self):
        self.input = layers.Input(4, True)
        self.input.set(torch.tensor([1,-2,0,3], dtype=torch.float64))
        self.softmax = layers.Softmax(self.input)

    def test_forward(self):
        self.softmax.forward()
        np.testing.assert_allclose(self.softmax.output.detach().numpy(), np.array([.1135496194,.0056533027,.0417725705, .8390245075]))


if __name__ == '__main__':
    unittest.main()
