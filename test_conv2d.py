from unittest import TestCase
import numpy as np
import unittest
import torch
import conv2d


class TestMultilayerConv(TestCase):
    def test_conv_forward(self):
        image = torch.tensor([[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
                              [[4, 3, 2, 1], [5, 4, 3, 2], [6, 5, 4, 3], [7, 6, 5, 4], [8, 7, 6, 5]]],
                             dtype=torch.float32)
        filter = torch.tensor(
            [[[[-1, 0, 0], [0, 0, 0], [0, 0, 1]],
              [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]],
             [[[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
              [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]],
             [[[0, 1, 0], [0, 0, 0], [0, -1, 0]],
              [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]]], dtype=torch.float32)
        map = conv2d.conv_filter_forward(image,filter)
        np.testing.assert_allclose(map.detach().numpy(),
                                   np.array([[[0,0],
                                              [0,0],
                                              [0,0]],
                                             [[7,7],
                                              [9,9],
                                              [11,11]],
                                             [[-4,-4],
                                              [-4,-4],
                                              [-4,-4]]],dtype=np.float32),
                                   rtol=2e-7)

    def test_conv_forward_pad(self):
        image = torch.tensor([[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
                              [[4, 3, 2, 1], [5, 4, 3, 2], [6, 5, 4, 3], [7, 6, 5, 4], [8, 7, 6, 5]]],
                             dtype=torch.float32)
        filter = torch.tensor(
            [[[[-1, 0, 0], [0, 0, 0], [0, 0, 1]],
              [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]],
             [[[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
              [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]],
             [[[0, 1, 0], [0, 0, 0], [0, -1, 0]],
              [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]]], dtype=torch.float32)
        map = conv2d.conv_filter_forward(image,filter,padding=1)
        np.testing.assert_allclose(map.detach().numpy(),
                                   np.array([[[3,-1,1,-3],
                                              [7,0,0,-7],
                                              [9,0,0,-9],
                                              [11,0,0,-11],
                                              [6,1,-1,-6]],
                                             [[24/9,4,4,24/9], # This and the following channel
                                                  # have pad values derived from the test result.
                                              [42/9,7,7,42/9],
                                              [6,9,9,6],
                                              [66/9,11,11,66/9],
                                              [48/9,8,8,48/9]],
                                             [[1,-5,-6,-7],
                                              [2,-4,-4,-5],
                                              [3,-4,-4,-6],
                                              [4,-4,-4,-7],
                                              [11,3,4,1]]],dtype=np.float32),
                                    rtol=2e-7)

    def test_conv_backward(self):
        image = torch.tensor([[[0, 1], [0, 0], [0, 0]],
                              [[0, 0], [0, 0], [1, 0]],
                              [[0, 0], [0, 1], [0, 0]]], dtype=torch.float32)
        filter = torch.tensor(
            [[[[1, 0], [0, -1]],
              [[0, -1], [1, 0]]],
             [[[1/4, 1/4], [1/4, 1/4]],
              [[1/4, 1/4], [1/4, 1/4]]],
             [[[0, -1], [0, 1]],
              [[0, 0], [-1, 1]]]], dtype=torch.float32)
        map = conv2d.conv_filter_backward(image, filter)
        np.testing.assert_allclose(map.detach().numpy(),
                                   np.array([[[[1],
                                               [-3/4]],
                                              [[0],
                                               [1/4]]]],dtype=np.float32))

    def test_conv_backward_pad(self):
        image = torch.tensor([[[0, 1], [0, 0], [0, 0]],
                              [[0, 0], [0, 0], [1, 0]],
                              [[0, 0], [0, 1], [0, 0]]], dtype=torch.float32)
        filter = torch.tensor(
            [[[[1, 0, 0], [0, 0, 0], [0, 0, -1]],
              [[0, 0, -1], [0, 0, 0], [1, 0, 0]]],
             [[[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
              [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]],
             [[[0, -1, 0], [0, 0, 0], [0, 1, 0]],
              [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]]], dtype=torch.float32)
        map = conv2d.conv_filter_backward(image, filter, padding=2)
        np.testing.assert_allclose(map.detach().numpy(),
                                   np.array([[[[0,-1,0,0],
                                               [0,0,1,0],
                                               [1/9,1/9,1/9,1],
                                               [1/9,1/9,-8/9,0],
                                               [1/9,1/9,1/9,0]],
                                              [[0,0,0,1],
                                               [0,0,0,0],
                                               [1/9,1/9,1/9,-1],
                                               [1/9,1/9,1/9,0],
                                               [1/9,1/9,1/9,0]]]],dtype=np.float32))

    def test_conv_expand(self):
        image = torch.tensor([[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
                              [[4, 3, 2, 1], [5, 4, 3, 2], [6, 5, 4, 3], [7, 6, 5, 4], [8, 7, 6, 5]]],
                             dtype=torch.float32)
        filter = torch.tensor([[[0, 1], [0, 0], [0, 0]],
                               [[0, 0], [0, 0], [1, 0]],
                               [[0, 0], [0, 1], [0, 0]]], dtype=torch.float32)
        map = conv2d.conv_expand_layers(image,filter)
        np.testing.assert_allclose(map.detach().numpy(),
                                   np.array([[[[2,3,4],[3,4,5],[4,5,6]],
                                              [[3,2,1],[4,3,2],[5,4,3]]],
                                             [[[3,4,5],[4,5,6],[5,6,7]],
                                              [[6,5,4],[7,6,5],[8,7,6]]],
                                             [[[3,4,5],[4,5,6],[5,6,7]],
                                              [[4,3,2],[5,4,3],[6,5,4]]]],dtype=np.float32))

    def test_conv_expand_pad(self):
        image = torch.tensor([[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
                              [[4, 3, 2, 1], [5, 4, 3, 2], [6, 5, 4, 3], [7, 6, 5, 4], [8, 7, 6, 5]]],
                             dtype=torch.float32)
        filter = torch.tensor([[[0, 1], [0, 0], [0, 0]],
                               [[0, 0], [0, 0], [1, 0]],
                               [[0, 0], [0, 1], [0, 0]]], dtype=torch.float32)
        map = conv2d.conv_expand_layers(image, filter, padding=1)
        np.testing.assert_allclose(map.detach().numpy(),
             np.array([[[[0,0,0,0,0],[1,2,3,4,0],[2,3,4,5,0],[3,4,5,6,0],[4,5,6,7,0]],
                        [[0,0,0,0,0],[4,3,2,1,0],[5,4,3,2,0],[6,5,4,3,0],[7,6,5,4,0]]],
                       [[[0,2,3,4,5],[0,3,4,5,6],[0,4,5,6,7],[0,5,6,7,8],[0,0,0,0,0]],
                        [[0,5,4,3,2],[0,6,5,4,3],[0,7,6,5,4],[0,8,7,6,5],[0,0,0,0,0]]],
                       [[[1,2,3,4,0],[2,3,4,5,0],[3,4,5,6,0],[4,5,6,7,0],[5,6,7,8,0]],
                        [[4,3,2,1,0],[5,4,3,2,0],[6,5,4,3,0],[7,6,5,4,0],[8,7,6,5,0]]]],
                       dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
