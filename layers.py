import torch
import conv2d

# Please be sure to read the comments in the main lab and think about your design before
# you begin to implement this part of the lab.

# Layers in this file are arranged in roughly the order they
# would appear in a network.


class Layer:
    def __init__(self, output_shape):
        """
        Initializes the output instance variable as a torch tensor of all zeros of the specified shape.
        """
        self.output = torch.zeros(output_shape)
        self.grad = torch.zeros(output_shape)

    def get_output(self):
        """
        Returns the output tensor of the layer
        """
        return self.output

    def get_grad(self):
        """
        Returns the grad tensor of the layer
        """
        return self.grad

    def set_grad(self, grad):
        """
        Sets the grad tensor to something other than zeros. Useful for testing and setting the grad of the last layer
        in a network to 1.
        """
        assert self.grad.shape == grad.shape, "grad shape does not match self.grad"
        self.grad = grad

    def accumulate_grad(self, gradient):
        """
        Adds the gradient it receives to the Layer's self.grad
        """
        assert self.grad.shape == gradient.shape, "gradient shape does not match self.grad"
        self.grad = self.grad + gradient

    def clear_grad(self):
        """
        Clears the grad tensor by resetting the values to zero
        """
        self.grad = self.grad * 0

    def step(self, step_size):
        """
        Performs a single step of stochastic gradient descent, updating the weights of the current layer
        based on the step_size parameter and the current gradients.

        pass in all layers that are not an Input layer
        """
        pass


class Input(Layer):
    def __init__(self, output_shape, train):
        """
        Initializes the output instance variable as a torch tensor of all zeros of the specified shape.
        Train is a boolean flag that indicate whether the Input layer is trainable
        """
        Layer.__init__(self, output_shape)
        self.train = train

    def set(self, output):
        """
        Set the output of this input layer.
        :param output: The output to set, as a numpy array. Raise an error if this output's size
                       would change.
        """
        assert self.output.shape == output.shape, 'incorrect shape of value'
        self.output = output

    def randomize(self):
        """
        Set the output of this input layer to random values sampled from the standard normal
        distribution (numpy has a nice method to do this). Ensure that the output does not
        change size.
        """
        self.output = torch.randn(self.output.shape, dtype=torch.float32)

    def forward(self):
        """
        Set this layer's output based on the outputs of the layers that feed into it.
        """
        pass # This layer's values do not change during forward propogation since no layers feed into it

    def backward(self):
        pass

    def step(self, step_size):
        """
        Performs a single step of stochastic gradient descent, updating the weights of the current layer
        based on the step_size parameter and the current gradients.

        pass in all layers that are not an Input layer
        """
        if self.train:
            self.output = self.output - (step_size * self.grad)


class Linear(Layer):
    def __init__(self, x_layer, W_layer, b_layer):
        """
        Accepts an input_layer (x), a weight layer (W), and a bias layer (b)
        """
        Layer.__init__(self, b_layer.output.shape)
        self.x = x_layer
        self.W = W_layer
        self.b = b_layer

    def forward(self):
        """
        Set this layer's output based on the outputs of the layers that feed into it. (Wx + b).
        """
        self.output = self.W.output @ self.x.output + self.b.output

    def backward(self):
        """
        Performs back-propagation through the single layer. Assumes self.grad already contains dJ/output.
        (The derivative of the objective function with respect to this layer's output).
        Accumulates the calculated gradients into the gradients of the previous layers.
        """
        dJ_dx = self.W.output.T @ self.grad
        dJ_dW = torch.outer(self.grad, self.x.output)
        dJ_db = self.grad
        self.x.accumulate_grad(dJ_dx)
        self.W.accumulate_grad(dJ_dW)
        self.b.accumulate_grad(dJ_db)


class ReLU(Layer):
    def __init__(self, input_layer):
        """
        Takes in a layer that will have a ReLU activation function applied
        """
        Layer.__init__(self, input_layer.output.shape)
        self.input = input_layer

    def forward(self):
        """
        Set this layer's output based on the outputs of the layers that feed into it. (Applies the ReLU function).
        """
        self.output = self.input.output * (self.input.output > 0)

    def backward(self):
        """
        Performs back-propagation through the single layer. Assumes self.grad already contains dJ/output.
        (The derivative of the objective function with respect to this layer's output).
        Accumulates the calculated gradients into the gradients of the previous layers.
        """
        dJ_dinput = self.grad * (self.input.output > 0)
        self.input.accumulate_grad(dJ_dinput)


class L2Loss(Layer):
    """
    This is a good loss function for regression problems.

    It implements the squared L2 norm of the inputs.
    """
    def __init__(self, pred_layer, true_layer):
        """
        Takes in a predicted/calculated layer and a true values layer.
        """
        assert pred_layer.output.shape == true_layer.output.shape, "input layers shapes do not match " + pred_layer.output.shape + " : " + true_layer.output.shape
        Layer.__init__(self, 1)
        self.pred_layer = pred_layer
        self.true_layer = true_layer

    def forward(self):
        """
        Set this layer's output based on the outputs of the layers that feed into it. 1/2 * L2 squared norm of pred_layer - true_layer
        """
        self.output = .5 * torch.linalg.norm(self.pred_layer.output - self.true_layer.output)

    def backward(self):
        """
        Performs back-propagation through the single layer. Assumes self.grad already contains dJ/output.
        (The derivative of the objective function with respect to this layer's output).
        Accumulates the calculated gradients into the gradients of the previous layers.
        """
        dJ_d = self.grad * (self.pred_layer.output - self.true_layer.output)
        self.true_layer.accumulate_grad(dJ_d)
        self.pred_layer.accumulate_grad(dJ_d)


class Sum(Layer):
    def __init__(self, first_layer, second_layer, final=False):
        """
        Takes in 2 layers to sum. If this is the final sum layer, gets the argmax from the first input layer.
        Therefore, if this is the final layer, the first input layer needs to have a get_argmax() function
        """
        assert first_layer.output.shape == second_layer.output.shape, "input layers shapes do not match " + first_layer.output.shape + " : " + second_layer.output.shape
        Layer.__init__(self, first_layer.output.shape)
        self.first_layer = first_layer
        self.second_layer = second_layer
        self.argmax = -1
        self.final = final

    def forward(self):
        """
        Set this layer's output based on the outputs of the layers that feed into it. (first_layer + second_layer)
        """
        self.output = self.first_layer.output + self.second_layer.output
        if self.final:
            self.argmax = self.first_layer.get_argmax()

    def backward(self):
        """
        Performs back-propagation through the single layer. Assumes self.grad already contains dJ/output.
        (The derivative of the objective function with respect to this layer's output).
        Accumulates the calculated gradients into the gradients of the previous layers.

        partial derivative of a sum layer is 1
        """
        self.first_layer.accumulate_grad(self.grad)
        self.second_layer.accumulate_grad(self.grad)

    def get_argmax(self):
        return self.argmax


class Regularization(Layer):
    def __init__(self, input_layer, lamb):
        """
        Takes in an input layer and a lambda value
        """
        Layer.__init__(self, 1)
        self.input_layer = input_layer
        self.lamb = lamb

    def forward(self):
        """
        Set this layer's output based on the outputs of the layers that feed into it.
        (1/2 * the squared Frobenius Norm of the input layer * lambda)
        """
        self.output = (self.lamb/2) * (torch.norm(self.input_layer.output) ** 2)

    def backward(self):
        """
        Performs back-propagation through the single layer. Assumes self.grad already contains dJ/output.
        (The derivative of the objective function with respect to this layer's output).
        Accumulates the calculated gradients into the gradients of the previous layers.
        """
        djdi = self.grad * self.lamb * self.input_layer.output
        self.input_layer.accumulate_grad(djdi)


class Softmax(Layer):
    """
    Applies a softmax
    """
    def __init__(self, input_layer):
        """
        Takes in a predicted/calculated layer
        """
        Layer.__init__(self, input_layer.output.shape)
        self.input_layer = input_layer

    def forward(self):
        """
        Set this layer's output based on the outputs of the layers that feed into it. (Applies the softmax function)
        Implements the stable version of softmax compared to the non-stable version in previous weeks
        """
        maxi = torch.max(self.input_layer.output)
        self.output = torch.exp(self.input_layer.output - maxi) / torch.sum(torch.exp(self.input_layer.output - maxi))

    def backward(self):
        # not implementing since we will only be using backward with the Softmax & Cross Entropy combined layer
        pass


class SoftmaxCrossEntropy(Layer):
    """
    Combines Softmax and Cross Entropy into 1 layer
    """
    def __init__(self, pred_layer, true_layer):
        """
        Takes in a predicted/calculated layer and a true values layer.
        The output of this layer is the loss
        The argmax of this layer is the predicted class, which can be used to calculate the accuracy of the model
        if you compare it with the argmax of the true values
        """
        assert pred_layer.output.shape == true_layer.output.shape, "input layers shapes do not match " + pred_layer.output.shape + " : " + true_layer.output.shape
        Layer.__init__(self, 1)
        self.pred_layer = pred_layer
        self.true_layer = true_layer
        self.argmax = None

    def forward(self):
        """
        Set this layer's output based on the outputs of the layers that feed into it. (Applies the softmax function and cross-entropy)
        Also calculates just the argmax of the softmax
        """
        # stable version
        maxi = torch.max(self.pred_layer.output)
        self.argmax = torch.argmax(torch.exp(self.pred_layer.output - maxi) / torch.sum(torch.exp(self.pred_layer.output - maxi)))
        self.output = -1 * torch.sum(self.true_layer.output * (self.pred_layer.output - maxi - torch.log(torch.sum(torch.exp(self.pred_layer.output - maxi)))))

    def backward(self):
        """
        Performs back-propagation through the single layer. Assumes self.grad already contains dJ/output.
        (The derivative of the objective function with respect to this layer's output).
        Accumulates the calculated gradients into the gradients of the previous layers.

        derivative of the predicted layer is softmax(pred) - true
        the derivative of the true_layer does not need to be calculated because it is the true values, they don't need to be adjusted
        """
        # stable version
        maxi = torch.max(self.pred_layer.output)
        dJ_dpred = torch.exp(self.pred_layer.output - maxi) / torch.sum(torch.exp(self.pred_layer.output - maxi)) - self.true_layer.output
        self.pred_layer.accumulate_grad(dJ_dpred)

    def get_argmax(self):
        """
        Returns the argmax of just the softmax function
        """
        return self.argmax


class Flatten(Layer):
    """
    Flattens a multidimensional input to a single dimension
    """
    def __init__(self, input_layer):
        """
        Takes in a multidimensional input layer
        """
        Layer.__init__(self, torch.numel(input_layer.output))
        self.input_layer = input_layer

    def forward(self):
        """
        Flattens the input
        """
        self.output = torch.flatten(self.input_layer.output)

    def backward(self):
        """
        Performs back-propagation through the single layer. Assumes self.grad already contains dJ/output.
        (The derivative of the objective function with respect to this layer's output).
        Accumulates the calculated gradients into the gradients of the previous layers.

        Partial derivative of flatten layer is 1
        """
        self.input_layer.accumulate_grad(torch.reshape(self.grad, self.input_layer.output.shape))


class Convolution(Layer):
    """
    Layer performs a convolution. Stride of 1.
    """
    def __init__(self, image_layer, filter_layer, padding=0):
        """
        Takes in a multidimensional image layer and a filter (input) layer
        :param image_layer: m x h x w torch tensor. m is the number of channels
        :param filter_layer: n x m x s0 x s1 torch tensor. n is the number of filters
                                                and m is the number of channels in each filter
        :param padding: pixels to add on each edge of each channel of the input image.  Pads the input
                        to be m x (h+2pad) x (w+2pad) where pad is the padding argument.
        :output shape: n x h+2pad-s0+1 x w+2pad-s1+1 tensor resulting from convolving image with filter.
        """
        assert len(image_layer.output.shape) == 3
        assert len(filter_layer.output.shape) == 4
        assert filter_layer.output.shape[1] == image_layer.output.shape[0]
        Layer.__init__(self, [filter_layer.output.shape[0],
                              image_layer.output.shape[1] + 2 * padding - filter_layer.output.shape[2] + 1,
                              image_layer.output.shape[2] + 2 * padding - filter_layer.output.shape[3] + 1])
        self.padding = padding
        self.image_layer = image_layer
        self.filter_layer = filter_layer

    def forward(self):
        """
        Convolves an m-channel image with a bank of n filters each with m channels
        to produce an n-channeled image.  To produce the nth channel, the nth filter
        is convolved with the image.
        """
        self.output = conv2d.conv_filter_forward(self.image_layer.output, self.filter_layer.output, self.padding)

    def backward(self):
        """
        Performs back-propagation through the single layer. Assumes self.grad already contains dJ/output.
        (The derivative of the objective function with respect to this layer's output).
        Accumulates the calculated gradients into the gradients of the previous layers.

        dJ/dImage = dJ/doutput convolved with Filter
        dJ/dFilter = dJ/doutput convolved with Image

        Only works when padding is 1
        """
        dJ_dFilter = conv2d.conv_expand_layers(self.image_layer.output, self.grad, self.padding)
        dJ_dImage = conv2d.conv_filter_backward(self.grad, torch.flip(torch.flip(self.filter_layer.output, [0, 1]), [1, 0]), self.padding)
        self.filter_layer.accumulate_grad(dJ_dFilter)
        self.image_layer.accumulate_grad(dJ_dImage.reshape(dJ_dImage.shape[1:4]))
