import torch


class Network:
    def __init__(self):
        """
        Initialize a layers attribute
        """
        self.layers = []
        self.output = None

    def get_output(self):
        return self.output

    def add(self, layer):
        """
        Adds a new layer to the network.
        The first layer added should be an input layer

        Sublayers can *only* be added after their inputs have been added.
        (In other words, the DAG of the graph must be flattened and added in order from input to output)
        :param layer: The sublayer to be added
        """
        self.layers.append(layer)

    def forward(self, input):
        """
        Compute the output of the network in the forward direction.

        :param input: A numpy array that will serve as the input for this forward pass
        :return: the argmax of the softmax decision (the index of the predicted value)
        """
        # Assign the input to the input layer's output before performing the forward evaluation of the network.
        #
        # Users will be expected to add layers to the network in the order they are evaluated, so
        # this method can simply call the forward method for each layer in order.
        self.layers[0].set(torch.from_numpy(input))  # the first layer must be an Input layer
        for i in range(1, len(self.layers)):
            self.layers[i].forward()
        self.output = self.layers[-1].get_output()
        return self.layers[-1].get_argmax()

    def backward(self):
        """
        Complete the backpropagation of the network
        """
        for layer in self.layers:
            layer.clear_grad()
        self.layers[-1].set_grad(torch.tensor([1]))  # Setting the last layer's grad to 1
        for i in range(len(self.layers)-1, 0, -1):
            self.layers[i].backward()

    def step(self, step_size):
        """
        Perform a single step of stochastic gradient descent for the network
        """
        for layer in self.layers:
            layer.step(step_size)
