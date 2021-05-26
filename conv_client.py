import torch
import numpy as np
import pandas as pd

import network
import layers
import time

from keras.datasets import cifar10


def one_hot_encode(array):
    b = np.zeros((array.size, array.max()+1))
    b[np.arange(array.size), array] = 1
    return b


# The Fashion-MNIST dataset consists of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images.
# The class labels for the data are: [T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot]
def fashion_mnist_client():
    """
    Loads in the fashion mnist datasets (training/testing). Network consists of 2 convolutional layers with a ReLU
    layer after each one, a flatten layer, and then a fully connected layer to produce the 10 output nodes.
    Achieved a 75.92% testing accuracy after 60 epochs, .005 step size

    Network layout: (1x28x28) - (10x28x28) - (10x28x28) -> 7840 - 10
    """
    test_df = pd.read_csv('../week3/fashion-mnist_test.csv')
    train_df = pd.read_csv('../week3/fashion-mnist_train.csv')
    y_train = one_hot_encode(np.array(train_df['label'])).astype(np.float32)
    del train_df['label']
    x_train = np.array(train_df, dtype=np.float32).reshape((60000, 1, 28, 28))
    y_test = one_hot_encode(np.array(test_df['label'])).astype(np.float32)
    del test_df['label']
    x_test = np.array(test_df, dtype=np.float32).reshape((10000, 1, 28, 28))

    # Normalize the data
    x_train /= 255.0
    x_test /= 255.0

    image_layer = layers.Input([1, 28, 28], False)
    filter1_layer = layers.Input([10, 1, 3, 3], True)
    filter1_layer.randomize()
    conv1_layer = layers.Convolution(image_layer, filter1_layer, padding=1)
    relu1_layer = layers.ReLU(conv1_layer)
    filter2_layer = layers.Input([10, 10, 3, 3], True)
    filter2_layer.randomize()
    conv2_layer = layers.Convolution(relu1_layer, filter2_layer, padding=1)
    relu2_layer = layers.ReLU(conv2_layer)
    flatten_layer = layers.Flatten(relu2_layer)
    w_layer = layers.Input([10, 7840], True)
    w_layer.randomize()
    b_layer = layers.Input(10, True)
    b_layer.randomize()
    linear_layer = layers.Linear(flatten_layer, w_layer, b_layer)
    y_layer = layers.Input(10, False)
    softmaxce_layer = layers.SoftmaxCrossEntropy(linear_layer, y_layer)

    net = network.Network()
    net.add(image_layer)
    net.add(filter1_layer)
    net.add(conv1_layer)
    net.add(relu1_layer)
    net.add(filter2_layer)
    net.add(conv2_layer)
    net.add(relu2_layer)
    net.add(flatten_layer)
    net.add(w_layer)
    net.add(b_layer)
    net.add(linear_layer)
    net.add(softmaxce_layer)

    start = time.perf_counter()

    for epoch in range(100):
        count_training = 0
        total_loss_training = 0
        for i in range(60000):  # 1 epoch of training
            y_layer.set(torch.from_numpy(y_train[i]))
            if net.forward(x_train[i]) == np.argmax(y_train[i]):
                count_training = count_training + 1
            total_loss_training = total_loss_training + net.get_output().item()
            net.backward()
            net.step(.005)

        count_testing = 0
        total_loss_testing = 0
        for i in range(10000):  # 1 epoch of testing
            y_layer.set(torch.from_numpy(y_test[i]))
            if net.forward(x_test[i]) == np.argmax(y_test[i]):
                count_testing = count_testing + 1
            total_loss_testing = total_loss_testing + net.get_output().item()
        print('Epoch ' + str(epoch+1) + ', Training acc: ' + str(count_training/60000 * 100) + '%, Training loss: ' + str(total_loss_training/60000)
              + ', Testing acc: ' + str(count_testing/10000 * 100) + '%, Testing loss: ' + str(total_loss_testing/10000))
    end = time.perf_counter()
    print('total training time: ' + str(end - start) + 'sec')


def cifar_10_client():
    """
    Loads in the cifar-10 datasets (training/testing). Network consists of 2 convolutional layers with a ReLU
    layer after each one, a flatten layer, and then a fully connected layer to produce the 10 output nodes.
    Also regularization is performed on all kernels and weights and biases of the fully connected layer.
    Achieved a 51.58% testing accuracy after 30 epochs, first 20 epochs with a .001 step size, last 10 with .0001.
    Regularization lambda of .01

    Network layout: (3x32x32) - (20x32x32) - (20x32x32) -> 20480 - 10
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = np.asarray(x_train, dtype=np.float32).reshape((50000, 3, 32, 32))  # flattening data
    y_train = one_hot_encode(np.asarray(y_train).reshape(50000,)).astype(np.float32)
    x_test = np.asarray(x_test, dtype=np.float32).reshape((10000, 3, 32, 32))
    y_test = one_hot_encode(np.asarray(y_test).reshape(10000,)).astype(np.float32)

    # Normalize the data
    x_train /= 255.0
    x_test /= 255.0

    image_layer = layers.Input([3, 32, 32], False)
    filter1_layer = layers.Input([20, 3, 3, 3], True)
    filter1_layer.randomize()
    conv1_layer = layers.Convolution(image_layer, filter1_layer, padding=1)
    relu1_layer = layers.ReLU(conv1_layer)
    filter2_layer = layers.Input([20, 20, 3, 3], True)
    filter2_layer.randomize()
    conv2_layer = layers.Convolution(relu1_layer, filter2_layer, padding=1)
    relu2_layer = layers.ReLU(conv2_layer)
    flatten_layer = layers.Flatten(relu2_layer)
    w_layer = layers.Input([10, 20480], True)
    w_layer.randomize()
    b_layer = layers.Input(10, True)
    b_layer.randomize()
    linear_layer = layers.Linear(flatten_layer, w_layer, b_layer)
    y_layer = layers.Input(10, False)
    softmaxce_layer = layers.SoftmaxCrossEntropy(linear_layer, y_layer)
    reg1_layer = layers.Regularization(filter1_layer, .01)
    reg2_layer = layers.Regularization(filter2_layer, .01)
    reg3_layer = layers.Regularization(w_layer, .01)
    regsum1_layer = layers.Sum(reg1_layer, reg2_layer)
    regsum2_layer = layers.Sum(regsum1_layer, reg3_layer)
    finalsum_layer = layers.Sum(softmaxce_layer, regsum2_layer, True)

    net = network.Network()
    net.add(image_layer)
    net.add(filter1_layer)
    net.add(conv1_layer)
    net.add(relu1_layer)
    net.add(filter2_layer)
    net.add(conv2_layer)
    net.add(relu2_layer)
    net.add(flatten_layer)
    net.add(w_layer)
    net.add(b_layer)
    net.add(linear_layer)
    net.add(softmaxce_layer)
    net.add(reg1_layer)
    net.add(reg2_layer)
    net.add(reg3_layer)
    net.add(regsum1_layer)
    net.add(regsum2_layer)
    net.add(finalsum_layer)

    start = time.perf_counter()

    for epoch in range(100):
        count_training = 0
        total_loss_training = 0
        for i in range(50000):  # 1 epoch of training
            y_layer.set(torch.from_numpy(y_train[i]))
            if net.forward(x_train[i]) == np.argmax(y_train[i]):
                count_training = count_training + 1
            total_loss_training = total_loss_training + net.get_output().item()
            net.backward()
            if epoch < 20:
                net.step(.001)
            else:
                net.step(.0001)

        count_testing = 0
        total_loss_testing = 0
        for i in range(10000):  # 1 epoch of testing
            y_layer.set(torch.from_numpy(y_test[i]))
            if net.forward(x_test[i]) == np.argmax(y_test[i]):
                count_testing = count_testing + 1
            total_loss_testing = total_loss_testing + net.get_output().item()
        print('Epoch ' + str(epoch + 1) + ', Training acc: ' + str(count_training / 50000 * 100) + '%, Training loss: ' + str(total_loss_training / 50000)
              + ', Testing acc: ' + str(count_testing / 10000 * 100) + '%, Testing loss: ' + str(total_loss_testing / 10000))
    end = time.perf_counter()
    print('total training time: ' + str(end - start) + 'sec')


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print("Running on the GPU")
    else:
        print("Running on the CPU")
    # fashion_mnist_client()
    cifar_10_client()