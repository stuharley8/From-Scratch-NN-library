import torch
import numpy as np
import pandas as pd

import network
import layers
import time

from keras.datasets import cifar10
from keras.datasets import cifar100


def one_hot_encode(array):
    b = np.zeros((array.size, array.max()+1))
    b[np.arange(array.size), array] = 1
    return b


# The Fashion-MNIST dataset consists of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images.
# The class labels for the data are: [T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot]
def fashion_mnist_client():
    """
    Loads in the fashion mnist datasets (training/testing). Creates a network containing 2 linear layers with a relu layer
    in between and a softmax + cross entropy layer at the end. Trains the network for 10 epochs using stochastic gradient descent
    with a batch size of 1. (Adjusts the weights & biases after each sample). Then tests the testing set to calculate
    the accuracy score.

    Network layout: 784 - 28 - 10
    """
    test_df = pd.read_csv('fashion-mnist_test.csv')
    train_df = pd.read_csv('fashion-mnist_train.csv')
    y_train = one_hot_encode(np.array(train_df['label'])).astype(np.float64)
    del train_df['label']
    x_train = np.array(train_df, dtype=np.float64)
    y_test = one_hot_encode(np.array(test_df['label'])).astype(np.float64)
    del test_df['label']
    x_test = np.array(test_df, dtype=np.float64)

    # Normalize the data
    x_train /= 255.0
    x_test /= 255.0

    x_layer = layers.Input(784, False)
    b1_layer = layers.Input(28, True)
    b1_layer.randomize()
    W1_layer = layers.Input([28, 784], True)
    W1_layer.randomize()
    linear1_layer = layers.Linear(x_layer, W1_layer, b1_layer)
    relu_layer = layers.ReLU(linear1_layer)
    b2_layer = layers.Input(10, True)
    b2_layer.randomize()
    W2_layer = layers.Input([10, 28], True)
    W2_layer.randomize()
    linear2_layer = layers.Linear(relu_layer, W2_layer, b2_layer)
    y_layer = layers.Input(10, False)
    softmaxce_layer = layers.SoftmaxCrossEntropy(linear2_layer, y_layer)

    net = network.Network()
    net.add(x_layer)
    net.add(W1_layer)
    net.add(b1_layer)
    net.add(linear1_layer)
    net.add(relu_layer)
    net.add(W2_layer)
    net.add(b2_layer)
    net.add(linear2_layer)
    net.add(softmaxce_layer)

    start = time.perf_counter()

    for epoch in range(10):
        count_training = 0
        total_loss_training = 0
        for i in range(60000):  # 1 epoch of training
            y_layer.set(torch.from_numpy(y_train[i]))
            if net.forward(x_train[i]) == np.argmax(y_train[i]):
                count_training = count_training + 1
            total_loss_training = total_loss_training + net.get_output().item()
            net.backward()
            net.step(.01)

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
    Loads in the cifar-10 datasets (training/testing). Creates a network containing 2 linear layers with a relu layer
    in between and a softmax + cross entropy layer at the end. Trains the network for 10 epochs using stochastic gradient descent
    with a batch size of 1. (Adjusts the weights & biases after each sample). Then tests the testing set to calculate
    the accuracy score.

    Network layout: 3072 - 100 - 10
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = np.asarray(x_train, dtype=np.float64).reshape(50000, 3072)  # flattening data
    y_train = one_hot_encode(np.asarray(y_train).reshape(50000,)).astype(np.float64)
    x_test = np.asarray(x_test, dtype=np.float64).reshape(10000, 3072)
    y_test = one_hot_encode(np.asarray(y_test).reshape(10000,)).astype(np.float64)

    # Normalize the data
    x_train /= 255.0
    x_test /= 255.0

    x_layer = layers.Input(3072, False)
    b1_layer = layers.Input(100, True)
    b1_layer.randomize()
    W1_layer = layers.Input([100, 3072], True)
    W1_layer.randomize()
    linear1_layer = layers.Linear(x_layer, W1_layer, b1_layer)
    relu_layer1 = layers.ReLU(linear1_layer)
    b2_layer = layers.Input(10, True)
    b2_layer.randomize()
    W2_layer = layers.Input([10, 100], True)
    W2_layer.randomize()
    linear2_layer = layers.Linear(relu_layer1, W2_layer, b2_layer)
    y_layer = layers.Input(10, False)
    softmaxce_layer = layers.SoftmaxCrossEntropy(linear2_layer, y_layer)
    reg1_layer = layers.Regularization(W1_layer, .01)
    reg2_layer = layers.Regularization(W2_layer, .01)
    regsum_layer = layers.Sum(reg1_layer, reg2_layer)
    finalsum_layer = layers.Sum(softmaxce_layer, regsum_layer, True)

    net = network.Network()
    net.add(x_layer)
    net.add(W1_layer)
    net.add(b1_layer)
    net.add(linear1_layer)
    net.add(relu_layer1)
    net.add(W2_layer)
    net.add(b2_layer)
    net.add(linear2_layer)
    net.add(reg1_layer)
    net.add(reg2_layer)
    net.add(regsum_layer)
    net.add(softmaxce_layer)
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
            net.step(.001)

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


def cifar_100_client():
    """
    Loads in the cifar-10 datasets (training/testing). Creates a network containing 2 linear layers with a relu layer
    in between and a softmax + cross entropy layer at the end. Trains the network for 10 epochs using stochastic gradient descent
    with a batch size of 1. (Adjusts the weights & biases after each sample). Then tests the testing set to calculate
    the accuracy score.

    Network layout: 3072 - 200 - 100

    For whatever reason I get nan loss values for this network. I have no idea why b/c I copied and pasted this code
    from cifar_10_client which works perfectly. I have implemented the stable form of softmax + cross entropy
    successfully and my tests confirm that it works. I have made sure data is in the right shape, type.
    idk what is going wrong.
    """
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = np.asarray(x_train, dtype=np.float64).reshape(50000, 3072)  # flattening data
    y_train = one_hot_encode(np.asarray(y_train).reshape(50000,)).astype(np.float64)
    x_test = np.asarray(x_test, dtype=np.float64).reshape(10000, 3072)
    y_test = one_hot_encode(np.asarray(y_test).reshape(10000,)).astype(np.float64)

    # Normalize the data
    x_train /= 255.0
    x_test /= 255.0

    x_layer = layers.Input(3072, False)
    b1_layer = layers.Input(200, True)
    b1_layer.randomize()
    W1_layer = layers.Input([200, 3072], True)
    W1_layer.randomize()
    linear1_layer = layers.Linear(x_layer, W1_layer, b1_layer)
    relu_layer1 = layers.ReLU(linear1_layer)
    b2_layer = layers.Input(100, True)
    b2_layer.randomize()
    W2_layer = layers.Input([100, 200], True)
    W2_layer.randomize()
    linear2_layer = layers.Linear(relu_layer1, W2_layer, b2_layer)
    y_layer = layers.Input(100, False)
    softmaxce_layer = layers.SoftmaxCrossEntropy(linear2_layer, y_layer)
    reg1_layer = layers.Regularization(W1_layer, .01)
    reg2_layer = layers.Regularization(W2_layer, .01)
    regsum_layer = layers.Sum(reg1_layer, reg2_layer)
    finalsum_layer = layers.Sum(softmaxce_layer, regsum_layer, True)

    net = network.Network()
    net.add(x_layer)
    net.add(W1_layer)
    net.add(b1_layer)
    net.add(linear1_layer)
    net.add(relu_layer1)
    net.add(W2_layer)
    net.add(b2_layer)
    net.add(linear2_layer)
    net.add(reg1_layer)
    net.add(reg2_layer)
    net.add(regsum_layer)
    net.add(softmaxce_layer)
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
            net.step(.001)

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
    torch.set_default_dtype(torch.float64)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        print("Running on the GPU")
    else:
        print("Running on the CPU")
    # fashion_mnist_client()
    # cifar_10_client()
    cifar_100_client()
