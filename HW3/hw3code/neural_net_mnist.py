# This is a code written foor CSE 546 HW3 A4, in spring 2021
# We are using neural net to distinguish the digits 2, 7 in the MNIST dataset.
# Author: Hongda Li
# Don't copy my code it has my style in it.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import transforms
from tqdm import tqdm
from math import sqrt
from collections import Iterable
import matplotlib.pyplot as plt

tensor = torch.tensor
zeros = torch.zeros
rand = torch.rand

MNIST_TRAIN = datasets.MNIST(root="./data",
                             train=True,
                             download=True,
                             transform=transforms.ToTensor())
MNIST_TEST = datasets.MNIST(root="./data",
                            train=False,
                            download=True,
                            transform=transforms.ToTensor())


class MyNN:

    def __init__(this, modelParameterMaker:callable):
        """
            Architecture:
            Direct linear stacking of weight and biases. with ReLU.
        :param modelParameterMaker:
            A function that makes the W, weight matrices and the weight vector
            b in the neural network.
        """
        this.Weights, this.Biases = modelParameterMaker()
        assert isinstance(this.Weights, Iterable)
        assert isinstance(this.Biases, Iterable)

    def feedforward(this, X, y):
        """
            X is a row data matrix.
        :param X:
        :param y:
        :return:
        """
        assert X.ndim == 2
        assert X.shape[0] == y.shape[0] or X.shape[1] == y.shape[0]
        a = X  # output of the first layer
        for W, b in list(zip(this.Weights, this.Biases))[:-1]:
            a = F.relu(a@W + b)
        # direct out put from last layer into the loss function.
        a = a @ this.Weights[-1] + this.Biases[-1]
        return F.cross_entropy(a, y)

    def predict(this, X):
        a = X  # output of the first layer
        for W, b in list(zip(this.Weights, this.Biases))[:-1]:
            a = F.relu(a @ W + b)
        a = a @ this.Weights[-1] + this.Biases[-1]
        Probability = F.softmax(a, dim=1)
        return torch.max(Probability, dim=1)[1]


    @property
    def parameters(this):
        # Weights and biases concat together.
        return list(this.Weights) + list(this.Biases)

    @staticmethod
    def A4a():
        """
            Get the parameters ready for A4a.
        :param gpu:
            Whether to use the GPU on the tensor.
        :return:
            2 iterables of the weights and biases.
        """
        W0 = zeros(28 ** 2, 64, requires_grad=True)
        alpha = 1/sqrt(W0.shape[1])
        W0.data += 2*alpha* rand(W0.shape) - alpha
        W1 = zeros(64, 10, requires_grad=True)
        alpha = 1 / sqrt(W1.shape[1])
        W1.data += 2*alpha*rand(W1.shape) - alpha
        b0 = zeros(1,64, requires_grad=True)
        b1 = zeros(1,10, requires_grad=True)
        return [W0, W1], [b0, b1]

    @staticmethod
    def A4b():
        W0 = zeros(28 ** 2, 32, requires_grad=True)
        alpha = 1 / sqrt(W0.shape[1])
        W0.data += 2 * alpha * rand(W0.shape) - alpha

        W1 = zeros(32, 32, requires_grad=True)
        alpha = 1 / sqrt(W1.shape[1])
        W1.data += 2 * alpha * rand(W1.shape) - alpha

        W2 = zeros(32, 10, requires_grad=True)
        alpha = 1 / sqrt(W2.shape[1])
        W2.data += 2 * alpha * rand(W2.shape) - alpha

        b0 = zeros(1, 32, requires_grad=True)
        b1 = zeros(1, 32, requires_grad=True)
        b2 = zeros(1, 10, requires_grad=True)

        return [W0, W1, W2], [b0, b1, b2]


def main():

    data_loader = torch.utils.data.DataLoader(MNIST_TRAIN,
                                              batch_size=250,
                                              shuffle=True)


    Epochs = 30

    def Accuracy(yhat, y):
        return sum(yhat == y)/yhat.numel()

    def RunMNIST(Model, Optimizer, part):
        EpochLosses = []
        for E in range(Epochs):
            EpochLoss = 0.0
            for X, y, in data_loader:
                X = X.view(-1, 784)
                Optimizer.zero_grad()
                Loss = Model.feedforward(X, y)
                EpochLoss += Loss.item()/X.shape[0]
                Loss.backward()
                Optimizer.step()
            EpochLosses.append(EpochLoss)
            X = torch.stack([D[0].reshape(-1) for D in MNIST_TRAIN], axis=0)
            y = torch.tensor([D[1] for D in MNIST_TRAIN])
            Rate = Accuracy(Model.predict(X), y)
            print(f"Epoch: {E}, Loss: {EpochLoss}")
            print(f"accuracy: {Rate}")
            if Rate > 0.99:
                print("Process terminated because 99% accuracy reached.")
                break
        X = torch.stack([D[0].reshape(-1) for D in MNIST_TEST], axis=0)
        y = torch.tensor([D[1] for D in MNIST_TEST])
        TestAccuracy = Accuracy(Model.predict(X), y)
        print(f"Test set accuracy is: {TestAccuracy}")
        plt.plot(EpochLosses)
        plt.title("A4a-Cross Entropy Loss at each Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Entropy Loss")
        plt.savefig(f"A4({part})-NN-mnist.png")
        plt.show()
        return TestAccuracy

    Model = MyNN(MyNN.A4a)
    Optimizer = optim.Adam(Model.parameters, lr=0.01)
    Rate = RunMNIST(Model, Optimizer, part="a")
    with open("A4a.txt", "w+") as f:
        f.write(Rate)

    Model = MyNN(MyNN.A4b)
    Optimizer = optim.Adam(Model.parameters, lr=0.01)
    Rate = RunMNIST(Model, Optimizer, part="b")
    with open("A4b.txt", "w+") as f:
        f.write(Rate)







if __name__ == "__main__":
    import os
    print(f"cwd:{os.getcwd()} ")
    main()