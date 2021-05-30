### CLASS: CSE 546 SPRING 2021 HW4, A3
### Name: Hongda Li
### My code has my style in it don't copy.


import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from tqdm import tqdm
import random as sysrandom


TRANSFORM  = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
TRAIN_SET = \
    datasets.MNIST('./data', download=True, train=True, transform=TRANSFORM)
TEST_SET = \
    datasets.MNIST('.data', download=False, train=True, transform=TRANSFORM)

TRAIN_SET, TEST_SET = \
    torch.utils.data.Subset(TRAIN_SET, range(0, 1000)), \
    torch.utils.data.Subset(TEST_SET, range(0, 1000))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class A4Model(nn.Module):

    def __init__(this, h:int, non_linear:bool=False):
        """

        :param kargs:
            d: The dimension of the data.
            h: The width of the hidden layer.
            activation: TRUE, FALSE
                Whether to use ReLU activation function on hidden and output layer.
        """
        super().__init__()
        d = 28**2
        this.L1 = nn.Linear(d, h)
        this.L2 = nn.Linear(h, d)
        this.NonLin = non_linear
        this.MSE = nn.MSELoss()


    def forward(this, X):
        """
            Feed Forward without Loss function capped onto the output layer.
        :param X:
            The data matrix, row data matrix.
        :return:

        """
        x = this.L1(X)
        if this.NonLin:
            x = nn.ReLU(x)
        x = this.L2(x)
        if this.NonLin:
            x = nn.ReLU(x)
        return x

    def GetEmbeding(this, X):

        pass

    def FeedForward(this, X):
        return this.MSE(this(X), X)


def MNISTTenUniqueDigitsLoader(train=True):
    data = \
        datasets.MNIST('./data', download=True, train=train, transform=TRANSFORM)
    Indices = []
    for II in range(10):
        Idx = torch.where(data.targets == II)[0]
        Indices.append(Idx[sysrandom.randint(0, len(Idx))])
    Subset = torch.utils.data.Subset(data, Indices)
    return torch.utils.data.DataLoader(Subset, batch_size=10, shuffle=False)


def BatchThisModel(theModel:A4Model,
                   dataLoader:torch.utils.data.DataLoader,
                   optimizer:optim.Adam=None,
                   transform:callable=None):
    """
        Batch this model for one epoch, give me the model optimizer and some
        extra thing, then it will collect the average loss of one epoch.
        Note:
        This one is for Regression Model, it assumes MSE loss, loss of each
        batch if divided by the total number of batches from the data loader.
    :param theModel:
    :param dataLoader:
    :param transform:
    :return:
    """
    AvgLoss = 0; theModel.to(DEVICE)
    L = len(dataLoader)
    for II, (X, _) in enumerate(tqdm(dataLoader)):
        if transform is not None: X = transform(X)
        X= X.to(DEVICE)
        if optimizer is None:
            with torch.no_grad():
                AvgLoss += float(theModel.FeedForward(X))/L
        else:
            optimizer.zero_grad()
            Loss = theModel.FeedForward(X)
            AvgLoss += Loss.item() / L
            Loss.backward()
            optimizer.step()
    with torch.no_grad():
        Loss = theModel.FeedForward(X)
        AvgLoss += Loss.item() / L
    return AvgLoss


def GetTrainTestLoaders(bs=100):
    TrainLoader = \
        torch.utils.data.DataLoader(TRAIN_SET, batch_size=bs, shuffle=True)
    TestLoader = \
        torch.utils.data.DataLoader(TEST_SET, batch_size=bs, shuffle=True)
    return TrainLoader, TestLoader


def Ts():
    from time import gmtime, strftime
    TimeStamp = strftime("%H-%M-%S", gmtime())
    return TimeStamp


def mkdir(dir):
    from pathlib import Path
    Path(dir).mkdir(parents=True, exist_ok=True)


def log(fname:str, content:str, dir):
    mkdir(dir)
    TimeStamp = Ts()
    with open(f"{dir}{TimeStamp}-{fname}.txt","w+") as f:
        f.write(content)


def main():
    def A4Run(NonLin, h):
        Bs, Epochs, lr = 100, 30, 0.01
        Losses = []
        Model = A4Model(h=h, non_linear=False)
        Optimizer = optim.Adam(Model.parameters(), lr=lr)
        TrainLoader, TestsLoader = GetTrainTestLoaders(bs=Bs)
        for Epoch in range(Epochs):
            Loss = BatchThisModel(theModel=Model,
                                  dataLoader=TrainLoader,
                                  optimizer=Optimizer,
                                  transform=lambda x: x.reshape(Bs, -1))
            Losses.append(Loss)
        log(fname=f"A4-final-train-Loss-h={h}-{'Non' if NonLin else ''}lin-model",
            content=f"Total Epochs: {Epochs} Train Final MSE Loss: {Losses[-1]}",
            dir="./A4logs/")
        def Visualize(Train=True):
            mkdir("./A4plots")
            DigitsLoader = MNISTTenUniqueDigitsLoader(Train)
            for X, _ in DigitsLoader:
                X = X.reshape(X.shape[0], -1)
                Reconstructed = Model(X.to(DEVICE))
            for II, (Row1, Row2) in enumerate(zip(Reconstructed.data, X.data)):
                Together = torch.zeros(28, 28*2)
                Together[:, :28] = Row1.reshape(28, 28)
                Together[:,28:]  = Row2.reshape(28, 28)
                plt.matshow(Together)
                plt.title(f"h:{h}, ReLU Model: {NonLin}, left reconstructed, right Original\n"
                f"{'Train Set' if Train else 'Test Set'}")
                plt.savefig(f"./A4plots/{Ts()}-h-{h}-{'non' if NonLin else ''}lin-digit-{II}.png")
                plt.show()

        Visualize()
        if h == 128:
            Loss = BatchThisModel(theModel=Model,
                                  dataLoader=TestsLoader,
                                  transform=lambda x: x.reshape(Bs, -1))
            log(fname=f"A4-test-loss-h={h}-{'Non' if NonLin else ''}lin-model",
                content=f"Total Epochs: {Epochs} Test Final MSE Loss: {Loss}",
                dir="./A4logs/")


        return Model, Losses

    A4Run(False, h=32)
    A4Run(False, h=64)
    A4Run(False, h=128)
    A4Run(True, h=32)
    A4Run(True, h=64)
    A4Run(True, h=128)



if __name__ == "__main__":
    import os
    print(os.curdir)
    print(os.getcwd())
    main()


