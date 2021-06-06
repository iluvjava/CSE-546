# This is for HW4 A3
# Course: CSE 546, SPRING 2021
# Name: Hongda Li
# My code has my tyle in it please don't copy.

import numpy as np
import scipy
from scipy import linalg
from torchvision import datasets
from mnist import MNIST
import matplotlib.pyplot as plt
from tqdm import tqdm
zeros = np.zeros
randint = np.random.randint
randn = np.random.randn
eigh = linalg.eigh
norm = np.linalg.norm
cumsum = np.cumsum



if "MNIST_DATA" not in dir(): # running on interactive console will be faster
    datasets.MNIST('./data', download=True, train=True)
    MNIST_DATA = MNIST("./data/MNIST/raw/")
    TRAIN_X, _= MNIST_DATA.load_training()
    TEST_X, _ = MNIST_DATA.load_testing()
    TRAIN_X= np.array(TRAIN_X, dtype=np.float)/255
    TEST_X = np.array(TEST_X, dtype=np.float)/255
    TRAIN_Y = np.array(MNIST_DATA.train_labels)
    print("Mnist Dataset is ready.")

# ========================== List of Helper Functions ==========================

def Ts():
    from datetime import datetime
    SysTime = datetime.now()
    TimeStamp = SysTime.strftime("%H-%M-%S")
    return TimeStamp


def mkdir(dir):
    from pathlib import Path
    Path(dir).mkdir(parents=True, exist_ok=True)


def log(fname:str, content:str, dir):
    mkdir(dir)
    TimeStamp = Ts()
    with open(f"{dir}{TimeStamp}-{fname}.txt","w+") as f:
        f.write(content)

# ==============================================================================


class SVDEmbedding:

    def __init__(this, X):
        assert type(X) is np.ndarray
        assert X.ndim == 2
        Mu = np.mean(X, axis=0, keepdims=True)
        StdX = (X - Mu)
        n, d = X.shape
        EigenValues, V = eigh((StdX.T@StdX)/n)
        this._V = V[:, ::-1]                 # reverse the order a bit
        this.n = n
        this.d = d
        this._EigenValues = EigenValues[::-1]
        this._Mu = Mu

    @property
    def V(this):
        return this._V.copy()

    @property
    def EigenValues(this):
        return this._EigenValues.copy()

    def Reconstruct(this, X:np.ndarray, k:int):
        assert X.ndim == 2
        assert X.shape[1] == this.d
        assert k <= this.d
        V = this._V[:, :k]
        return (V @ V.T @ (X.T - this._Mu)).T + this._Mu


    def ReconstructLoss(this, X, k:int):
        return norm(X - this.Reconstruct(X, k), "fro")**2/X.shape[0]

    def GetAnalysisFor(this, X, k, loss:bool=True):
        """
            Given a list of numbers denoting the set of: number of eigenvalues
            we want to use to reconstruct this data matrix, this will return a
            map mapping the number of eigenvalues, the reconstructed row data
            matrix, and the loss on the row data matrix.
        :param X:
        :param k:
        :param loss:
            Where you want the lossm or you want the
        :return:
        """
        assert np.sum(np.array(k) <= this.d)
        Reconstructed = zeros(X.shape)
        Res = []
        for II in tqdm(range(0, np.max(k))):
            Reconstructed[:, :] += \
                (this._V[:, II:II + 1] @ this._V[:, II:II + 1].T @ (X - this._Mu).T).T
            if II + 1 in k:
                Loss =  norm((X - this._Mu) - Reconstructed, "fro")**2/X.shape[0]
                Res.append((II + 1, Loss if loss else Reconstructed.copy()))
        return Res


def main():
    OutFolder = "./A3out"
    mkdir(OutFolder)
    Instance = SVDEmbedding(TRAIN_X)

    def A3a():
        # print out specific eigen values
        with open(f"{OutFolder}/{Ts()}-A3a-eigenvalue-sum.txt", "w+") as f:
            for II in [1, 2, 10, 30, 50]:
                f.write(f"{II} EigenValue {Instance.EigenValues[II - 1]}\n")
            f.write("The sum of all eigen values for the COVAR matrix is:\n")
            f.write(f"{np.sum(Instance.EigenValues)}\n")
    # A3a()

    def A3c():
        def PlotReconstructionError(X):
            Analysis = Instance.GetAnalysisFor(X, k=list(range(1, 101)))
            Ks = [Item[0] for Item in Analysis]
            MSELoss = [Item[1] for Item in Analysis]
            plt.plot(Ks, MSELoss)
        print("Getting Reconstruction graph for Train set")
        PlotReconstructionError(TRAIN_X)
        print("Getting Reconstruction graph for Test set")
        PlotReconstructionError(TEST_X)
        plt.legend(["Train Loss", "Test Loss"])
        plt.title("PCA Reconstruction MSE vs Number of Principle Modes")
        plt.savefig(f"{OutFolder}/{Ts()}-PCA-restruct-MSE.png")
        plt.show()
        # Accumuated Eigenvalues
        plt.plot(list(range(1, 101)), 1 - cumsum(Instance.EigenValues[:100])
                 / np.sum(Instance.EigenValues[:100]))
        plt.title("Eigenvalue Culmulative Energy")
        plt.xlabel("Sum of the first k eigenvalues")
        plt.ylabel("1 - Portion of used Eigenvalues")
        plt.savefig(f"{OutFolder}/{Ts()}-PCA-restruct-Energy.png")
        plt.show()
        plt.cla()

    A3c()

    def A3d():
        ToPlot = zeros((28*2, 28*5))
        for II in range(10):
            X = (II%5)*28
            Y = (II//5)*28
            V = Instance.V[:, II].reshape(28, 28)
            ToPlot[Y: Y + 28, X: X + 28] = V

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(ToPlot)
        fig.colorbar(cax)
        plt.title("Top 10 Principal Components")
        plt.savefig(f"{OutFolder}/{Ts()}-top10-principal-modes.png")
        plt.show()

    A3d()

    def A3e():
        from random import randint
        def RandomChooseDigits():
            Chosen = []
            for II in [2, 6, 7]:
                Indices = np.argwhere(TRAIN_Y == II).reshape(-1)
                Chosen.append(Indices[randint(0, len(Indices))])
            return Chosen
        ChosenDigits = RandomChooseDigits()
        Analysis = Instance.GetAnalysisFor\
        (
            TRAIN_X[ChosenDigits],
            loss=False,
            k=[5, 15, 40, 100]
        )
        ToPlot = zeros((3 * 28, 5 * 28))
        for II in range(3):
            Original = TRAIN_X[ChosenDigits[II]]
            Original = Original.reshape((28, 28))
            ToPlot[II*28:(II + 1)*28, :28] = Original
            for JJ, (k, X) in enumerate(Analysis):
                Reconstructed = X[II].reshape((28, 28))
                ToPlot[II*28: (II + 1)*28, (JJ + 1)*28: (JJ + 2)*28] = \
                    Reconstructed
        plt.matshow(ToPlot)
        plt.title("Reconstruction on train with k = [5, 15, 40, 100]")
        plt.savefig(f"{OutFolder}/{Ts()}-pca-reconstruction.png")
        plt.show()
    A3e()

    def A4d():
        from random import randint
        def RandomChooseDigits():
            Chosen = []
            for II in [2, 6, 7]:
                Indices = np.argwhere(TRAIN_Y == II).reshape(-1)
                Chosen.append(Indices[randint(0, len(Indices))])
            return Chosen

        ChosenDigits = RandomChooseDigits()
        Analysis = Instance.GetAnalysisFor \
                (
                TRAIN_X[ChosenDigits],
                loss=False,
                k=[32, 64, 128]
            )
        ToPlot = zeros((3 * 28, 4 * 28))
        for II in range(3):
            Original = TRAIN_X[ChosenDigits[II]]
            Original = Original.reshape((28, 28))
            ToPlot[II * 28:(II + 1) * 28, :28] = Original
            for JJ, (k, X) in enumerate(Analysis):
                Reconstructed = X[II].reshape((28, 28))
                ToPlot[II * 28: (II + 1) * 28, (JJ + 1) * 28: (JJ + 2) * 28] = \
                    Reconstructed
        plt.matshow(ToPlot)
        plt.title("Reconstruction on train with k = [32, 64, 128]")
        plt.savefig(f"{OutFolder}/{Ts()}-pca-reconstruction.png")
        plt.show()
    A4d()


if __name__ == "__main__":
    import os
    print(os.getcwd())
    print(os.curdir)
    main()