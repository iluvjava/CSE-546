### CLASS CSE 564 SPRING 2021 HW4 A4
### Name: Hongda Li
### My code has my style in it don't copy.

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from mnist import MNIST
from tqdm import tqdm
zeros = np.zeros
randint = np.random.randint
randn = np.random.randn


if "MNIST_DATA" not in dir(): # running on interactive console will be faster
    datasets.MNIST('./data', download=True, train=True)
    MNIST_DATA = MNIST("./data/MNIST/raw/")
    TRAIN_X, _= MNIST_DATA.load_training()
    TEST_X, _ = MNIST_DATA.load_testing()
    TRAIN_X= np.array(TRAIN_X, dtype=np.float)/255
    TEST_X = np.array(TEST_X, dtype=np.float)/255
    print("Mnist Dataset is ready.")



# ======================= Helper Functions =====================================

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


# ==============================================================================

class KMean:

    def __init__(this, k:int, X:np.ndarray):
        """

        :param k: Number of cluster
        :param X: Row data matrix in np array type
        """
        assert k < X.shape[0] and k > 1
        assert X.ndim == 2
        n, d= X.shape[0], X.shape[1]
        this._X = X
        # this._AugX = X[:, :, np.newaxis]
        this.Assignment = {}
        this._C = np.transpose(this._X[randint(0, n, k), :][...,np.newaxis],
                               (2, 1, 0))
        this._Distances = np.zeros((n, 1, k))
        this._ComputeAssignment()



    @property
    def Centroids(this):
        return np.transpose(this._C, (2, 1, 0))[..., 0].copy()
    @property
    def X(this):
        return this._X.copy()
    @property
    def AugX(this):
        return this._AugX.copy()
    @property
    def C(this):
        return this._C.copy()

    def TransferLearningFrom(this, other):
        this._C = other.C
        this._ComputeAssignment()

    def _ComputeCentroid(this):
        """
            Compute centroid using the current assignment.

        :return:
        """
        for Centroid, Idx in this.Assignment.items():
            this._C[..., Centroid] = \
                np.mean(this._X[Idx], axis=0, keepdims=True)

    def _ComputeAssignment(this):
        """
        Given current centroids make an assignment.
        :return:
        """

        for CIdx in range(this._C.shape[2]):
            Centroid = this._C[..., CIdx]
            this._Distances[..., CIdx] = \
                np.sum((this._X - Centroid)**2, axis=1, keepdims=True)

        AssignmentVec = np.argmin(this._Distances, axis=2).reshape(-1)
        NewAssignment = {}
        for Idx, Class in enumerate(AssignmentVec):
            IdxArr = NewAssignment.get(Class, [])
            IdxArr.append(Idx)
            NewAssignment[Class] = IdxArr
        this.Assignment = NewAssignment

    def Update(this):
        this._ComputeCentroid()
        this._ComputeAssignment()

    def Loss(this, Xtest=None):
        X = this._X if Xtest is None else Xtest
        Centroids = this.Centroids
        Loss = 0
        for CentroidIdx, Idx in this.Assignment.items():
            Loss += np.sum((X[Idx] - Centroids[CentroidIdx, :])**2)
        return Loss/X.shape[0]


def main():
    def BasicTest():
        Points1 = randn(1000, 2)
        Points2 = np.array([[3, 3]]) + randn(1000, 2)
        PointsAll = np.concatenate((Points1, Points2), axis=0)
        Km = KMean(X=PointsAll, k=2)
        Losses = []
        for II in range(10):
            Km.Update()
            Losses.append(Km.Loss())
        plt.plot(Losses)
        plt.show()
        return Km

    def Learn(Km:KMean, n=None):
        Losses = []
        if n is not None:
            for _ in tqdm(range(n)):
                Km.Update()
                Losses.append(Km.Loss())
        else:
            C = Km.Centroids
            while True:
                Km.Update()
                Losses.append(Km.Loss())
                Delta = np.linalg.norm(C - Km.Centroids, np.inf)
                print(f"Delta: {Delta}")
                if Delta < 1e-1:
                    break
                C = Km.Centroids
        return Km, Losses

    def ClusterMnist(k=10,X=None):
        if X is None: X = TRAIN_X
        Km, Losses = Learn(KMean(X=X,k=k))
        return Km, Losses


    def A5b():
        Km, Losses = Learn(KMean(X=TRAIN_X,k=10))
        plt.plot(Losses)
        plt.title("A5(b) Kmean k=10")
        plt.xlabel("Iteration")
        plt.ylabel("Average Loss")
        mkdir("./A5bplots")
        plt.savefig(f"./A5bplots/{Ts()}-A5b-k=10-losses.png")
        plt.show()
        AllCentroid = zeros((28*2, 28*5))
        for Idx, Centroid in enumerate(Km.Centroids):
            Image = Centroid.reshape((28, 28))
            VerticalOffset, HorizontalOffset = (Idx//5)*28, (Idx%5)*28
            AllCentroid[VerticalOffset:VerticalOffset+28,
            HorizontalOffset:HorizontalOffset+28] = Image
        plt.matshow(AllCentroid)
        plt.title("A5(b):Cenroids fond by Kmean")
        plt.savefig(f"./A5bplots/{Ts()}-A5b-k=10-centroids.png")
        plt.show()

    def A5c():
        NumberOfCluster = list(map(lambda x: 2**x,range(1, 7)))
        TrainLosses, TestLosses = [], []
        for K in NumberOfCluster:
            Km, Losses = ClusterMnist(k=K, X=TRAIN_X[:5000])
            TrainLosses.append(Losses[-1])
            TestLosses.append(Km.Loss(TEST_X))
        plt.plot(NumberOfCluster, TrainLosses, ".-")
        plt.plot(NumberOfCluster, TestLosses, ".-")
        plt.legend(["Losses on Train Set", "Losses on Test Set"])
        plt.title("K-Mean on MNIST, Cluster Number vs Loss")
        plt.xlabel("Number of Cluster")
        plt.ylabel("Loss")
        plt.savefig(f"./A5bplots/{Ts()}-A5b-k-vs-loss.png")
        plt.show()
    A5b()
    A5c()




if __name__ == "__main__":
    import os
    print(f"{os.getcwd()}")
    print(f"{os.curdir}")
    main()
