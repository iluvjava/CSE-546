# name: Hongda Li
# This is for A6 HW3 CSE 546 SPRING 2021
# Don't copy my code it has my style in it.


import torch
import torchvision
datasets = torchvision.datasets
transforms = torchvision.transforms
F = torch.nn.functional
nn = torch.nn
optim = torch.optim
sqrt = torch.sqrt
from tqdm import tqdm

from scipy.optimize import shgo
import matplotlib.pyplot as plt
import copy

TRANSFORMS = {
    'train': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ]),
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
CIFAR_TRAIN = datasets.CIFAR10(root="./data",
                             train=True,
                             download=True,
                             transform=TRANSFORMS["val"])
CIFAR_TEST = datasets.CIFAR10(root="./data",
                             train=True,
                             download=True,
                             transform=TRANSFORMS["val"])
# CIFAR_TRAIN, CIFAR_VAL = \
#      torch.utils.data.random_split(CIFAR_TRAIN,
#                                    [45000, 50000 - 45000])
CIFAR_TRAIN, CIFAR_VAL = torch.utils.data.Subset(CIFAR_TRAIN,
                                                 range(0, 5000)),\
                         torch.utils.data.Subset(CIFAR_TRAIN,
                                                 range(1000, 2000))
CLASSES = \
    ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def BatchThisModel(theModel, theDataLoader, optimizer=None, dataTransform:callable=None):
    """
        Performs one epochs of training or inference, depending on whether
        optimizer is None or not.
    :param theModel:
    :param theDataLoader:
    :param optimizer:
    :return:
    """
    theModel.to(DEVICE)
    AvgLoss = 0
    AvgAccuracy = 0
    for X, y in theDataLoader:
        if dataTransform:
            X = dataTransform(X)
        X, y= X.to(DEVICE), y.to(DEVICE)
        if optimizer is not None: # GD
            optimizer.zero_grad()
            Loss = theModel.FeedForward(X, y)
            AvgLoss += Loss.item()
            Loss.backward()
            optimizer.step()

        else: # Inference
            with torch.no_grad():
                Loss = theModel.FeedForward(X, y)
                AvgLoss += Loss.item()
        with torch.no_grad():
            AvgAccuracy +=\
                float(torch.sum(theModel.predict(X) == y)/(len(theDataLoader)*len(y)))
    return AvgAccuracy

def GetTrainValDataLoader(bs):
    TrainSet = torch.utils.data.DataLoader(CIFAR_TRAIN,
                                           batch_size=bs)
    ValSet = torch.utils.data.DataLoader(CIFAR_VAL,
                                         batch_size=bs)
    return TrainSet, ValSet


class BestModelRegister:
    """
    Pass to the trainer and stores all the losses and data from the training process.
    """
    def __init__(this):
        this.BestModel = None
        this.ModelType = None
        # Hyper parameters maps to Epochs Acc
        this.HyperParameterAccList = {}
        # Best Acc maps to best Params tuple
        this.BestAccToParams = {}
        # Absolute Best ACC from whatever agorithm run the hypertune.
        this.BestAcc = 0

    def save(this):
        import time
        from pathlib import Path
        Path("./a6bestmodel").mkdir(parents=True, exist_ok=True)
        torch.save(this.BestModel.state_dict(), f"./a6bestmodel/{time.strftime('%H-%M-%S-%b-%d-%Y')}")

    def Top9AccList(this):
        """
            Get the hyper params with top 10 accuracy.
        :return:
        """
        List = list(this.BestAccToParams.keys())
        List.sort(reverse=True)
        List = List[:min(9, len(List))]
        Result = {}
        for K in List:
            Result[K] = this.HyperParameterAccList[this.BestAccToParams[K]]
        return Result

    def ProducePlotPrintResult(this):
        from pathlib import Path
        Path("./a6bestmodel").mkdir(parents=True, exist_ok=True)
        ModelTypeMap = {1: "Logistic", 2: "Single Hidden", 3:"CNN"}
        TheLegends = [9, 1, 2, 3, 4, 5, 6, 7, 8]
        TopList = this.Top9AccList()
        # Plot the training acc
        for _, V in TopList.items():
            plt.plot(V[0])
        plt.xlabel("Epochs")
        plt.ylabel("Their Train acuracy")
        plt.title(f"Model: {ModelTypeMap[this.ModelType]} Top 9 ranked by peak val acc")
        plt.savefig(f"A6-{ModelTypeMap[this.ModelType]}-train-acc.png")
        plt.legend([f"top {R}" for R in TheLegends])
        plt.show()
        # Plot the validation acc
        for _, V in TopList.items():
            plt.plot(V[1])
        plt.xlabel("Epochs")
        plt.ylabel("Their Val acuracy")
        plt.title(f"Model: {ModelTypeMap[this.ModelType]} Top 9 ranked by peak val acc")
        plt.savefig(f"A6-{ModelTypeMap[this.ModelType]}-val-acc.png")
        plt.legend([f"top {R}" for R in TheLegends])
        plt.show()
        # Plot the top 1 model found:
        plt.plot(V[0])
        plt.plot(V[1])
        plt.legend(["train", "val"])
        plt.xlabel("epochs")
        plt.ylabel("acc")
        plt.savefig(f"A6-{ModelTypeMap[this.ModelType]}-best-acc.png")
        plt.show()
        # write the results
        with open(f"./a6bestmodel/top9A6 {ModelTypeMap[this.ModelType]}.txt", "w+") as f:
            if this.ModelType == 1:
                f.write(f"BatchSize, Learning_Rate\n")
                for K, _ in TopList.items():
                    Params = this.BestAccToParams[K]
                    f.write(f"{Params[0]}, {Params[1]}\n")
            elif this.ModelType == 2:
                f.write(f"BatchSize, Learning_Rate, Hidden_Layer_width\n")
                for K, _ in TopList.items():
                    Params = this.BestAccToParams[K]
                    f.write(f"{Params[0]}, {Params[1]}, {Params[2]}\n")
            else:
                assert False, "Unrecognized type, or not yet implemented"




class ModelA(nn.Module):
    def __init__(this):
        super().__init__()
        this.Linear = nn.Linear(3*32**2, 10)

    def FeedForward(this, X, y):
        yhat = this(X)
        return F.cross_entropy(yhat, y)

    def forward(this, X):
        x = this.Linear(X)
        return x

    def predict(this, X):
        return torch.argmax(this(X), axis=1)

    @staticmethod
    def GetHyperTuneFunc(Epochs, verbose, modelRegister):
        modelRegister.ModelType = 1
        def flatten(x):
            return x.view(x.shape[0], -1)
        def HyperTuneFunc(x, mem={}):
            """
            This function is passed to SHGO for optimization.
            :param x:
            :param mem:
            :return:
            """
            BatchSize, Lr = int(x[0]), x[1]
            if (BatchSize, Lr) in mem:
                return mem[BatchSize, Lr]
            Model = ModelA()
            if verbose: print(f"ModelA, Hypertune, Bs: {BatchSize}, lr: {Lr}")
            Optimizer = optim.Adam(Model.parameters(), lr=Lr)
            T, V = GetTrainValDataLoader(BatchSize)
            BestAcc = -float("inf")
            TrainAcc, ValAcc = [], []
            for II in tqdm(range(Epochs)):
                Acc = BatchThisModel(Model, T, optimizer=Optimizer, dataTransform=flatten)
                TrainAcc.append(Acc)
                Acc = BatchThisModel(Model, V, dataTransform=flatten)
                ValAcc.append(Acc)
                if Acc > BestAcc:
                    BestAcc = Acc
            modelRegister.HyperParameterAccList[BatchSize, Lr] = (TrainAcc, ValAcc)
            modelRegister.BestAccToParams[BestAcc] = (BatchSize, Lr)
            if Acc > modelRegister.BestAcc:
                modelRegister.BestAcc = Acc
                modelRegister.BestModel = Model
                if verbose: print(f"Best Acc Update: {Acc}")
            mem[BatchSize, Lr] = 1 - Acc # Memeorization.
            return 1 - Acc
        return HyperTuneFunc


class ModelB(nn.Module):
    def __init__(this, hiddenWidth):
        super().__init__()
        this.L1 = nn.Linear(3*32**2, hiddenWidth)

    def FeedForward(this, X, y):
        return F.cross_entropy(this(X), y)

    def forward(this, X):
        x = this.L1(X)
        x = F.relu(x)
        return x

    def predict(this, X):
        return torch.argmax(this(X), axis=1)

    @staticmethod
    def GetHyperTuneFunc(Epochs, verbose, modelRegister):
        modelRegister.ModelType = 2
        def flatten(x):
            return x.view(x.shape[0], -1)
        def HyperTuneFunc(x, mem={}):
            """
            This function is passed to SHGO for optimization.
            :param x:
            :param mem:
            :return:
            """
            BatchSize, Lr, HiddenLayerWidth= int(x[0]), x[1], int(x[2])
            if (BatchSize, Lr, HiddenLayerWidth) in mem:
                return mem[BatchSize, Lr, HiddenLayerWidth]
            Model = ModelB(HiddenLayerWidth)
            if verbose: print(f"ModelB, Hypertune, Bs: {BatchSize}, lr: {Lr}, HLW: {HiddenLayerWidth}")
            Optimizer = optim.Adam(Model.parameters(), lr=Lr)
            T, V = GetTrainValDataLoader(BatchSize)
            BestAcc = -float("inf")
            TrainAcc, ValAcc = [], []
            for II in tqdm(range(Epochs)):
                Acc = BatchThisModel(Model, T, optimizer=Optimizer, dataTransform=flatten)
                TrainAcc.append(Acc)
                Acc = BatchThisModel(Model, V, dataTransform=flatten)
                ValAcc.append(Acc)
                if Acc > BestAcc:
                    BestAcc = Acc
            modelRegister.HyperParameterAccList[BatchSize, Lr, HiddenLayerWidth] = (TrainAcc, ValAcc)
            modelRegister.BestAccToParams[BestAcc] = (BatchSize, Lr, HiddenLayerWidth)
            if Acc > modelRegister.BestAcc:
                modelRegister.BestAcc = Acc
                modelRegister.BestModel = Model
                if verbose: print(f"Best Acc Update: {Acc}")
            mem[BatchSize, Lr] = 1 - Acc  # Memeorization.
            return 1 - Acc

        return HyperTuneFunc


class ModelC(torch.nn.Module):
    def __init__(this):
        super().__init__()

    def FeedForward(this):
        pass


def main():
    def TuneModel1():
        ModelRegister = BestModelRegister()
        TheFunc = ModelA.GetHyperTuneFunc(20, True, ModelRegister)
        result = shgo(TheFunc,
             [(20, 500), (1e-4, 0.01)], options={"maxev":10, "ftol": 1e-2, "maxfev": 3})
        print(result)
        print(ModelRegister.Top9AccList())
        ModelRegister.ProducePlotPrintResult()

    def TuneModel2():
        ModelRegister = BestModelRegister()
        TheFunc = ModelB.GetHyperTuneFunc(15, True, ModelRegister)
        result = shgo(TheFunc,
                      [(100, 1000), (1e-5, 0.01), (50, 800)], options={"maxev": 10, "ftol": 1e-2, "maxfev": 10})
        print(result)
        print(ModelRegister.Top9AccList())
        ModelRegister.ProducePlotPrintResult()
    TuneModel2()

if __name__ == "__main__":
    import os
    print(f"curdir: {os.curdir} ")
    print(f"cwd: {os.getcwd()}")
    main()