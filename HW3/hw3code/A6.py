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

#DEVICE = "cpu"
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
                                                 range(0, 10000)),\
                         torch.utils.data.Subset(CIFAR_TRAIN,
                                                 range(10000, 10000 + 1000))
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

def GetTS():
    import time
    Ts = time.strftime('%H-%M-%S-%b-%d-%Y')
    return Ts



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
        import time
        Ts = time.strftime('%H-%M-%S-%b-%d-%Y')

        ModelTypeMap = {1: "Logistic", 2: "Single Hidden", 3:"CNN"}
        TheLegends = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        TopList = this.Top9AccList()
        # Plot the training acc
        for _, V in TopList.items():
            plt.plot(V[0])
        plt.xlabel("Epochs")
        plt.ylabel("Their Train acuracy")
        plt.title(f"Model: {ModelTypeMap[this.ModelType]} Top 9 ranked by peak val acc")
        plt.legend([f"top {R}" for R in TheLegends])
        plt.savefig(f"./a6bestmodel/{Ts}-{ModelTypeMap[this.ModelType]}-train-acc.png")
        plt.show()
        # Plot the validation acc
        for _, V in TopList.items():
            plt.plot(V[1])
        plt.xlabel("Epochs")
        plt.ylabel("Their Val acuracy")
        plt.title(f"Model: {ModelTypeMap[this.ModelType]} Top 9 ranked by peak val acc")
        plt.legend([f"top {R}" for R in TheLegends])
        plt.savefig(f"./a6bestmodel/{Ts}-{ModelTypeMap[this.ModelType]}-val-acc.png")
        plt.show()
        # Plot the top 1 model found:
        plt.plot(TopList[max(TopList.keys())][0])
        plt.plot(TopList[max(TopList.keys())][1])
        plt.legend(["train", "val"])
        plt.xlabel("epochs")
        plt.ylabel("acc")
        plt.title("Best model train val acc")
        plt.savefig(f"./a6bestmodel/{Ts}-{ModelTypeMap[this.ModelType]}-best-acc.png")
        plt.show()
        # write the results


        with open(f"./a6bestmodel/top9A6-{Ts}" +
                  f"-{ModelTypeMap[this.ModelType]}.txt", "w+") as f:
            if this.ModelType == 1:

                f.write(f"max_val_acc, BatchSize, Learning_Rate\n")
                for K, _ in TopList.items():
                    Params = this.BestAccToParams[K]
                    f.write(f"{K}, {Params[0]}, {Params[1]}\n")
            elif this.ModelType == 2:
                f.write(f"max_val_acc, BatchSize, Learning_Rate, Hidden_Layer_width\n")
                for K, _ in TopList.items():
                    Params = this.BestAccToParams[K]
                    f.write(f"{K}, {Params[0]}, {Params[1]}, {Params[2]}\n")
            elif this.ModelType == 3:
                f.write(f"max_val_acc, BatchSize, Learning_Rate, Num_Channels, Conv_Kernel, MaxPool_Kernel\n")
                for K, _ in TopList.items():
                    Params = this.BestAccToParams[K]
                    f.write(f"{K}, {Params[0]}, {Params[1]}, {Params[2]}, {Params[3]}, {Params[4]}\n")
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
            mem[BatchSize, Lr, HiddenLayerWidth] = 1 - Acc  # Memeorization.
            return 1 - Acc

        return HyperTuneFunc


class ModelC(torch.nn.Module):
    def __init__(this, c, k1, k2):
        """

        :param c:
            Chennel for conv2d
        :param k1:
            Kernel for conv2d
        :param k2:
            Kernel for max pool
        """
        super().__init__()
        this.Con = nn.Conv2d(3, c, k1)
        this.Mp = nn.MaxPool2d(k2, k2)
        Width = int((32 - k1 + 1)/k2)
        this.L1 = nn.Linear(c*Width**2, 10)

    def FeedForward(this, X, y):
        return F.cross_entropy(this(X), y)

    def forward(this, X):
        x = this.Con(X)
        x = F.relu(x)
        x = this.Mp(x)
        x = torch.flatten(x, 1)
        return this.L1(x)

    def predict(this, X):
        return torch.argmax(this(X), axis=1)


    @staticmethod
    def GetHyperTuneFunc(Epochs, verbose, modelRegister):
        modelRegister.ModelType = 3
        def HyperTuneFunc(x, mem={}):
            """
            This function is passed to SHGO for optimization.
            :param x:
            :param mem:
            :return:
            """
            BatchSize, Lr, Channels, Kernel1, Kernel2 = \
                int(x[0]), x[1], int(x[2]), int(x[3]), int(x[4])
            if (BatchSize, Lr, Channels, Kernel1, Kernel2) in mem:
                return mem[BatchSize, Lr, Channels, Kernel1, Kernel2]
            Model = ModelC(Channels, Kernel1, Kernel2)
            if verbose: print(f"ModelC, Bs: {BatchSize}, lr: {Lr}," +
                              f" Channels: {Channels}, K1: {Kernel1}, K2: {Kernel2}")
            Optimizer = optim.Adam(Model.parameters(), lr=Lr)
            T, V = GetTrainValDataLoader(BatchSize)
            BestAcc = -float("inf")
            TrainAcc, ValAcc = [], []
            for II in tqdm(range(Epochs)):
                Acc = BatchThisModel(Model, T, optimizer=Optimizer)
                TrainAcc.append(Acc)
                Acc = BatchThisModel(Model, V)
                ValAcc.append(Acc)
                if Acc > BestAcc:
                    BestAcc = Acc
            modelRegister.HyperParameterAccList[BatchSize, Lr, Channels, Kernel1, Kernel2] \
                = (TrainAcc, ValAcc)
            modelRegister.BestAccToParams[BestAcc] = (BatchSize, Lr, Channels, Kernel1, Kernel2)
            if Acc > modelRegister.BestAcc:
                modelRegister.BestAcc = Acc
                modelRegister.BestModel = Model
                if verbose: print(f"Best Acc Update: {Acc}")
            mem[BatchSize, Lr, Channels, Kernel1, Kernel2] = 1 - Acc  # Memeorization.
            return 1 - Acc

        return HyperTuneFunc


def main():
    def TuneModel1():
        ModelRegister = BestModelRegister()
        TheFunc = ModelA.GetHyperTuneFunc(20, True, ModelRegister)
        result = shgo(TheFunc,
             [(100, 100), (5e-6, 0.01)],
             options={"maxev":50, "ftol": 1e-2, "maxfev": 10})
        print(result)
        print(ModelRegister.Top9AccList())
        ModelRegister.ProducePlotPrintResult()
        TestSet = torch.utils.data.DataLoader(CIFAR_VAL,
                                         batch_size=2000)
        Acc = BatchThisModel(ModelRegister.BestModel,
                             TestSet,
                             dataTransform=lambda x: x.view(x.shape[0], -1))
        with open(f"./a6bestmodel/{GetTS()}-best-model-logistic-test.txt", "w+") as f:
            f.write(str(Acc))


    def TuneModel2():
        ModelRegister = BestModelRegister()
        TheFunc = ModelB.GetHyperTuneFunc(20, True, ModelRegister)
        result = shgo(TheFunc,
                      [(100, 100), (1e-6, 0.01), (20, 3000)],
                      options={"maxev": 20, "ftol": 1e-2, "maxfev": 10})
        print(result)
        print(ModelRegister.Top9AccList())
        ModelRegister.ProducePlotPrintResult()
        TestSet = torch.utils.data.DataLoader(CIFAR_VAL,
                                              batch_size=2000)
        Acc = BatchThisModel(ModelRegister.BestModel, TestSet,
                             dataTransform=lambda x: x.view(x.shape[0], -1))
        with open(f"./a6bestmodel/{GetTS()}-best-model-hidden-test.txt", "w+") as f:
            f.write(str(Acc))

    def TuneModel3():
        ModelRegister = BestModelRegister()
        TheFunc = ModelC.GetHyperTuneFunc(20, True, ModelRegister)
        result = shgo(TheFunc,
                      [(100, 100), (1e-6, 0.01), (10, 200), (2, 5), (2, 4)],
                      options={"maxev": 20, "ftol": 1e-2, "maxfev": 10})
        print(result)
        print(ModelRegister.Top9AccList())
        ModelRegister.ProducePlotPrintResult()
        TestSet = torch.utils.data.DataLoader(CIFAR_VAL,
                                              batch_size=2000)
        Acc = BatchThisModel(ModelRegister.BestModel, TestSet)
        with open(f"./a6bestmodel/{GetTS()}-best-model-hidden-test.txt", "w+") as f:
            f.write(str(Acc))
        pass
    #TuneModel2()
    # TuneModel1()
    TuneModel3()

if __name__ == "__main__":
    import os
    print(f"curdir: {os.curdir} ")
    print(f"cwd: {os.getcwd()}")
    print(f"Pytorch device: {DEVICE}")
    main()
