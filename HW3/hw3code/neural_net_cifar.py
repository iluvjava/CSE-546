import torch
import torchvision
datasets = torchvision.datasets
transforms = torchvision.transforms
F = torch.nn.functional
nn = torch.nn
optim = torch.optim

import matplotlib.pyplot as plt
import copy

TRANSFORMS  = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
CIFAR_TRAIN = datasets.CIFAR10(root="./data",
                             train=True,
                             download=True,
                             transform=TRANSFORMS["train"])
CIFAR_TEST = datasets.CIFAR10(root="./data",
                             train=True,
                             download=True,
                             transform=TRANSFORMS["test"])
CIFAR_TRAIN, CIFAR_VAL = \
     torch.utils.data.random_split(CIFAR_TRAIN, [45000, 50000 - 45000])
# CIFAR_TRAIN, CIFAR_VAL = torch.utils.data.Subset(CIFAR_TRAIN, range(0, 5000, 3)),\
#                          torch.utils.data.Subset(CIFAR_TRAIN, range(1, 5000, 3))
CLASSES = \
    ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class MyCifar:

    def __init__(this, finetune=False):
        Model = torchvision.models.alexnet(pretrained=True)
        if not finetune:
            for Param in Model.parameters():
                Param.require_grad = False
        Model.classifier[-1] = nn.Linear(4096, 10)
        Model.to(DEVICE)
        this.Model = Model
        this.FineTune = finetune

    def __call__(this, *args, **kwargs):
        return this.Model(*args, **kwargs)

    def FeedForward(this, X, y):
        X, y = X.to(DEVICE), y.to(DEVICE)
        return F.cross_entropy(this(X), y)

    def predict(this, X):
        X = X.to(DEVICE)
        return torch.argmax(this(X), axis=1)

    @property
    def Parameters(this):
        if this.FineTune:
            return this.Model.parameters()
        return this.Model.classifier[-1].parameters()


def Run(finetune:bool, batchsize:int, epochs:int):
    Model = MyCifar(finetune=finetune)
    BatchSize = batchsize
    TrainSet = torch.utils.data.DataLoader(CIFAR_TRAIN,
                                            batch_size=BatchSize)
    TrainTotal = len(TrainSet)*BatchSize
    ValSet = torch.utils.data.DataLoader(CIFAR_VAL,
                                         batch_size=BatchSize)
    TestSet = torch.utils.data.DataLoader(CIFAR_TEST, batch_size=BatchSize)
    ValTotal = len(ValSet)*BatchSize
    Optimizer = optim.RMSprop(Model.Parameters, lr=0.1/BatchSize)
    Epochs = epochs
    TrainLosses, ValLosses, TrainAccuracy, ValAccuracy = [], [], [], []
    BestModel, BestAccuracy = None, 0
    for II in range(Epochs):
        AvgLoss = Correct = 0
        for X, y in TrainSet:
            Optimizer.zero_grad()
            Loss = Model.FeedForward(X, y)
            Loss.backward()
            Optimizer.step()
            with torch.no_grad():
                AvgLoss += float(Model.FeedForward(X, y)) / len(TrainSet)
                Correct += float(torch.sum(Model.predict(X).to("cpu") == y))/TrainTotal
        TrainLosses.append(AvgLoss)
        TrainAccuracy.append(Correct)
        print(f"Epoch: {II}, Train Loss: {AvgLoss}, Train Acc: {TrainAccuracy[-1]}", end="; ")
        Correct = AvgLoss = 0
        for X, y in ValSet:
            with torch.no_grad():
                Loss = Model.FeedForward(X, y)
                Correct += float(torch.sum(Model.predict(X).to("cpu") == y))/ValTotal
            AvgLoss += float(Loss)/len(ValSet)
        ValAccuracy.append(Correct)
        ValLosses.append(AvgLoss)
        print(f"Val Loss: {AvgLoss}, Val Acc: {ValAccuracy[-1]}")
        if ValAccuracy[-1] > BestAccuracy:
            BestAccuracy = ValAccuracy[-1]
            BestModel = copy.deepcopy(Model.Model.state_dict())

    # Plot train, val losses
    plt.plot(TrainLosses)
    plt.plot(ValLosses)
    plt.legend(["Train Loss", "Val Loss"])
    plt.title(f"Train and Validation Loss, finetune: {finetune}")
    plt.xlabel("Epoch")
    plt.savefig(f"A5a-train-val-loss-{finetune}.png")
    plt.show()

    # Plot train val acc
    plt.plot(TrainAccuracy)
    plt.plot(ValAccuracy)
    plt.legend(["Train Acc", "Val Acc"])
    plt.title(f"Train and Validation Accuracy, finetune: {finetune}")
    plt.xlabel("Epoch")
    plt.savefig(f"A5a-train-val-acc-{finetune}.png")
    plt.show()

    # Test Accuracy
    TestAcc = 0
    Model = MyCifar()
    Model.Model.load_state_dict(BestModel)
    Model.Model.eval()
    for X, y in TestSet:
        with torch.no_grad():
            TestAcc += torch.sum(Model.predict(X).to("cpu") == y)/len(CIFAR_TEST)
    print(f"TestAcc: {TestAcc}")
    with open("a5-test-acc.txt", "w+") as f:
        f.write(TestAcc)
    return Model


def main():
    Run(True, 100, epochs=30)
    Run(False, 500, epochs=20)

if __name__ == "__main__":
    import os
    print(f"current dir:{os.curdir}")
    print(f"cwd: {os.getcwd()}")
    Model = main()