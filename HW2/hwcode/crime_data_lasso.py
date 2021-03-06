### Name: Hongda Alto Li
### Class: CSE 546
### This is for A5 of the HW2.
### Don't copy my code this is my code it has my style in it.
### !!!!!
### Requires: lasso.py

from lasso import LassoRegression, np, LassoLambdaMax
import pandas as pd
import matplotlib.pyplot as plt


def PrepareData():
    def ReadAndGet(fname):
        df = pd.read_table(fname)
        y = df["ViolentCrimesPerPop"].to_numpy()
        X = df.loc[:, df.columns != "ViolentCrimesPerPop"].to_numpy()
        print(f"{'='*10} File name: {fname} {'='*10}")
        print("Features set summary")
        print(df.loc[:, df.columns != "ViolentCrimesPerPop"].info())
        print("Labels set summary")
        print(df["ViolentCrimesPerPop"])
        print("features are already standardized. ")
        return X, y, df
    Xtrain, Ytrain, TrainDf = ReadAndGet("crime-train.txt")
    Xtest, Ytest, TestDf = ReadAndGet("crime-test.txt")
    return Xtrain, Xtest, Ytrain, Ytest, TrainDf, TestDf


def A5PlotsAndShow():
    print("summary on the data: ")
    Xtrain, Xtest, Ytrain, Ytest, TrainDf, TestDf = PrepareData()
    def A5c():
        print(f"{'='*10} Running A5(c), lambda non zeros count {'='*10}")
        LambdaMax = LassoLambdaMax(Xtrain, Ytrain)
        Lambda = LambdaMax                  # Results
        Lambdas = []                        # Results
        Ws = []                             # Predictors list
        Model = None                        # The model
        FlagPre, FlagCur = False, False     # whether lambda is < 0.01
        while (FlagPre) or (not FlagCur):   # not the case that Preivious is > 0.01, current is < 0.01
            print(f"Using Lambda: {Lambda}")
            FlagPre = FlagCur
            FlagCur = Lambda < 0.01         # Last one!
            if Model is None:
                Model = LassoRegression(regularization_lambda=Lambda, delta=1e-4)
            else:
                Model.Lambda = Lambda
            Model.fit(Xtrain, Ytrain)
            Ws.append(Model.w)
            Lambdas.append(Lambda)
            Lambda /= 2
        NoneZerosCount = [sum(_ != 0) for _ in Ws]
        plt.plot(Lambdas, NoneZerosCount)
        plt.xscale("log")
        plt.title("Crime Data non-zeros lasso")
        plt.xlabel("$\\lambda$")
        plt.ylabel("# of Non zeros in $\hat{w}$")
        plt.savefig("A5a-plot.png")
        plt.show()
        return Ws, Lambdas
    Ws, Lambdas = A5c()

    def A5d():
        # Features to pick up: agePct12t29,pctWSocSec,pctUrban,agePct65up
        Features = ["agePct12t29","pctWSocSec","pctUrban","agePct65up", "householdsize"]
        FeaturesIndices = []
        for II, Feature in enumerate(TrainDf.columns):
            ### FREATURES INDICES - 1 Because 2 has one element less than the original data frame!!!
            if Feature in Features: FeaturesIndices.append(II - 1)
        FeaturesLassoPath = np.array([w[FeaturesIndices, ...].reshape(-1) for w in Ws])
        for II in range(len(Features)):
            plt.plot(Lambdas, FeaturesLassoPath[:, II])
        plt.legend(Features)
        plt.xlabel("$\\lambda$")
        plt.ylabel("$value of $\\hat{w_j}$")
        plt.xscale("log")
        plt.title("Lasso Regulrization Path")
        plt.savefig("A5d-plot.png")
        plt.show()
    A5d()

    def A5e():
        print(f"{'='*10} Running A5(e), plotting the Squared Errors {'='*10}")
        def SquareLoss(Data, Labels):
            X = Data[np.newaxis, ...]
            y = Labels[np.newaxis, ...]
            y = y[..., np.newaxis]                         # 1 x n x 1
            Ws_local = np.array(Ws) # 1 x d x m
            SquareLoss = np.mean((X @ Ws_local - y) ** 2, axis=1).reshape(-1)
            return SquareLoss
        plt.plot(Lambdas, SquareLoss(Xtrain, Ytrain))
        plt.plot(Lambdas, SquareLoss(Xtest, Ytest))
        plt.title("Square Loss for different $\\lambda$ on Test set")
        plt.xlabel("$\\lambda$")
        plt.ylabel("Square Loss")
        plt.xscale("log")
        plt.legend(["Train Loss", "Test Loss"])
        plt.savefig("A5e-plot.png")
        plt.show()

    A5e()

    def A5f():
        print(f"{'='*10} Max, mean Model Parameters {'='*10}")
        Lambda = 30
        Model  = LassoRegression(regularization_lambda=Lambda)
        Model.fit(Xtrain, Ytrain)
        w = Model.w
        WLargestIndx = np.argmax(w) + 1
        WSmallestIdx = np.argmin(w) + 1
        print(f"Features with the largest Lasso Coefficient is: {TrainDf.columns[WLargestIndx]}")
        print(f"Features with the smallest Lasso Coefficient is: {TrainDf.columns[WSmallestIdx]}")
        print(f"The largest value is: {w[WLargestIndx - 1, ...]}")
        print(f"The smallest value is: {w[WSmallestIdx - 1, ...]}")
    A5f()


def main():
    A5PlotsAndShow()


if __name__ == "__main__":
    import os
    print(f"script running at: {os.curdir}")
    print(f"cwd: {os.getcwd()}")
    print(f"script is ready to run")
    main()