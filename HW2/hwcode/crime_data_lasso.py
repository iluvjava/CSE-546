### Name: Hongda Alto Li
### Class: CSE 546
### This is for A5 of the HW2.
### Don't copy my code this is my code it has my style in it.

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
                Model = LassoRegression(regularization_lambda=Lambda)
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
        Features = ["agePct12t29","pctWSocSec","pctUrban","agePct65up"]
        FeaturesIndices = []
        for II, Feature in enumerate(TrainDf.columns):
            if Feature in Features: FeaturesIndices.append(II)
        FeaturesLassoPath = np.array([w[FeaturesIndices, ...] for w in Ws])
        for II, Feature in enumerate(TrainDf.columns):
            plt.plot(Lambdas, FeaturesLassoPath[:, II])
        plt.legend(Features)
        plt.xlabel("$\\lambda$")
        plt.ylabel("$value of $\\hat{w_j}$")
        plt.title("Lasso Regulrization Path")
        plt.savefig("A5d-plot.png")
        plt.show()
    A5d()

    def A5e():
        pass
    A5d()



def main():
    A5PlotsAndShow()


if __name__ == "__main__":
    import os
    print(f"script running at: {os.curdir}")
    print(f"cwd: {os.getcwd()}")
    print(f"script is ready to run")
    main()