### This is the script that produce plots and data for A3
### This is for CSE 546 SPRING 2021, HW3.
### Author: Hongda Alto Li
### Requries: kernel_ridge_regression.py


import numpy as np
cos, sin, pi = np.cos, np.sin, np.pi
rand, randn, randint = np.random.rand, np.random.randn, np.random.randint
norm = np.linalg.norm
zeros = np.zeros
mean = np.mean
sum = np.sum
min = np.min
max = np.max
linspace = np.linspace
logspace = np.logspace
percentile = np.percentile
from kernel_ridge_regression import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.optimize import minimize, shgo
from scipy.optimize import Bounds

### Some constant for the whole script:




def RBFKernel(X, Y, gamma):
    """
        Use this kernel to take the inner products between the columns of X, Y
    :param x:
    :return:
    """

    return rbf_kernel(X, Y, gamma=gamma)


def MyPolyKernel(X, Y, d):
    """
        Use this kernel to take the inner product between the columns of the X, Y
        matrix.
    :param x:
    :return:
    """
    if Y is None: X = Y
    return polynomial_kernel(X, Y, gamma=1, degree=d)


def CrossValErrorEstimate(X, y, regualarizer, kernelfunc, split=None,param_norm=False):
    Errors = []
    AlphaNorm = []
    if split is None:
        split = X.shape[0]
    kf = KFold(n_splits=split, shuffle=False)

    for TrainIdx, TestIdx in kf.split(X):
        Model = KernelRidge(regularizer_lambda=regualarizer, kernelfunc=kernelfunc)
        Model.fit(X[TrainIdx], y[TrainIdx])
        yhat = Model.predict(X[TestIdx])
        Error = (sum(yhat - y[TestIdx])**2)/len(TestIdx)
        Errors.append(Error)
        AlphaNorm.append(norm(Model.w, np.inf))
    if param_norm:
        return mean(Errors), min(AlphaNorm)
    return mean(Errors)


def main(n=30, KfoldSplit=30):
    f = lambda x: 4 * sin(pi * x) * cos(6 * pi * x ** 2)
    def GenerateXY():
        x = rand(n)
        y = f(x) + randn(n)
        return x[:, np.newaxis], y

    X, y = GenerateXY()   #  THIS IS SHARED! FOR ALL

    def PolyKernelHypertune():
        def GetError(deg, l):
            Kernefun = lambda x, y: MyPolyKernel(x, y, deg)
            Error = CrossValErrorEstimate(X,
                                          y,
                                          regualarizer=l,
                                          kernelfunc=Kernefun,
                                          split=KfoldSplit)
            return Error
        Result = shgo(lambda x: GetError(x[0], x[1]),
             bounds=[(1, 30), (0, 2*n)],
             n=300, sampling_method="sobol")
        print(f"SHGO Optimization Results: {Result}")
        return (Result.x[0], 0)

    def GaussianKernelHypertune():
        # Grid search, Fix the training sample
        # X, y = GenerateXY()
        def GetError(gamma, l):
            l, gamma = abs(l), abs(gamma)
            Kernelfun = lambda x, y: RBFKernel(x,y, gamma=gamma)
            Error = CrossValErrorEstimate(X,
                                          y,
                                          regualarizer=l,
                                          kernelfunc=Kernelfun,
                                          split=KfoldSplit
                                          )
            return Error
        # GRID SEARCH INITIAL GUESS
        Result = shgo(lambda x: GetError(x[0], x[1]),
                      bounds=[(0, 1000), (0, 10*n)],
                      n=300, sampling_method='sobol')
        print("Optimization results: ")
        print(Result)
        print(f"Guassian Bestparams: {Result.x}")
        return Result.x
    # ============ Hyper Param! ================================================
    # GaussianBest = [20, 0.058]
    # PolyBest = [19, 0.05] # PolyKernelHypertune()

    GaussianBest = GaussianKernelHypertune()
    PolyBest = PolyKernelHypertune()

    print(f"guassian kernel best is: [gamma, lambda] {GaussianBest}")
    print(f"Poly kernel best is: [deg, lambda] {PolyBest}")

    def DrawPolyModel():
        x = linspace(0, 1, 1000)
        Model = KernelRidge(regularizer_lambda=PolyBest[1],
                            kernelfunc=
                            lambda X, Y: MyPolyKernel(X, Y, PolyBest[0]))
        Model.fit(X, y)
        plt.plot(x, Model.predict(x[:, np.newaxis]).reshape(-1))
        plt.scatter(X.reshape(-1), y, c="red")
        plt.title(f"poly kernel ridge regression\n "
                  f"degree: {PolyBest[0]}, lambda: {PolyBest[1]}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        plt.savefig("A3b-poly.png")
        return Model

    BestPolyModel = DrawPolyModel()

    def DrawGuassianModel():
        x = linspace(0, 1, 1000)
        Model = KernelRidge(regularizer_lambda=GaussianBest[1],
                            kernelfunc=
                            lambda X, Y: RBFKernel(X, Y, GaussianBest[0])
                            )
        Model.fit(X, y)
        plt.plot(x, Model.predict(x[:, np.newaxis]).reshape(-1))
        plt.scatter(X.reshape(-1), y, c="red")
        plt.title(f"guassian kernel ridge regression\n "
                  f"gamma: {GaussianBest[0]}, lambda: {GaussianBest[1]}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        plt.savefig("A3b-gauss.png")
        return Model
    BestGaussianModel = DrawGuassianModel()

    # --------------------------------------------------------------------------
    # Boopstrap and estimating the confident interval.

    def BoopStraping(Kernelfunc, Lambda, x):
        # Given the kernel func, produce the confidence interval.
        BagOfModels = []
        UpperPercentile = []
        LowerPercentile = []
        print("Boopstraping fitting the model")
        for _ in range(300):
            Indices = randint(0, n, 30)
            XTild, Ytild = X[Indices], y[Indices]
            Model = KernelRidge(
                kernelfunc=Kernelfunc,
                regularizer_lambda=Lambda
            )
            Model.fit(XTild, Ytild)
            BagOfModels.append(Model)
        ModelPredictions = np.array(
            [Model.predict(x[:, np.newaxis]).reshape(-1) for Model in BagOfModels])
        for II in range(ModelPredictions.shape[1]):
            UpperPercentile.append(percentile(ModelPredictions[:, II], 95))
            LowerPercentile.append(percentile(ModelPredictions[:, II], 5))

        return UpperPercentile, LowerPercentile


    def BoopStrapModelDifference(GaussModel, PolyModel):
        m = 1000

        for _ in range(300):
            pass

        pass

    Xgrid = linspace(0, 1, 100)
    UpperPercentile, LowerPercentile = BoopStraping(
        lambda x, y: MyPolyKernel(x, y, PolyBest[0]),
        PolyBest[1], Xgrid
    )
    # Plot the polynomial Boop strap,
    plt.title("Bootstrap Confident bands for Poly Kernel Ridge Regression\n "
              f"Degree: {round(PolyBest[0], 3)}, Lambda: {round(PolyBest[1], 3)}, 95, 5 % interval")
    # plt.plot(Xgrid, UpperPercentile)
    # plt.plot(Xgrid, LowerPercentile)
    plt.ylim([max(y) * 1.1, min(y) * 1.1])
    plt.fill_between(Xgrid, UpperPercentile, LowerPercentile, color='b', alpha=.1)
    plt.plot(Xgrid, BestPolyModel.predict(Xgrid[:, np.newaxis]))
    plt.scatter(X.reshape(-1), y, c="red")
    plt.savefig("Poly-boopstraped.png")
    plt.show()
    ## Plot the Gaussian Boopstrap
    UpperPercentile, LowerPercentile = BoopStraping(
        lambda x, y: RBFKernel(x, y, GaussianBest[0]),
        GaussianBest[1], Xgrid
    )
    plt.title("Bootsrap Confident Bands for Gaussian Kernel Ridge Regression\n"
              f"Gamma: {round(GaussianBest[0], 3)}, Lambda: {round(GaussianBest[1], 3)}, 95, 5 % interval")
    # plt.plot(Xgrid, UpperPercentile)
    # plt.plot(Xgrid, LowerPercentile)
    plt.ylim([max(y) * 1.1, min(y) * 1.1])
    plt.fill_between(Xgrid, UpperPercentile, LowerPercentile, color='b', alpha=.1)
    plt.plot(Xgrid, BestGaussianModel.predict(Xgrid[:, np.newaxis]))
    plt.scatter(X.reshape(-1), y, c="red")
    plt.savefig("gaussian-boopstraped.png")
    plt.show()

    # A3 Part (e). Additional Boopstrap to compare the models.
    if n == 300 and KfoldSplit == 10:
        print("a3 (3) not yet implemented yet. ")
        pass





if __name__ == "__main__":
    main(300, 10)