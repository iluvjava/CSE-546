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
var = np.var
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
    kf = KFold(n_splits=split, shuffle=False) # Make it deterministic given the same training sample.

    for TrainIdx, TestIdx in kf.split(X):
        Model = KernelRidge(regularizer_lambda=regualarizer, kernelfunc=kernelfunc)
        Model.fit(X[TrainIdx], y[TrainIdx])
        yhat = Model.predict(X[TestIdx]).reshape(-1)
        Error = (sum(yhat - y[TestIdx])**2)/len(TestIdx)
        Errors.append(Error)
        AlphaNorm.append(norm(Model.w, np.inf))
    if param_norm:
        return mean(Errors), min(AlphaNorm)
    return mean(Errors)


def GenerateXY(n):
    f = lambda x: 4 * sin(pi * x) * cos(6 * pi * x ** 2)
    x = rand(n)
    x.sort()
    y = f(x) + randn(n)
    return x[:, np.newaxis], y


def main(n=30, KfoldSplit=30):
    X, y = GenerateXY(n)   #  THIS IS SHARED! FOR ALL
    f = lambda x: 4 * sin(pi * x) * cos(6 * pi * x ** 2)
    def PolyKernelHypertune():
        def GetError(deg, l):
            Kernefun = lambda x, y: MyPolyKernel(x, y, deg)
            Error = CrossValErrorEstimate(X,
                                          y,
                                          regualarizer=l,
                                          kernelfunc=Kernefun,
                                          split=KfoldSplit)
            return Error
        BestError = float("inf")
        Best = None
        for Deg in np.linspace(7, 31):
            Result = shgo(lambda x: GetError(Deg, x),
                bounds=[(0, 0.05)],
                n=100,
                sampling_method="simplicial",
                options={"f_tol": 1e-8}
                          )
            if Result.fun < BestError:
                print(f"Poly Kernel Best error update for deg: {Deg}, Lambda: {Result.x}, Error: {Result.fun}")
                BestError = Result.fun
                Best = (Deg, Result.x[0])
            else:
                print(f"failed at: deg: {Deg}, Lambda: {Result.x[0]}, Error: {Result.fun}")

        # Result = shgo(lambda x: GetError(x, 0),
        #      bounds=[(1, 100)],
        #      n=200, sampling_method="simplicial",
        #               options={"f_tol": 1e-8, "disp": True})
        # print(f"SHGO Optimization Results: {Result}")
        return Best

    def GaussianKernelHypertune():
        # Grid search, Fix the training sample
        # X, y = GenerateXY()
        Xs = X.reshape(-1)
        Distance = []
        for II in range(Xs.size):
            for JJ in range(II + 1, Xs.size):
                Distance.append(1/(norm(Xs[II] - Xs[JJ])**2))
        GammaLower, GammaHigher = \
            percentile(Distance, 25), percentile(Distance, 75)
        print(f"Hypertune Gamma kernel search range: {GammaLower, GammaHigher}")
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
                      bounds=[(GammaLower, GammaHigher), (0, 1)],
                      n=500, sampling_method='sobol',
                      options={"f_tol": 1e-4, "disp": True})
        print("Optimization results: ")
        print(Result)
        print(f"Guassian Bestparams: {Result.x}")
        return Result.x
    # ============ Hyper Param! ================================================
    GaussianBest = [20, 0.058]
    PolyBest = [19, 0.05]

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
        plt.ylim([max(y)*1.1, min(y)*1.1])
        plt.plot(x, Model.predict(x[:, np.newaxis]).reshape(-1))
        plt.plot(x, f(x))
        plt.scatter(X.reshape(-1), y, c="red")
        plt.title(f"poly kernel ridge regression\n "
                  f"degree: {PolyBest[0]}, lambda: {PolyBest[1]}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["Poly", "Truth", "Data Points"])
        plt.savefig("A3b-poly.png")
        plt.show()
        return Model

    BestPolyModel = DrawPolyModel()

    def DrawGuassianModel():
        x = linspace(0, 1, 1000)
        Model = KernelRidge(regularizer_lambda=GaussianBest[1],
                            kernelfunc=
                            lambda X, Y: RBFKernel(X, Y, GaussianBest[0])
                            )
        Model.fit(X, y)
        plt.ylim([min(y)*1.1, max(y)*1.1])
        plt.plot(x, Model.predict(x[:, np.newaxis]).reshape(-1))
        plt.plot(x, f(x))
        plt.scatter(X.reshape(-1), y, c="red")
        plt.title(f"guassian kernel ridge regression\n "
                  f"gamma: {GaussianBest[0]}, lambda: {GaussianBest[1]}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["Gaussian", "Truth", "Data Points"])
        plt.savefig("A3b-gauss.png")
        plt.show()
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
        X, y = GenerateXY(m)
        AllMeanDiff = []
        print("solving A3(e)")
        for _ in range(300):
            Idx = randint(0, m, m)
            Idx.sort()
            Idx = np.unique(Idx)
            Xtild, ytild = X[Idx], y[Idx]
            yhat1 = GaussModel.predict(Xtild).reshape(-1)
            yhat2 = PolyModel.predict(Xtild).reshape(-1)
            Var1 = var(yhat1 - ytild)
            Var2 = var(yhat2 - ytild)
            AllMeanDiff.append(Var2 - Var1)
            print(f"Bootstrap sample: {_}, The difference of MSE is: {AllMeanDiff[-1]}")
        UpperPercentile, LowerPercentile =\
            percentile(AllMeanDiff, 95), percentile(AllMeanDiff, 5)
        return LowerPercentile, UpperPercentile


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
    plt.plot(Xgrid, BestPolyModel.predict(Xgrid[:, np.newaxis]))
    plt.plot(Xgrid, f(Xgrid))
    plt.fill_between(Xgrid, UpperPercentile, LowerPercentile, color='b', alpha=.1)
    plt.scatter(X.reshape(-1), y, c="red")
    plt.legend(["poly", "truth", "confidentband", "data"])
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
    plt.plot(Xgrid, BestGaussianModel.predict(Xgrid[:, np.newaxis]))
    plt.plot(Xgrid, f(Xgrid))
    plt.fill_between(Xgrid, UpperPercentile, LowerPercentile, color='b', alpha=.1)
    plt.scatter(X.reshape(-1), y, c="red")
    plt.legend(["guassian", "truth", "confidentband", "data"])
    plt.savefig("gaussian-boopstraped.png")
    plt.show()

    # A3 Part (e). Additional Boopstrap to compare the models.
    if n == 300 and KfoldSplit == 10:
        print("For A3 (e) we are going to use the idea of "
              "boopstrap to find the confidence interval of the"
              " best MSE model minus the Poly Model ")

        LowerPercentile, UpperPercentile = \
            BoopStrapModelDifference(BestGaussianModel, BestPolyModel)
        print(f"The Upper and lower bound of the confidence interval is:"
              f" {LowerPercentile, UpperPercentile}")
        with open("goodplots/A3e.txt", "a") as file:
            file.write(f"Upper, Lower bound: {LowerPercentile, UpperPercentile}")



if __name__ == "__main__":
    import os
    print(f"cwd: {os.getcwd()}")
    main(n=300, KfoldSplit=10)
    #main()