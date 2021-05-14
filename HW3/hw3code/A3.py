### This is the script that produce plots and data for A3
### This is for CSE 546 SPRING 2021, HW3.
### Author: Hongda Alto Li
### Requries: kernel_ridge_regression.py


import numpy as np
cos, sin, pi = np.cos, np.sin, np.pi
rand, randn = np.random.rand, np.random.randn
norm = np.linalg.norm
zeros = np.zeros
mean = np.mean
sum = np.sum
min = np.min
max = np.max
linspace = np.linspace
logspace = np.logspace
from kernel_ridge_regression import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds

def RBFKernel(X, Y, gamma):
    """
        X is the row data matrix.
    :param x:
    :return:
    """

    return rbf_kernel(X, Y, gamma=gamma)


def MyPolyKernel(X, Y, d):
    """
        X is the row data matrix.
    :param x:
    :return:
    """
    if Y is None: X = Y
    return polynomial_kernel(X, Y, gamma=1, degree=d)


def main():
    n = 30 # Global parameter
    f = lambda x: 4 * sin(pi * x) * cos(6 * pi * x ** 2)
    KfoldSplit = 30
    def GenerateXY():
        x = rand(n)
        y = f(x) + randn(n)
        return x[:, np.newaxis], y

    def CrossValErrorEstimate(X, y, regualarizer, kernelfunc, param_norm=False):
        Errors = []
        AlphaNorm = []
        kf = KFold(n_splits=KfoldSplit, shuffle=False)
        for TrainIdx, TestIdx in kf.split(X):
            Model = KernelRidge(regularizer_lambda=regualarizer, kernelfunc=kernelfunc)
            Model.fit(X[TrainIdx], y[TrainIdx])
            yhat = Model.predict(X[TestIdx])
            Error = sum(yhat - y[TestIdx])**2
            Errors.append(Error)
            AlphaNorm.append(norm(Model.w, np.inf))
        if param_norm:
            return mean(Errors), min(AlphaNorm)
        return mean(Errors)

    def PolyKernelHypertune():
        MaxDegree = 30
        BestParameter = None
        SmallestError = float("inf")
        Errors = {}
        X, y = GenerateXY()
        for degree in range(1, MaxDegree + 1):
            Lambda = 16
            while True:
                Error, AlphaNorm = CrossValErrorEstimate(
                    X, y, Lambda, lambda X, Y: MyPolyKernel(X, Y, degree), True
                )
                Errors[Lambda, degree] = Error
                if Error < SmallestError:
                    SmallestError = Error
                    BestParameter = (degree, Lambda)
                    print(f"Best param updated [deg, lambda]:{BestParameter} ")
                if Lambda < 2**(-10):
                    break
                Lambda /=2
        print(f"Best hyperparam Seems to be: {BestParameter}")
        return BestParameter

    def GaussianKernelHypertune():
        # Grid search, Fix the training sample
        X, y = GenerateXY()
        def GetError(gamma, l):
            l, gamma = abs(l), abs(gamma)
            Kernelfun = lambda x, y: RBFKernel(x,y, gamma=gamma)
            Error = CrossValErrorEstimate(X, y, regualarizer=l, kernelfunc=Kernelfun)
            return Error
        # GRID SEARCH INITIAL GUESS
        Gammas, Lambdas = linspace(1, 20, 10), logspace(-8, 2, num=10, base=2)
        BestError = float("inf")
        BestParams = None
        for gammastart, gammaend in zip(Gammas[:-1], Gammas[1:]):
            GammaGuess = (gammastart + gammaend)/2
            for lstart, lend in zip(Lambdas[:-1], Lambdas[1:]):
                LambdaGuess = (lstart + lend)/2
                # Bounded optimization
                Res = minimize(lambda x: GetError(x[0], x[1]),
                               np.array([GammaGuess, LambdaGuess]),
                               bounds=[(gammastart, gammaend), (lstart, lend)],
                               method="Powell"
                               )
                Xmin = Res.x
                Fval = Res.fun
                if Fval < BestError:
                    BestError = Fval
                    BestParams = np.abs(Xmin)
                    print(f"Best param updated, [gamma, lambda] {BestParams}")
        print(f"Guassian Bestparams: {BestParams}")
        return BestParams

    GaussianBest = GaussianKernelHypertune()
    PolyBest = PolyKernelHypertune()
    print(f"guassian kernel best is: [gamma, lambda] {GaussianBest}")
    print(f"Poly kernel best is: [deg, lambda] {PolyBest}")
    X, y = GenerateXY()

    def DrawPolyModel():
        x = linspace(0, 1, 1000)
        Model = KernelRidge(regularizer_lambda=PolyBest[1],
                            kernelfunc= lambda X, Y: MyPolyKernel(X, Y, PolyBest[0]))
        Model.fit(X, y)
        plt.plot(x, Model.predict(x[:, np.newaxis]).reshape(-1))
        plt.scatter(X.reshape(-1), y, c="red")
        plt.title(f"poly kernel ridge regression\n degree: {PolyBest[0]}, lambda: {PolyBest[1]}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        plt.savefig("A3b-poly.png")
    DrawPolyModel()

    def DrawGuassianModel():
        x = linspace(0, 1, 1000)
        Model = KernelRidge(regularizer_lambda=GaussianBest[1],
                            kernelfunc=lambda X, Y: RBFKernel(X, Y, GaussianBest[0]))
        Model.fit(X, y)
        plt.plot(x, Model.predict(x[:, np.newaxis]).reshape(-1))
        plt.scatter(X.reshape(-1), y, c="red")
        plt.title(f"guassian kernel ridge regression\n gamma: {GaussianBest[0]}, lambda: {GaussianBest[1]}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        plt.savefig("A3b-gauss.png")
    DrawGuassianModel()

    # --------------------------------------------------------------------------
    # Bootstrap and estimating the confident interval.


if __name__ == "__main__":
    main()