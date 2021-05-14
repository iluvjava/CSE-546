### let's try out some of the hyper parameters we found for the model from the other file.



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
    n = 30  # Global
    f = lambda x: 4 * sin(pi * x) * cos(6 * pi * x ** 2)

    def GenerateXY():
        x = rand(n)
        y = f(x) + randn(n)
        return x[:, np.newaxis], y
    pass

    def PolyKernelExample():
        X, y = GenerateXY()

        Model = KernelRidge(regularizer_lambda=0.03018816458288159,
                            kernelfunc=lambda X, Y: MyPolyKernel(X, Y, 39))
        Model.fit(X, y)
        x = np.linspace(min(X), max(X), 100)

        yhat = Model.predict(x[:, np.newaxis])
        plt.scatter(X.reshape(-1), y, c="red")
        plt.plot(x, yhat)
        plt.show()
    PolyKernelExample()

    def GaussianKernelExample():
        X, y = GenerateXY()

        Model = KernelRidge(regularizer_lambda=0.21884506,
                            kernelfunc=lambda X, Y: RBFKernel(X, Y, gamma=172.91284562))
        Model.fit(X, y)
        x = np.linspace(0, 1, 100)

        yhat = Model.predict(x[:, np.newaxis])
        plt.scatter(X.reshape(-1), y, c="red")
        plt.plot(x, yhat)
        plt.show()
    GaussianKernelExample()

if __name__ == "__main__":
    main()