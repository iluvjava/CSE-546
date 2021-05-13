### This is a script for CSE 546 SPRING 2021, HW3, A.3
### Implementing the kernel ridge regression and visualize some stuff.
### Author: Hongda Li

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

linspace = np.linspace
randn = np.random.randn
pinvh = linalg.pinvh
inv = linalg.inv
eye = np.eye
mean = np.mean
std = np.std

class KernelRidge:

    def __init__(this, regularizer_lambda, kernelfunc: callable):
        """

        :param regularizer_lambda:
        :param kernelfunc:
            Takes in the WHOLE traning matrix and compute the kenrnel matrix K.

        """
        this.Lambda = regularizer_lambda
        this.KernelMatrix = None
        this.X = None
        this.Kernel = kernelfunc
        this.Alpha = None
        this.Bias = None

    def fit(this, X, y):
        """

        :param x:
        :param y:
        :return:
        """
        assert type(X) is np.ndarray and type(y) is np.ndarray, "X, y, must be numpy array"
        assert X.ndim == 2 and y.ndim <= 2
        Warn = "X, y dimension problem"
        if y.ndim == 2:
            assert y.shape[0] == X.shape[0], Warn
            assert y.shape[1] == 1, Warn
        else:
            assert y.shape[0] == X.shape[0], Warn
            y = y[:, np.newaxis]
        # Standardized.
        this.X = (X - mean(X, axis=0, keepdims=True))/std(X, axis=0, keepdims=True)
        n, d = X.shape
        Lambda = this.Lambda
        K = this.Kernel(this.X)
        assert K.ndim == 2 and K.shape[0] == K.shape[1] and K.shape[0] == n, \
            "your kernel function implemention is wrong, kernel matrix is having the wrong shape"
        assert np.all(np.abs(K-K.T) < 1e-9), "kernel matrix is not symmetric."
        this.KernelMatrix = K
        # get the bias
        this.Bias = mean(y.reshape(-1))
        # get the alpha.
        this.Alpha = pinvh(K + Lambda*eye(n))@(y - this.Bias)


    def predict(this, Xtest):
        assert this.X is not None, "Can't predict when not trained yet. "
        Xtrain = this.X
        XtestStd = (Xtest - mean(Xtest, axis=0, keepdims=True))/std(Xtest, axis=0, keepdims=True)
        return XtestStd@Xtrain.T@this.Alpha + this.Bias



def main():
    def SimpleTest():
        N = 100
        w, b = 1, 1
        x = linspace(0, 1000, N)
        eps = randn(N)*100
        y = w*x + b + eps
        X = x[:, np.newaxis]
        def KernelFunc(X):
            return X@X.T
        Model = KernelRidge(regularizer_lambda=0.01, kernelfunc=KernelFunc)
        Model.fit(X, y)
        Yhat = Model.predict(X)
        plt.plot(x, y)
        plt.plot(x, Yhat)
        plt.show()
    SimpleTest()


if __name__ == "__main__":
    main()
