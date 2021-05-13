### This is a script for CSE 546 SPRING 2021, HW3, A.3
### Implementing the kernel ridge regression and visualize some stuff.
### Author: Hongda Li

import numpy as np
import scipy

invh = scipy.linalg.invh
eye = np.eye
mean = np.mean

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
        this.OffSet = None

    def fit(this, X, y):
        """

        :param x:
        :param y:
        :return:
        """
        assert type(X) == np.array and type(y) == np.array, "X, y, must be numpy array"
        assert X.ndim == 2 and y.ndim <= 2
        Warn = "X, y dimension problem"
        if y.ndim == 2:
            assert y.shape[0] == X.shape[0], Warn
            assert y.shape[1] == 1, Warn
        else:
            assert y.shape[0] == X.shape[0], Warn
            y = y[-1, np.newaxis]
        this.X = X
        n, d = X.shape
        Lambda = this.Lambda
        K = this.Kernel(X)
        assert K.ndim == 2 and K.shape[0] == K.shape[2] and K.shape[0] == n, \
            "your kernel function implemention is wrong, kernel matrix is having the wrong shape"
        assert np.all(np.abs(K-K.T) < 1e-9), "kernel matrix is not symmetric."
        this.KernelMatrix = K
        ## Standardize
        this.OffSet = mean(X, axis=1, keepdims=X.shape[1] != 1)
        this.Alpha = invh(K + Lambda*eye(n))@y


    def predict(this, Xtest):
        assert this.X is not None, "Can't predict when not trained yet. "
        Xtest = Xtest - this.OffSet
        Xtrain = this.X
        return Xtest@Xtrain.T@this.Alpha + this.OffSet



def main():
    import matplotlib.pyplot as plt
    linspace = np.linspace
    randn = np.random.randn()
    def SimpleTest():
        N = 100
        w, b = 1, 1
        x = linspace(0, 1, N)
        eps = randn(N)*0.1
        y = w*x + b + eps
        X = x[:, np.newaxis]

        pass
    pass


if __name__ == "__main__":
    main()
