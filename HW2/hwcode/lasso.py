"""
This code is for CSE 546, Spring 2021, Question A4.

Author: Hongda Alto Li, Github Account: iluvjava.

Please don't copy my code cause my code is well crafted and has my style in it.

"""

import numpy as np
array = np.array
zeros = np.zeros
norm = np.linalg.norm
inf = np.inf
mean = np.mean
sum = np.sum
abs = np.abs
sign = np.sign
ones = np.ones
max = np.max

class LassoRegression:

    def __init__(this, regularization_lambda, delta=1e-5, verbose=False):
        """
            create an instance of Lasso fit
        :param regularization_lambda:
        :param deltaw:
        :param verbose:
        """
        this.Lambda = regularization_lambda
        this.Verbose = verbose
        this.Delta = 1e-5

    def fit(this, X, y):
        assert type(X) is np.ndarray and type(y) is np.ndarray, \
            "Must use numpy array to train the lasso."
        assert len(X.shape) == 2 and len(y.shape) == 1, \
            "X must be row data matrix, 2d and y must be a 1d vector, numerical."
        assert X.shape[0] == y.shape[0], \
            "The number of rows of X must equal to the number elements in y. "
        MaxItr = 10000
        n, d = X.shape[0], X.shape[1]
        y2d = y[:, np.newaxis]
        deltaW = this.Delta*ones((n, 1))*1.1
        w = zeros((d, 1))
        l = this.Lambda
        Itr = 0
        while norm(deltaW, inf) > this.Delta and Itr < MaxItr:
            Itr += 1
            # compute offset vector.
            b = mean(y2d - X@w)
            # Compuate all the k at once, because it's not related to w_k
            a = 2*sum(X**2, axis=0)
            for k in range(d):
                a_k = a[k]
                Indices = [J for J in range(d) if J != k]
                c_k = 2*sum(
                    X[::, [k]]
                    *
                    (y2d - (b + X[:, Indices]@w[Indices]))
                )
                w_k = 0 if (abs(c_k) < l) else (c_k - sign(c_k)*l)/a_k
                deltaW = abs(w_k - w[k])
                w[k] = w_k
                this._print(f"optimizing on w_{k}, get w_{k} = {w[k]}")
        if MaxItr == Itr: raise Exception("Coordinate descent Max Itr reached without converging")
        this._weights = w
        this._b = b


    @property
    def w(this):
        return this._weights.copy()

    @property
    def b(this):
        return this._b.copy()

    def _print(this, mesg):
        if this.Verbose: print(mesg)


def LassoLambdaMax(X, y):
    """
        Given the samples matrix and the labels, the function reurns the minimal lambda
        such that after running the lasso algorithm, all the features model
        parameters will be set to zeros by this lambda.
    :param X:
        This is the row data matrix, n by d matrix.
    :param y:
        This is label vector, 1d vector with a length of d.
    :return:
    """
    assert type(X) is np.ndarray and type(y) is np.ndarray, \
        "Must use numpy array to train the lasso."
    assert len(X.shape) == 2 and len(y.shape) == 1, \
        "X must be row data matrix, 2d and y must be a 1d vector, numerical."
    assert X.shape[0] == y.shape[0], \
        "The number of rows of X must equal to the number elements in y. "
    y = y[:, np.newaxis]
    return max(2*abs(X.T@(y - mean(y))))


def GetLassoSyntheticTestdata(n, d, k):
    pass


def main():
    def SimpleTest():
        N, d = 40, 4
        X = np.random.rand(N, d)
        Noise = np.random.randn(N, 1)*1e-3
        Wtrue = np.random.randint(0, 100, (d, 1))
        y = X@Wtrue + Noise
        y = y.reshape(-1)

        LambdaMax = LassoLambdaMax(X, y)
        print(f"LambdaMax: {LambdaMax}")

        Model = LassoRegression(regularization_lambda=LambdaMax)
        Model.fit(X, y)
        print(Model.w)
        print(Model.b)

        Model = LassoRegression(regularization_lambda=LambdaMax/2)
        Model.fit(X, y)
        print(Model.w)
        print(Model.b)

        Model = LassoRegression(regularization_lambda=0)
        Model.fit(X, y)
        print(Model.w)
        print(Model.b)
    SimpleTest()


if __name__ == "__main__":
    import os
    print(f"cwd: {os.getcwd()}")
    print(f"dir: {os.curdir}")
    main()