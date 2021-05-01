

class LassoRegression:

    def __init__(this, regularization_lambda, delta=1e-5, verbose=False):
        """
            create an instance of Lasso fit 
        :param regularization_lambda:
        :param deltaw:
        :param verbose:
        """
        this.Lambda = regularization_lambda
        this.Verbose = True
        this.Delta = 1e-5

    def fit(this, X, y):
        assert type(X) is np.ndarray and type(y) is np.ndarray, "Must use numpy array to train the lasso."
        assert len(X.shape) == 2 and len(y.shape) == 1, "X must be row data matrix, 2d and y must be a 1d vector, numerical."
        assert X.shape[0] == y.shape[0], "The number of rows of X must equal to the number elements in y. "
        n, d = X.shape[0], X.shape[1]
        y2d = y[:, np.newaxis]
        deltaW = zeros((n, 1))
        w = zeros((d, 1))
        l = this.Lambda
        while norm(deltaW, inf) > this.Delta:
            # compute offset vector.
            b = mean(y2d - X@w)
            # Compuate all the k at once, because it's not related to w_k
            a = sum(2*X**2, axis=1)
            for k in range(d):
                a_k = a[k]
                Indices = [J for J in range(k) if J != k]
                c_k = 2*sum(
                    X[::, k]
                    *
                    (y2d - (b + X[:, Indices]@w[Indices]))
                )
                w_k = 0 if (abs(c_k) < l) else (c_k + sign(-l - c_k)*l)/a_k
                deltaW = abs(w_k - w[k])
                w[k] = w_k

        this._weights = w
        this._b = b

    @property
    def w(this):
        return this._weights.copy()

    @property
    def b(this):
        return this.b.copy()



    def _print(this, mesg):
        if this.Verbose: print(mesg)






def main():
    print(type(array([1, 2, 3])))
    print(type(array([1, 2, 3])) is np.ndarray)


if __name__ == "__main__":
    import os
    print(f"cwd: {os.getcwd()}")
    print(f"dir: {os.curdir}")
    import numpy as np
    array = np.array
    zeros = np.zeros
    norm = np.linalg.norm
    inf = np.inf
    mean = np.mean
    sum = np.sum
    abs = np.abs
    sign = np.sign
    main()