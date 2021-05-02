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
randn = np.random.randn

import matplotlib.pyplot as plt
plot = plt.plot
xscale = plt.xscale
show = plt.show
title = plt.title
xlabel = plt.xlabel
ylabel = plt.ylabel
legend = plt.legend
saveas = plt.savefig

class LassoRegression:

    def __init__(this, regularization_lambda, delta=1e-4, verbose=False):
        """
            create an instance of Lasso fit
        :param regularization_lambda:
            The lambda for L2 norm
        :param delta:
            The tolerance for measuring the convergence of the w parameter for coordinate descend.
        :param verbose:
            Whether to print out all the message when doing the coordinate descned.
        """
        this.Lambda = regularization_lambda
        this.Verbose = verbose
        this.Delta = delta
        this._weights = None
        this._b = None

    def fit(this, X, y):
        """
            Fit that data.
            NOTE: Data standardization is not used.
        :param X:
            Row data matrix. Should be n by d where n is the number of samples and d is the number of features.
        :param y:
            Label vector, strictly on dimensional.
        :return:
            This model.
        """
        assert type(X) is np.ndarray and type(y) is np.ndarray, \
            "Must use numpy array to train the lasso."
        assert len(X.shape) == 2 and len(y.shape) == 1, \
            "X must be row data matrix, 2d and y must be a 1d vector, numerical."
        assert X.shape[0] == y.shape[0], \
            "The number of rows of X must equal to the number elements in y. "
        MaxItr = 10000
        n, d   = X.shape[0], X.shape[1]
        y2d    = y[:, np.newaxis]
        deltaW = this.Delta*ones((d, 1))*1.1                   # Amount of changes for each predictor while iterating
        w      = zeros((d, 1)) if (this._weights is None) else this._weights.copy()
                                                               # !!! Use previous for optimization if model is
                                                               # asked to optimize for a second time.
        l = this.Lambda                                        # Regularizer !!!
        Itr = 0
        while norm(deltaW, inf) > this.Delta and Itr < MaxItr:
            Itr += 1
            b    = mean(y2d - X@w)     # compute offset vector.
            a    = 2*sum(X**2, axis=0) # Compuate all the k at once, because it's not related to w_k
            for k in range(d):
                a_k     = a[k]
                Indices = [J for J in range(d) if J != k]
                c_k     = 2*sum(
                                X[::, [k]]
                                *
                                (y2d - (b + X[:, Indices]@w[Indices]))
                            )
                w_k       = 0 if (abs(c_k) < l) else (c_k - sign(c_k)*l)/a_k
                deltaW[k, ...] = abs(w_k - w[k, ...])
                w[k]      = w_k
            this._print(f"delta w is: {deltaW.reshape(-1)}")
            this._print(f"lambda is: {this.Lambda}")
        if MaxItr == Itr: raise Exception("Coordinate descent Max Itr reached without converging")
        this._weights = w
        this._b = b
        return this

    def predict(this, X):
        if this.w is None:
            raise Exception("Can't predict on a lasso that is not trained yet")
        d = this.w.shape[0]
        assert d == X.shape[1], "The number of features used to predict doesn't match with what is trained on "
        return X@this.w + this.b

    @property
    def w(this):   # get the weights of the model.
        return this._weights.copy()

    @property
    def b(this):   # Get the offset of the model.
        return this._b.copy()

    def _print(this, mesg):  # print out the message if in verbose mode.
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


def GetLassoSyntheticTestdata(n:int, d:int, k:int, sigma=1):
    assert (n > 0) and (d > 0) and (k > 0), "n, d, k all have to be larger than zeros"
    assert k < n, "k has to be < n"
    WTruth = array([JJ/k if JJ <= k else 0 for JJ in range(1, d + 1)], dtype=np.float)[:, np.newaxis]
    Noise = np.random.randn(n, 1)*sigma
    X = randn(n, d) # std normal for best stability of the coordinate descend algorithm.
    return X, (X@WTruth + Noise).reshape(-1), WTruth


def A4a_b():

    # Part (a)
    n, d, k = 500, 1000, 100
    X, y, Wtrue = GetLassoSyntheticTestdata(n, d, k)
    LambdaMax = LassoLambdaMax(X, y)
    Ws = []
    # Feature Chosen for each lambda
    Lfc = [] # lambda and features count
    Lambda = LambdaMax
    r = 1.1
    while len(Lfc) == 0 or Lfc[-1][1] < k:
        Model = LassoRegression(regularization_lambda=Lambda)
        Model.fit(X, y)
        NonZeros = sum(Model.w != 0)
        Ws.append(Model.w)
        print(f"NonZeros: {NonZeros}")
        Lfc.append((Lambda, NonZeros))
        Lambda /= r
    plot([_[0] for _ in Lfc], [_[1] for _ in Lfc], "ko")
    xscale("log")
    xlabel(f"$\lambda$, reduction ratio: {r}")
    ylabel("Non Zeroes $w_j$")
    title("A4: Nonezeros $W_j$ vs Lambda for Lasso")
    plt.savefig("A4a-plot.png")
    show()


    # Part (b)
    # The first k elements in Wtrue is always going to be non-zeroes.
    # FDR: (Incorrect Nonzeroes in w_hat)/(total number of nonzeroes in w_hat)
    # TPR: (# of correct non-zeroes in w_har)/(k)

    WTrueNonZeroes = Wtrue != 0
    FDR = []
    TPR = []
    Lambdas = [_[0] for _ in Lfc]
    for WHat in Ws:
        WHatNonZeros = WHat != 0
        if sum(WHatNonZeros) == 0:
            Lambdas.pop(0)
            continue
        FDR.append(sum(WHatNonZeros * ~WTrueNonZeroes)/sum(WHatNonZeros))
        TPR.append(sum(WHatNonZeros[:100])/k)
    plot(Lambdas, FDR)
    plot(Lambdas, TPR)
    title("FDR vs TPR")
    xlabel("$\lambda$")
    xscale("log")
    legend(["FDR", "TPR"])
    plt.savefig("A4b-plot.png")
    show()


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
    A4a_b()


if __name__ == "__main__":
    import os
    print(f"cwd: {os.getcwd()}")
    print(f"dir: {os.curdir}")
    main()