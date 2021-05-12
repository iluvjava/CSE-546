### This is for CSE 546 2021 Spring
### Name: Honda Li
### Don't copy my code cause my code has my style in it.

import numpy as np
zeros = np.zeros
exp = np.exp
mean = np.mean
norm = np.linalg.norm
array = np.array
ones = np.ones
log = np.log
sign = np.sign

class BinaryLogisticRegression:

    def __init__(this, regularizer_lambda=1e-1, stepsize = 0.1):
        this._StepSize = stepsize
        this._Lambda = regularizer_lambda
        this._w = None
        this._b = None

    @property
    def StepSize(this):
        return this._StepSize

    @property
    def Lambda(this):
        return this._Lambda

    @property
    def w(this):
        return this._w.copy()

    @property
    def b(this):
        return this._b


    def GenerateGradients(this, X, y, grad=False):
        """
            This function is gonna yield all my parameters during the gradient
            descend iterations, given the samples and the labels, it does it,
            indefinitely.
            NOTE:
                There is no STANDARDIZATION ON THE DATA.
        :param X:
            The row data matrix. np array, n by k
        :param y:
            The 1D label vector, n elements in it, should be a vector of {1, -1}
            for binary label.
        :return:
            updated model parameters.
        """

        assert type(X) is np.ndarray and type(y) is np.ndarray, \
            "Must use numpy array to train the lasso."
        assert len(X.shape) == 2 and len(y.shape) == 1, \
            "X must be row data matrix," \
            " 2d and y must be a 1d vector, numerical."
        assert X.shape[0] == y.shape[0], \
            "The number of rows of X must equal to the number elements in y. "
        n, k = X.shape[0], X.shape[1]
        this._w = zeros((k, 1)) if this._w is None else this._w
        this._b = 0 if this._b is None else this._b
        while True:
            GradW, GradB = this._ComputeGradientFor(X, y)
            this._w += - GradW*this._StepSize
            this._b += - GradB*this._StepSize
            if grad:
                yield this.w, this.b, GradW, GradB
            else:
                yield this.w, this.b

    def _ComputeGradientFor(this, X, y):
        """
            An internal method for getting the gradient given the samples
            and the labels.
        :param X:
            Row data matrix.
        :param y:
            1d binary label vectors.
        :return:
        """
        if this._b is None: raise Exception("Parameters unestablished, "
                                            "can't compute gradient. ")
        # Get the parameters.
        w, b = this._w, this._b
        n, k = X.shape[0], X.shape[1]
        y = y[:, np.newaxis]  # n by 1
        def Mu(w, b): # This is a VECTOR function, returns a n by 1 VECTOR
            return 1 / (1 + exp(-y * (b + X @ w)))

        def gradientW(w, b):
            v = y * (Mu(w, b) - 1)
            return (1 / n) * (X.T @ v) + 2 * this._Lambda * w

        def gradientB(w, b):
            return mean(y * (Mu(w, b) - 1))

        return gradientW(w, b), gradientB(w, b)


    def UpdateParametersUsing(this, X, y):
        """
        This function updates and returns the parameters.
        :return:
            The new parameter after one step of gradient descent.
        """
        assert type(X) is np.ndarray and type(y) is np.ndarray, \
            "Must use numpy array to train the lasso."
        assert len(X.shape) == 2 and len(y.shape) == 1, \
            "X must be row data matrix," \
            " 2d and y must be a 1d vector, numerical."
        assert X.shape[0] == y.shape[0], \
            "The number of rows of X must equal to the number elements in y. "
        n, k = X.shape
        this._w = zeros((k, 1)) if this._w is None else this._w
        this._b = 0 if this._b is None else this._b
        GradW, GradB = this._ComputeGradientFor(X, y)
        this._w += - GradW * this._StepSize
        this._b += - GradB * this._StepSize
        return this.w, this.b


    def CurrentLoss(this, X, y):
        """
        Given a data and label, get loss with the newst model parameters.
        :param X:
            Row data matrix.
        :param y:
            1D label data.
        :return:
            A scaler for the logistic loss on the function.
        """
        assert type(X) is np.ndarray and type(y) is np.ndarray, \
            "Must use numpy array to train the lasso."
        assert len(X.shape) == 2 and len(y.shape) == 1, \
            "X must be row data matrix," \
            " 2d and y must be a 1d vector, numerical."
        assert X.shape[0] == y.shape[0], \
            "The number of rows of X must equal to the number elements in y. "
        n, k = X.shape
        y = y[:, np.newaxis]                                # n by 1
        w = zeros((k, 1)) if this._w is None else this._w   # k by 1
        b = 0 if this._b is None else this._b
        return mean(log(1 + exp(-y*(b + X@w)))) + this._Lambda*norm(w)**2

    def Predict(this, X):
        assert type(X) is np.ndarray, \
            "Must use numpy array to train the lasso."
        assert len(X.shape) == 2, "Row data matrix has to be 2 D"
        n, _ = X.shape
        if this._b is None:
            return zeros(n)
        return sign(X@this.w + this.b).astype(np.int).reshape(-1)


def main():
    def SimpleTest():
        def DistributionCenteredAt(a, b, N):
            return np.random.randn(N, 2)*0.0001 + array([[a, b]]), ones(N)
        N = 200
        X1, y1 = DistributionCenteredAt(0, -2, N)
        X2, y2 = DistributionCenteredAt(0, 2, N)
        X = zeros((2*N, 2))
        X[:N, :] = X1
        X[N:, :] = X2
        y = zeros(2*N)
        y[:N] = y1
        y[N:] = -y2
        Itr = 0
        Model = BinaryLogisticRegression(stepsize=1e-1)
        for w, b, gw, gb, in Model.GenerateGradients(X, y, grad=True):
            print(f"Model.CurrentLoss(X, y): {Model.CurrentLoss(X, y)}, w: {w}, b: {b}")
            print(f"Gradient w : {norm(gw)}")
            print(f"Graident b: {norm(gb)}")
            if Itr > 1000: break
            Itr += 1
    SimpleTest()



if __name__ == "__main__":
    import os
    print(f"script running in {os.curdir}")
    print(f"cwd: {os.getcwd()}")
    main()