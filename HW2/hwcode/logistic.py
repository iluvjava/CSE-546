### This is for CSE 546 2021 Spring
### Name: Honda Li
### Don't copy my code cause my code has my style in it.

import numpy as np
zeros = np.zeros
exp = np.exp
mean = np.mean

class BinaryLogisticRegression:

    def __init__(this, regularizer_lambda=1e-1, stepsize = 0.1):
        this._StepSize = stepsize
        this._Lambda = regularizer_lambda

    def fit(this, X, y):
        assert type(X) is np.ndarray and type(y) is np.ndarray, \
            "Must use numpy array to train the lasso."
        assert len(X.shape) == 2 and len(y.shape) == 1, \
            "X must be row data matrix," \
            " 2d and y must be a 1d vector, numerical."
        assert X.shape[0] == y.shape[0], \
            "The number of rows of X must equal to the number elements in y. "
        n, k = X.shape[0], X.shape[1]
        y = y[:, np.newaxis]  # n by 1
        w = zeros((k, 1))     # k by 1
        b = 0
        def Mu(w, b):
            # This is a VECTOR function, returns a n by 1 VECTOR
            return 1/(1 + exp(y*(b - X@w)))

        def gradientW(w, b):
            v = y*(1 - Mu(w, b))
            return (-1/n)*(X.T@v)

        def gradientB(w, b):
            return -mean(y*Mu(w, b))


    



def main():
    pass

if __name__ == "__main__":
    main()