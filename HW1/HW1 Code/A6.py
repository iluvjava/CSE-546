### This file constains code for solving HW1 A6 for class CSE 546/446 SPRING 2021.
from mnist import MNIST
import numpy as np
arr = np.array
eye = np.eye
pinv = np.linalg.pinv
argmax = np.argmax


def load_dataset():
    mndata = MNIST("./data/")
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255
    X_test = X_test/255
    return X_train, X_test, labels_train, labels_test


def train(X, Y, reg_lambda):
    """


    Args:
        X: Should be a n by 784 matrix, with all the samples pack vertically as rows into the matrix.
        Y: Should be a n by 10 matrix, comtaninig all the labels for the digits pack vertically as rows for the matrix.
        reg_lambda:
            This is the regularization constant for the system.
    Returns:
        The trained linear model for the system, should be a 784 by 10 matrix such that its transpose multiply by the
        feature vector will produce the label vector.

    """
    Y = Y.astype(np.float64)
    return pinv(X.T@X + reg_lambda*eye(X.shape[1]))@X.T@Y


def predict(W, Xnew):
    """

    Args:
        W: Should be a 784 by 10 matrix, which is the linear model.
        Xnew: Should be a n by 784 matrix that contains all the samples we want to to predict with using
        this given model.
    Returns:
        A single vector containing all the digits predicted using this model.
    """
    return argmax(W.T@Xnew.T, axis=0)


def main():
    X1, X2, Y1, Y2 = load_dataset()
    print(X1.shape) # (60000, 784)
    print(X2.shape) # (10000, 784)
    print(Y1.shape) # (60000, )
    print(Y2.shape) # (10000, )
    print(X1.dtype)
    print(X2.dtype)
    print(Y1.dtype)
    print(Y2.dtype)
    print("Ok we are ready to train the model and make prediction now. ")

    def TrainTheModel(X1, Y1):
        Y = (np.array([[II] for II in range(10)]) == Y1).astype(np.float)
        return train(X1, Y.T, reg_lambda=1e-4)
    def ErrorRate(y1, y2):
        return sum(y1 != y2)/y1.size

    W = TrainTheModel(X1, Y1)
    PredictedLabels = predict(W, X2)
    print(f"The error rate on the test set is: {ErrorRate(PredictedLabels, Y2)}")
    Predictedlabels = predict(W, X1)
    print(f"The error rate on the train set is: {ErrorRate(Predictedlabels, Y1)}")


if __name__ == "__main__":
    import os
    print(f"cwd: {os.getcwd()}")
    print(f"script running at {os.curdir}")
    main()
