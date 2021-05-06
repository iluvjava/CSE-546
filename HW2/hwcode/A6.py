### This is for CSE 546, SPRING 2021, HW2, Problem A6
### I am going to use the implementation in logistic.py to get the loss
### For minist under different scienario, and gradient descent strategies.
### !!!!!
### Requires: logistic.py
### minst_python, and the data for all the MINIST Images.

from mnist import MNIST
from logistic import BinaryLogisticRegression, np

zeros = np.zeros
exp = np.exp
mean = np.mean
norm = np.linalg.norm
array = np.array
ones = np.ones
where = np.where

def load_dataset():
    """
        Load the data MNIST Data set.
    :return: Standardized MNIST, split into training and test set.
    """
    mndata = MNIST("./data/")
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255
    X_test = X_test/255
    return X_train, X_test, labels_train, labels_test


def main():
    def Extract2And7(X, y):
        Indices = where((y == 2) | (y == 7))[0]
        X = X[Indices, :]
        y = y[Indices]
        y = (y == 2).astype(np.float) - (y == 7).astype(np.float)
        return X, y

    XTrain, XTest, YTrain, YTest = load_dataset()
    XTrain, YTrain = Extract2And7(XTrain, YTrain)
    XTest, YTest = Extract2And7(XTest, YTest)
    def A6a():
        pass

    def A6b():
        pass

    def A6c():
        pass





if __name__ == "__main__":
    import os
    print(f"Script is runnning at: {os.curdir}")
    print(f"cwd:{os.getcwd()}")
    main()


