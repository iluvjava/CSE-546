### This file constains code for solving HW1 B2 for class CSE 546/446 SPRING 2021.
from mnist import MNIST
import numpy as np
from scipy.linalg import pinvh
import matplotlib.pyplot as plt
arr = np.array
eye = np.eye
pinv = np.linalg.pinv
argmax = np.argmax
randn = np.random.normal
rand = np.random.rand
sqrt = np.sqrt
pi = np.pi
cos = np.cos
arange = np.arange
mean = np.mean
argmin = np.argmin

plot = plt.plot
show = plt.show
saveas = plt.savefig
legend = plt.legend
title = plt.title
ylabel = plt.ylabel
xlabel = plt.xlabel


def load_dataset():
    mndata = MNIST("./data/")
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255
    X_test = X_test/255
    return X_train, X_test, labels_train, labels_test


def train(X, Y, reg_lambda):
    """
    d: The number of features for each of the samples.

    Args:
        X: Should be a "n by d" matrix, with all the samples pack vertically as rows into the matrix.
        Y: Should be a "n by 10" matrix, comtaninig all the labels for the digits pack vertically as
        rows for the matrix.
        reg_lambda:
            This is the regularization constant for the system.
    Returns:
        The trained linear model for the system, should be a 784 by 10 matrix such that its
        transpose multiply by the
        feature vector will produce the label vector.

    """
    Y = Y.astype(np.float64)
    return pinvh(X.T@X + reg_lambda*eye(X.shape[1]))@X.T@Y


def predict(W, Xnew):
    """

    Args:
        W: Should be a d by 10 matrix, which is the linear model.
        Xnew: Should be a n by 784 matrix that contains all the samples we want to
        to predict with using this given model.
    Returns:
        A single vector containing all the digits predicted using this model.
    """
    return argmax(W.T@Xnew.T, axis=0)


def KFoldGenerate(k, X, Y):
    """
        Generate k folds of data for K fold validations. same as sk.learn.model_selection.

    Args:
        k: The numer of folds.
        X: The row data matrix with training data.
        Y: The label of the training data set.
    Yields:
        Each of the train and test set separate by this program.
    """
    assert X.ndim == 2 and Y.ndim == 1 and X.shape[0] == Y.size,\
        "The row data matrix has to be 2d with a lable vecor that has compatible size"
    # Shuffle the data.

    N = X.shape[0]
    n = N/k  # Partition size, could be a float.
    for II in range(k):
        ValidatePartitionStart = int(II*n)
        ValidatePartitionEnd = int((II + 1)*n)
        Xvalidate = X[ValidatePartitionStart: ValidatePartitionEnd, :]
        Yvalidate = Y[ValidatePartitionStart: ValidatePartitionEnd]
        TrainIndices = [Row for Row in range(N) if (Row < ValidatePartitionStart or Row >= ValidatePartitionEnd)]
        Xtrain = X[TrainIndices, :]
        Ytrain = Y[TrainIndices]
        yield Xtrain, Xvalidate, Ytrain, Yvalidate


class FeaturesRemap:

    def __init__(this, p):
        this.G = randn(0, sqrt(0.1), (p, 784))
        this.b = rand(p, 1)*2*pi

    def __call__(this, X):
        """
        This is the functional call implementation. It trans form the row data matrix to the new
        cosine feature space.
        Returns:
            The transformed feature using a given data set.
        """
        return (cos(this.G@X.T + this.b)).T

def TestKfoldGenearate():
    X = randn(0, 1, (17, 3))
    Y = rand(17)
    for X1, X2, Y1, Y2 in KFoldGenerate(5, X, Y): print(X1, X2, Y1, Y2)


def main():
    X1, X2, Y1, Y2 = load_dataset()
    # Reduce size of the training set for speed, or else it takes too long to run.
    # training size is 10 000, so then cross set is going to be
    X1 = X1[::6, :]
    Y1 = Y1[::6]
    print(X1.shape) # (60000, 784)
    print(X2.shape) # (10000, 784)
    print(Y1.shape) # (60000, )
    print(Y2.shape) # (10000, )
    print(X1.dtype)
    print(X2.dtype)
    print(Y1.dtype)
    print(Y2.dtype)
    print("Ok we are ready to train the model and make prediction now. ")
    TestKfoldGenearate()
    print("Test finished... Time to train")

    ## USE THIS!
    def TrainTheModel(X1, Y1):
        # transform the Y labels from vector into Y matrix.
        Y = (np.array([[II] for II in range(10)]) == Y1).astype(np.float)
        return train(X1, Y.T, reg_lambda=1e-4)

    def ErrorRate(y1, y2):
        return sum(y1 != y2)/y1.size

    Pdegreees = arange(300, 3000, 100)
    KfoldTrainErrorRate = []
    KfoldValidateErrorRate = []
    for p in Pdegreees:
        TrainErrorsRate, ValErrorsRate= [], []
        Mapper = FeaturesRemap(p)
        print(f"pvalue is: {p}")
        for Xtrain, Xvalidate, Ytrain, Yvalidate in KFoldGenerate(5, X1, Y1):
            Xtrain = Mapper(Xtrain); print(f"Xtrain Map: {Xtrain.shape}")
            Xvalidate = Mapper(Xvalidate); print(f"Xval Map: {Xvalidate.shape}")
            Model = TrainTheModel(Xtrain, Ytrain); print("Model Train")
            PredictTrain = predict(Model, Xtrain); print("Predict Train Labels")
            Predictval = predict(Model, Xvalidate); print("Predict val labels")
            TrainErrorsRate.append(ErrorRate(PredictTrain, Ytrain))
            ValErrorsRate.append(ErrorRate(Predictval, Yvalidate))
            print("one of the k-fold ends.")
        print(f"List of train error score: {TrainErrorsRate}")
        print(f"List of Val Error score: {ValErrorsRate}")
        KfoldTrainErrorRate.append(mean(TrainErrorsRate))
        KfoldValidateErrorRate.append(mean(ValErrorsRate))
    plot(Pdegreees, KfoldValidateErrorRate)
    plot(Pdegreees, KfoldTrainErrorRate)
    title("B2 Error rate")
    legend(["Kfold Validation Error rate", "kfold training error rate"])
    ylabel("Percentage of wrong label"); xlabel("P; row count of G matrix")
    MinPindex = argmin(KfoldValidateErrorRate)
    MinP = Pdegreees[MinPindex]
    MinValError = KfoldValidateErrorRate[MinPindex]
    print(f"The best P is: {MinP}")
    print(f"The corresponding validation error is: {MinValError}")
    plot(MinP, MinValError, "bo")
    saveas("B2plot.png", format="png")




if __name__ == "__main__":
    import os
    print(f"cwd: {os.getcwd()}")
    print(f"script running at {os.curdir}")
    main()
