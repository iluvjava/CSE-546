### This is for CSE 546, SPRING 2021, HW2, Problem A6
### I am going to use the implementation in logistic.py to get the loss
### For minist under different scienario, and gradient descent strategies.
### !!!!!
### Requires: logistic.py
### minst_python, and the data for all the MINIST Images.

from mnist import MNIST
from logistic import BinaryLogisticRegression, np
import matplotlib.pyplot as plt

zeros = np.zeros
exp = np.exp
mean = np.mean
norm = np.linalg.norm
array = np.array
ones = np.ones
where = np.where
abs = np.abs
randint = np.random.randint

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

    def A6b():
        print(f"{'='*10} Smooth Gradient Descend {'='*10}")
        print(f"Computing Best stepsize...")
        StepSize = (1/norm(XTrain.T@XTrain) + 2*1e-1)*2;
        print(f"Step size is: {StepSize}")
        Model = BinaryLogisticRegression(stepsize=StepSize)
        LossTrain = [Model.CurrentLoss(XTrain, YTrain)]
        LossTest = [Model.CurrentLoss(XTest, YTest)]
        MisClassifyTrain = []  #[mean(abs(Model.Predict(XTrain) - YTrain))]
        MisClassifyTest = [] #[mean(abs(Model.Predict(XTest) - YTest))]
        MisClassifyItr = []
        Itr, MaxItr = 0, 5000
        for w, b, gradw, gradb in Model.GenerateGradients(XTrain, YTrain, grad=True):
            LossTrain.append(Model.CurrentLoss(XTrain, YTrain))
            LossTest.append(Model.CurrentLoss(XTest, YTest))
            if Itr%10 == 0:  # Get the loss every 10 iteration, speeds things up.
                MisClassifyTrain.append(0.5*mean(
                    abs(
                        Model.Predict(XTrain) - YTrain)))
                MisClassifyTest.append(0.5*mean(
                    abs(
                        Model.Predict(XTest) - YTest)))
                MisClassifyItr.append(Itr)
                print(f"Miss Classify Train: {MisClassifyTrain[-1]}, "
                      f"Misclassify Test: {MisClassifyTest[-1]}")
            if len(LossTrain) > 2 and norm(gradw) < 1e-4:
                break
            if Itr > MaxItr:
                break
            Itr += 1
            print(f"itr: {Itr}, Graident W: {norm(gradw)}, Gradient b: {gradb}, |w|: {norm(w)}")
            print(f"LossTrain: {LossTrain[-1]}, LossTest: {LossTest[-1]}")
        plt.plot(LossTrain)
        plt.plot(LossTest)
        plt.xlabel("Iteration Count")
        plt.legend(["Training Loss", "Test Loss"])
        plt.title("Smooth Gradient Descend: Loss Functions")
        plt.savefig("A6-smooth-loss.png")
        plt.show()
        plt.plot(MisClassifyItr, MisClassifyTrain)
        plt.plot(MisClassifyItr, MisClassifyTest)
        plt.xlabel("Iteration Count")
        plt.legend(["Training Misclassify", "Test Misclassify"])
        plt.title("Smooth Gradient Descend: MissClassify")
        plt.savefig("A6-smooth-classification.png")
        plt.show()


    def A6c(batchSize:int):
        print(f"{'=' * 10} SGD, Batchsize: {batchSize} {'=' * 10}")

        print("Computing the best regularizer using training set: ")
        Lambda = 1e-1 / (XTrain.shape[0] / batchSize)
        print("Computing Best stepsize...")
        StepSize = (1 / norm(XTrain.T @ XTrain) + 2 * 1e-1) * 2
        print(f"Step size is: {StepSize}")
        Model = BinaryLogisticRegression(stepsize=StepSize,
                                         regularizer_lambda=Lambda)
        LossTrain = [Model.CurrentLoss(XTrain, YTrain)]
        LossTest = [Model.CurrentLoss(XTest, YTest)]
        MisClassifyTrain = []
        MisClassifyTest = []
        MisClassifyItr = []
        Itr, MaxItr = 0, 1000
        while Itr < MaxItr:
            n, _ = XTrain.shape; RandInt = randint(n - batchSize)
            X = XTrain[RandInt: RandInt + batchSize]
            y = YTrain[RandInt: RandInt + batchSize]
            Model.UpdateParametersUsing(X, y)
            LossTrain.append(Model.CurrentLoss(XTrain, YTrain))
            LossTest.append(Model.CurrentLoss(XTest, YTest))
            MisClassifyTrain.append(0.5 * mean(
                abs(
                    Model.Predict(XTrain) - YTrain)))
            MisClassifyTest.append(0.5 * mean(
                abs(
                    Model.Predict(XTest) - YTest)))
            MisClassifyItr.append(Itr)
            print(f"LossTrain: {LossTrain[-1]}, LossTest: {LossTest[-1]}")
            print(f"Miss Classify Train: {MisClassifyTrain[-1]}, "
                  f"Misclassify Test: {MisClassifyTest[-1]}")
            Itr += 1
        plt.plot(LossTest)
        plt.plot(LossTrain)
        plt.legend(["Training Loss", "Test Loss"])
        plt.xlabel("Iterations")
        plt.title(f"SGD, BatchSize:{batchSize}, Loss Graph")
        plt.savefig(f"A6-sgd-{batchSize}-loss.png")
        plt.show()
        plt.plot(MisClassifyTrain)
        plt.plot(MisClassifyTest)
        plt.title(f"SGD, BatchSize: {batchSize}, Missclassification Rate")
        plt.xlabel("Iteration")
        plt.ylabel("Percentage")
        plt.savefig(f"A6-sgd-{batchSize}-classification.png")
        plt.legend(["Missclassification Train", "Missclassification Test"])
        plt.show()
    # A6b()
    A6c(1)
    A6c(10)



if __name__ == "__main__":
    import os
    print(f"Script is runnning at: {os.curdir}")
    print(f"cwd:{os.getcwd()}")
    main()


