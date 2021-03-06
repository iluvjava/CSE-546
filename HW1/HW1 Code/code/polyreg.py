'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(this, degree=1, reg_lambda=1E-8):
        """
        Constructor
        """
        this.Degree = degree
        this.Lambda = reg_lambda


    def polyfeatures(self, X, degree):
        """
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        """
        assert len(X.shape) == 2 and X.shape[1] == 1, "Wrong input shape for X for polyfeatures."
        return np.hstack([X**II for II in range(1, degree + 1)])


    def fit(this, X, y):
        """
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        """

        X = this.polyfeatures(X, this.Degree)
        this.FeatureSTD = np.std(X, axis=0, keepdims=True)
        X = X / this.FeatureSTD
        XAug = np.zeros((X.shape[0], X.shape[1] + 1))
        XAug[:, :-1] = X
        XAug[:, -1] = np.ones(X.shape[0])
        X = XAug
        Regularizer = np.eye(X.shape[1])*this.Lambda
        Regularizer = this.Lambda*np.diag(np.ones(X.shape[1]))
        Regularizer[-1, -1] = 0
        this.ModelCoefficients = np.linalg.pinv(X.T@X + Regularizer)@X.T@y


    def predict(this, X):
        """
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        """
        # Standardize input features using training data.
        X = this.polyfeatures(X, this.Degree)
        X = X / this.FeatureSTD
        XAug = np.zeros((X.shape[0], X.shape[1] + 1))
        XAug[:, :-1] = X
        XAug[:, -1] = np.ones(X.shape[0])
        PredictedY = XAug@this.ModelCoefficients
        return PredictedY



#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------



def learningCurve(Xtrain, Ytrain, Xtest, Ytest, reg_lambda, degree):
    """
    Compute learning curve

    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree

    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]

    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """

    n = len(Xtrain)
    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    for II in range(2, n):
        TrainSetFeatures, TrainSetLabels = Xtrain[: II+ 1], Ytrain[:II + 1]
        Model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
        Model.fit(TrainSetFeatures, TrainSetLabels)

        TrainPredicted = Model.predict(TrainSetFeatures)
        errorTrain[II] = np.mean((TrainPredicted - TrainSetLabels)**2)

        TestPredicted = Model.predict(Xtest)
        errorTest[II] = np.mean((Ytest- TestPredicted)**2)

    return errorTrain, errorTest
