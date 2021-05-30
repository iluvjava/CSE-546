### CLASS CSE 564 SPRING 2021 HW4 A4
### Name: Hongda Li
### My code has my style in it don't copy.


import numpy as np
zeros = np.zeros
randint = np.random.randint


class KMean:

    def __init__(this, k:int, X:np.ndarray):
        """

        :param k: Number of cluster
        :param X: Row data matrix in np array type
        """
        assert k < X.shape[0] and k > 1
        assert X.ndim == 2
        n, d= X.shape[0], X.shape[2]
        this.X = X
        this.AugX = X[:, :, np.newaxis]
        this.Assignment = {}
        this.C = X[randint(0, n, k)].reshape((1, d, k))

    def _Initialize(this):

        pass

    def _ComputeCentroid(this):

        pass

    def _ReassignCentroid(this):
        pass

    def Update(self):
        pass



