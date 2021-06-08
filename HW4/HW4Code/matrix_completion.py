
import torch
import torch.nn as nn
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from math import isnan
import csv

DATA_PATH = "./data/ml-100k/u.data"



def MoiveAvgRating(sparseDataMatrix:sparse.coo_matrix):
    DenseMatrix = sparseDataMatrix.toarray().astype(np.float)
    # number of rated movie
    RatedMovies = np.sum(np.array(DenseMatrix != 0, dtype=np.float),
                         axis=1, keepdims=True)
    TotalRatings = np.sum(DenseMatrix, axis=1, keepdims=True)
    AverageRatings = TotalRatings /RatedMovies  # average rating for each movie
    GlobalAvg = np.sum(TotalRatings.reshape(-1))/np.sum(RatedMovies.reshape(-1))
    # take care of movie nobody viewed.
    return np.nan_to_num(AverageRatings, nan=GlobalAvg)


if "DATA" not in dir():
    DATA = []
    with open(DATA_PATH) as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        for row in spamreader:
            DATA.append([int(row[0]) - 1, int(row[1]) - 1, int(row[2])])
    DATA = np.array(DATA)
    NUM_OBSERVATIONS = len(DATA)           # num_observations = 100,000
    NUM_USERS = max(DATA[:, 0]) + 1        # num_users = 943, indexed 0,...,942
    NUM_ITEMS = max(DATA[:, 1]) + 1        # num_items = 1682 indexed 0,...,1681
    np.random.seed(1)
    NUM_TRAIN = int(0.8*NUM_OBSERVATIONS)
    perm = np.random.permutation(DATA.shape[0])
    TRAIN = DATA[perm[0:NUM_TRAIN], :]
    TEST = DATA[perm[NUM_TRAIN::], :]
    del perm
    TRAIN_SPR = sparse.coo_matrix(
        (TRAIN[:, 2], (TRAIN[:,1], TRAIN[:, 0])), (NUM_ITEMS, NUM_USERS)
    )
    TEST_SPR = sparse.coo_matrix(
        (TEST[:, 2], (TEST[:,1], TEST[:, 0])), (NUM_ITEMS, NUM_USERS)
    )
    TRAIN_AVG = MoiveAvgRating(TRAIN_SPR)
    print("DATA HAS BEEN LOADED. ")


# ========================== List of Helper Functions ==========================

def Ts():
    from datetime import datetime
    SysTime = datetime.now()
    TimeStamp = SysTime.strftime("%H-%M-%S")
    return TimeStamp


def mkdir(dir):
    from pathlib import Path
    Path(dir).mkdir(parents=True, exist_ok=True)


def log(fname:str, content:str, dir):
    mkdir(dir)
    TimeStamp = Ts()
    with open(f"{dir}{TimeStamp}-{fname}.txt","w+") as f:
        f.write(content)

# ==============================================================================

def Epsilon(approx, train=True):
    Sparse = TRAIN_SPR if train else TEST_SPR
    DiffSum = 0
    for Idx, (II, JJ, Rating) in enumerate(
            zip(Sparse.row, Sparse.col, Sparse.data)
    ):
        DiffSum += (Rating - approx[II, JJ]) ** 2
    return DiffSum / Idx

def PartA():
    DiffSum = 0
    for Idx, (II, _, Rating) in enumerate(
            zip(TEST_SPR.row, TEST_SPR.col, TEST_SPR.data)
        ):
        DiffSum += (Rating - TRAIN_AVG[II, 0])**2
    return DiffSum/Idx

def PartB(ranks=[1, 2, 5, 10, 20, 50]):
    Ranks = sorted(ranks + [0])
    RTilde = TRAIN_SPR.asfptype()  # Filled with zeros.
    U, Sigma, VTransposed = svds(RTilde, k=942)
    U, Sigma, VTransposed = U[:, ::-1], Sigma[::-1], VTransposed[::-1]
    Approximation = np.zeros(RTilde.shape)
    MSETrain, MSETest = [], []

    for RankStart, RankEnd in zip(Ranks[: -1], Ranks[1:]):
        Approximation += U[:, RankStart: RankEnd]\
            @\
            np.diag(Sigma[RankStart: RankEnd])\
            @\
            VTransposed[RankStart: RankEnd]
        MSETrain.append(Epsilon(Approximation, True))
        MSETest.append(Epsilon(Approximation, False))
    return ranks, MSETrain, MSETest


def main():
    FolderPath = "./B1"
    mkdir(FolderPath)
    def ParA():
        PartAError = PartA()
        with open(f"{FolderPath}/", "w+") as f:
            f.write(f"For part (a), the error on test set is: {PartAError}")
        print(f"Fora part (b):")
    ParA()
    def ParB():
        Ranks, TrainErr, TestErr = PartB()
        print(f"Train Errors {TrainErr, TestErr}")
        plt.plot(Ranks, TrainErr, "-o")
        plt.plot(Ranks, TestErr, "-o")
        plt.legend(["Train MSE", "Test MSE"])
        plt.title("Ranks and Reconstruction (Filled with Zeroes)")
        plt.savefig(f"{FolderPath}/{Ts()}-b1-b.png")
        plt.show()
    ParB()


if __name__ == "__main__":
    import os
    print(os.getcwd())
    print(os.curdir)
    main()


