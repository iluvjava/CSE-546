# Code is for B1(d), HW1, CSE 546.
import numpy as np
sin = np.sin
cos = np.cos
array = np.array
PI = np.pi
rnd = np.random.normal
arange = np.arange

import math
ceil = math.ceil
zeros = np.zeros
mean = np.mean

import matplotlib.pyplot as plt
scatter = plt.scatter
plot = plt.plot
show = plt.show


def f(x):
    return 4*sin(x*PI)*cos(6*PI*x)


def Epsilon(length):
    return rnd(0, 1, length)


def GenerateData(n=256):
    Xgrid = arange(0, 1, 1/256)
    return Xgrid, f(Xgrid) + Epsilon(n)


def fHat(data, m, n=256):
    y = zeros(n)
    for j in range(1, int(n/m) + 1):
        UpperBound = min(j*m, n)
        y[(j - 1)*m: UpperBound] = mean(data[(y - 1)*m: UpperBound])
    return y


def fBar(xgrid, m, n=256):
    return fHat(f(xgrid), m, n)


def AvgBiasSqaured(data):
    pass


def AvgVariance(data):
    pass


def AvgEmpiricalErr(data):
    pass


def main():
    Xgrid, Data = GenerateData()
    for m in [2**II for II in range(6)]:
        ...
    pass


if __name__ == "__main__":
    import os
    print(f"wd: {os.getcwd()}")
    print(f"script running on{os.curdir}")
    main()