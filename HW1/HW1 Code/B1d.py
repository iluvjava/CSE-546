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
xticks = plt.xticks
title = plt.title
legend = plt.legend
save = plt.savefig
ylabel = plt.ylabel

def f(x):
    return 4*sin(x*PI)*cos(6*PI*x)


def Epsilon(length):
    return rnd(0, 1, length)


def GenerateData(n=256):
    Xgrid = array([(II + 1)/n for II in range(n)])
    return Xgrid, f(Xgrid) + Epsilon(n)


def yHat(data, m, n=256):
    """
        Given the data, fit it with average data points on each of the partition.
    Args:
        data:
        m:
        n:
    Returns:
        Predicted value
    """
    y = zeros(n)
    for JJ in range(int(n/m)):
        UpperBound = min((JJ + 1)*m, n)
        LowerBound = JJ*m
        y[LowerBound: UpperBound] = mean(data[LowerBound: UpperBound])
    return y


def fBar(xgrid, m, n=256):
    return yHat(f(xgrid), m, n)


def AvgBiasSqaured(m, n=256):
    """
        The expected vaue of the biases error squared. Because it's expected value,
        this will use the underlying generative model instead of using data to get
        the error from the biases squared.

    Args:
        data:

    Returns:

    """
    Xgrid = array([(II + 1)/n for II in range(n)])
    FBar = fBar(Xgrid, m, n)
    F = f(Xgrid)
    return mean((FBar - F)**2)


def AvgVariance(m):
    return 1/m


def AvgEmpiricalErr(yhat, n=256):
    Xgrid = array([(II + 1)/n for II in range(n)])
    return mean((f(Xgrid) - yhat)**2)


def main():
    def FitDemo():
        Xs, Ys = GenerateData()
        scatter(Xs, Ys)
        Yhat = yHat(Ys, 8)
        YBar = fBar(Xs, 8)
        plot(Xs, Yhat, c="red")
        plot(Xs, YBar, c="green")
        show()
    FitDemo()
    def PlotErrors():
        Error1 = [] # Empirical error from 256 random samples.
        Error2 = [] # Expected Bias Squared
        Error3 = [] # Expcted Variance Square
        Error4 = [] # Expected Errors
        _, SampledData = GenerateData()
        for m in [2**II for II in range(6)]:
            Error1.append(AvgEmpiricalErr(yHat(SampledData, m)))
            Error2.append(AvgBiasSqaured(m))
            Error3.append(AvgVariance(m))
            Error4.append(Error2[-1] + Error3[-1])
        plot(Error1)
        plot(Error2)
        plot(Error3)
        plot(Error4)
        xticks(range(6), [f"m=$2^{II}$" for II in range(6)])
        ylabel("Error")
        legend(["Empirical Error", "E[Bias^2]", "E[Var]", "E[Bias^2 + Var]"])
        title("B.1(d): Variance Decomposition")
        save("B.1(d)plot.png")

    PlotErrors()


if __name__ == "__main__":
    import os
    print(f"wd: {os.getcwd()}")
    print(f"script running on{os.curdir}")
    main()