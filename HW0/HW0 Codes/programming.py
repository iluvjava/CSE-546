## This is for the programming part of HW0 for class cse 546
import numpy as np
import matplotlib.pyplot as plt
inv = np.linalg.inv
solve = np.linalg.solve
rand = np.random.normal # Standard Gaussian
sum = np.sum
zeros = np.zeros
linspace = np.linspace
sqrt = np.sqrt

plot = plt.plot
show = plt.show
title = plt.title
xlabel = plt.xlabel
ylabel = plt.ylabel
legend = plt.legend
savefig = plt.savefig

A = np.array([[0, 2, 4], [2, 4, 2], [3, 3, 1]])
b = np.array([[-2], [-2], [4]])
c = np.array([[1], [1], [1]])


def A11a():
    print("A^{-1} is: ")
    print(inv(A))

def A11b():
    Solved = solve(A, b)
    print("Solved A{-1}b is: ")
    print(Solved)
    Ac = A@c
    print("Value for Ac is:")
    print(Ac)


def CDF(randomdVar:callable, xAxis, n:int=1000):
    """
        Given a random variable (PDF), this function will make a bounch of observation and
        look for the emprical CDF output, given the x-axis of course.

    :param ranomdVar:
        A function that return the random variable, and it should not have any argument.
    :param xAxis:
        The value you want to query the empirical CDF function
    :param n:
        The number of observations to make from the random variable.
    :return:
        The output of the empirical CDF function.
    """
    randomVarVec = np.vectorize(lambda x: randomdVar())
    Observations = randomVarVec(zeros(n))
    Observations = np.sort(Observations)
    Counter = 0
    Ys = zeros(xAxis.size)
    for II, x in np.ndenumerate(xAxis):
        while Counter < Observations.size and Observations[Counter] < x:
            Counter += 1
        Ys[II] = Counter
    return Ys/n


def A12a():
    """
        Take notice that the variance of the function hat(f)_n is what we want,
        this is shown in A.6 part (a).
        The sufficient value of n is:
    :return:
    """
    n = 4000
    Xs =  linspace(-3, 3, 1000)
    Z = lambda : rand(0, 1)
    Ys = CDF(randomdVar=Z, xAxis=Xs, n=n)
    plot(Xs, Ys)
    def Y(k):
        B = lambda: 1 if rand() < 0 else -1
        Bvec = np.vectorize(lambda x:B())
        y = Bvec(zeros(k))
        return sum(y)/sqrt(k)
    for k in [1, 8, 64, 512]:
        Ys = CDF(randomdVar=lambda :Y(k), xAxis=Xs, n=n)
        plot(Xs, Ys)
    title("CDF of $Y_k$, $Z_i$")
    xlabel("observation, P(X < x)")
    ylabel("probability")
    legend(["Guassian","k = 1", "k = 8", "k = 64", "k = 512"])
    savefig("CDF.png", format="png")


def main():
    A11a()
    A11b()
    A12a()

if __name__ == "__main__":
    import os
    print(f"current cwd: {os.getcwd()}")
    print(f"Cur dir {os.curdir}")
    main()