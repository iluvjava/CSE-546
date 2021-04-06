# Plotings for HW0 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
linspace = np.linspace
plot = plt.plot
savefig = plt.savefig
xlabel = plt.xlabel
ylabel = plt.ylabel
title = plt.title
meshgrid = np.meshgrid

def main(): 
    plotFirst()
    plotSecond()


def plotFirst():
    xs = linspace(0, 2, 1000)
    ys = xs/2 - 1
    plot(xs, ys)
    xlabel("$x_1$")
    ylabel("$x_2$")
    title("Hyper Plane 1 w = [-1, 2]")
    savefig("hyperplane1.png", format="png")


def plotSecond():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x = linspace(-1, 1, 1000)
    X, Y = meshgrid(x, x)
    Z = -X - Y
    _ = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-2, 2)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.set_title("Hyper Plane 2 w = [1 1 1]")
    savefig("hyperplane2.png", format="png")

if __name__ == "__main__":
    import os
    print(f"cwd: {os.getcwd()}")
    print(f"curdir: {os.curdir}")
    main()

