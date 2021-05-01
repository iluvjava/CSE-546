### Name: Hongda Alto Li
### Class: CSE 546
### This is for A5 of the HW2.
### Don't copy my code this is my code it has my style in it.

from lasso import LassoRegression, np, LassoLambdaMax
import pandas as pd


def main():
    df = pd.read_table("crime-train.txt")
    print(df.head())
    


if __name__ == "__main__":
    import os
    print(f"script running at: {os.curdir}")
    print(f"cwd: {os.getcwd()}")
    print(f"script is ready to run")
    main()