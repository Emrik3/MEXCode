import matplotlib.pyplot as plt
import numpy as np


def chebApprox(x, n1, n2):
    return 2 * x * n1 - n2


def sgn(x, n):
    s = 0
    chebList = []
    chebList.append(1)
    chebList.append(x)
    for k in range(2, 2 * n + 1):
        chebList.append(chebApprox(x, chebList[-1], chebList[-2]))
    for k in range(n):
        i = 2 * k + 1
        s = s + chebList[i] / i
    return 4 / np.pi * s


def main():
    n = 50
    x = np.linspace(0.1, 1)
    s = sgn(x, n)
    plt.plot(x, s)
    plt.show()


main()
