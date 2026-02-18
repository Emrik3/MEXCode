import matplotlib.pyplot as plt
import numpy as np


def p(z, c):
    return c[3] * z + c[4] * z**3 + c[5] * z**5


def g(z, c):
    return (
        c[0] * z
        + c[1] * z**3
        + c[2] * z**5
        + c[-2] * p(z, c) ** 3
        + c[-1] * p(z, c) ** 5
    )


def r(zs, c):
    """
    Added penalty term at the start to make it grow quickley, but it is hard to do gauss newton given that we want bounds on maximum error.
    """
    return np.array([(g(z, c) - 1) / (10 * z) for z in zs]).T


def J(zs, c):
    return np.array(
        [
            [
                z,
                z**3,
                z**5,
                3 * c[-2] * p(z, c) ** 2 * z + 5 * c[-1] * p(z, c) ** 4 * z,
                3 * c[-2] * p(z, c) ** 2 * z**3 + 5 * c[-1] * p(z, c) ** 4 * z**3,
                3 * c[-2] * p(z, c) ** 2 * z**5 + 5 * c[-1] * p(z, c) ** 4 * z**5,
                p(z, c) ** 3,
                p(z, c) ** 5,
            ]
            for z in zs
        ]
    )


def chebyshev(n, l, u, alpha=2.0):
    """
    Includes the endpoints l and u.
    """

    k = np.arange(n)
    x = np.cos((k / (n - 1)) ** alpha * np.pi)
    return (l + u) / 2 + (u - l) / 2 * x


def gn(tol, max_iter, zs, c=np.array([0, 0, 0, 8, -25, 17, -3, 0.5]).T):
    """
    Very sensitive to starting guess, should have starting guess be the value that is from remez maybe.
    """
    err = 1
    i = 0
    while err > tol and i < max_iter:
        print(err)
        c_old = c
        try:
            c = c - np.linalg.inv(J(zs, c).T @ J(zs, c)) @ J(zs, c).T @ r(zs, c)
        except np.linalg.LinAlgError:
            break
        err = np.linalg.norm(c_old - c) / np.linalg.norm(c)
        i += 1
    return c


def main():
    l = 0.001
    zs = chebyshev(100, l, 1, alpha=2)
    c = gn(5e-5, 2000, zs)
    x = np.linspace(0, 1, 1000)
    print(f"g(l) = {g(l, c)}")
    plt.plot(x, g(x, c))
    plt.show()


if __name__ == "__main__":
    main()
