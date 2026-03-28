import time
from itertools import repeat
from types import LambdaType

import matplotlib.pyplot as plt
import numpy as np
import torch
from evalPol import eval3, eval5, eval9, eval17, sastre8
from sympy.series.approximants import approximants


def f(c, x):
    for cc in c:
        out = 0
        for i in range(len(cc)):
            out += cc[i] * x ** (2 * i + 1)
        x = out
    return x


# TODO: Something wrong in this polynomial eval? Why does the final polynomial only look like low degree and not very high degree?
def p(allc, m, x):
    out = np.zeros(m + 2)
    out[0] = 1
    out[1] = x**2
    A = []
    B = []

    idx = 0

    # Build A
    for i in range(m):
        k = i + 2
        A.append(allc[idx : idx + k])
        idx += k

    # Build B
    for i in range(m):
        k = i + 2
        B.append(allc[idx : idx + k])
        idx += k

    # Build c
    c = allc[idx : idx + m]

    # Evaluate
    for i in range(m):
        out1 = 0
        out2 = 0

        for j in range(len(A[i])):
            out1 += A[i][j] * out[j]
            out2 += B[i][j] * out[j]

        out[i + 2] += c[i] * out1 * out2

    return x * np.sum(out)


# Derivative of polynomial
def pp(allc, m, x, eps):
    """Numerical"""
    return (p(allc, m, x + eps) - p(allc, m, x - eps)) / (2 * eps)


# Second derivative
def ppp(allc, m, x, eps):
    """Numerical"""
    return (p(allc, m, x + eps) - 2 * p(allc, m, x) + p(allc, m, x - eps)) / eps**2


def g(zs, allc, m):
    return p(allc, m, zs)


def r(zs, allc, m, fc):
    """
    Added penalty term at the start to make it grow quickley, but it is hard to do gauss newton given that we want bounds on maximum error.
    """
    return np.array([(g(z, allc, m) - f(fc, z)) for z in zs])


def J(zs, allc, m):
    h = 1e-10
    # TODO: Need to flatten the coeffs for this
    return np.array(
        [
            [
                (
                    g(z, allc + h * np.eye(len(allc))[i], m)
                    - g(z, allc - h * np.eye(len(allc))[i], m)
                )
                / (2 * h)
                for i in range(len(allc))
            ]
            for z in zs
        ]
    )


def gn(fc, tol, max_iter, zs, c0, m, alpha0=1.0, rho=0.5, a=1e-4):
    """
    Very sensitive to starting guess, should have starting guess be the value that is from remez maybe.
    """
    c = c0.copy()
    err = 1 - 0.001
    for i in range(max_iter):
        res = r(zs, c, m, fc)
        jac = J(zs, c, m)
        F = 0.5 * np.dot(res, res)
        gr = jac.T @ res

        p, *_ = np.linalg.lstsq(jac, -res, rcond=None)
        if gr @ p >= 0:
            raise RuntimeError("Direction is not descent.")

        # Armijo
        alpha = alpha0
        while True:
            c_trial = c + alpha * p
            res_trial = r(zs, c_trial, m, fc)
            F_trial = 0.5 * np.dot(res_trial, res_trial)

            if F_trial <= F + a * alpha * (gr @ p):
                break
            alpha *= rho
        c = c_trial
        err = np.max([np.abs(g(z, c, m) - 1) for z in zs])
        err2 = np.sum([np.abs(g(z, c, m) - 1) ** 2 for z in zs])

        print(err)
        if err2 < tol:
            break

    return c


def roots(c, l, u, steps=10000):
    xs = np.linspace(l, u, steps)
    guesses = []
    eps = 1e-8

    prev_x = xs[0]
    prev_val = fp(c, prev_x, eps)

    for x in xs[1:]:
        curr_val = fp(c, x, eps)

        # Check for sign change
        if prev_val * curr_val < 0:
            guesses.append((prev_x + x) / 2)
        prev_x = x
        prev_val = curr_val

    return guesses


# Derivative of polynomial
def fp(allc, x, eps):
    """Numerical"""
    return (f(allc, x + eps) - f(allc, x - eps)) / (2 * eps)


# Second derivative
def fpp(allc, x, eps):
    """Numerical"""
    return (f(allc, x + eps) - 2 * f(allc, x) + f(allc, x - eps)) / eps**2


def newton_pol(x, c, tol):
    err = 100.0
    eps = 1e-8
    while err > tol:
        x_new = x - fp(c, x, eps) / fpp(c, x, eps)
        err = np.abs(x_new - x)
        x = x_new
    return x


def plot_pol(c, m):
    x = np.linspace(0, 1, 1000)
    vals = [p(c, m, xx) for xx in x]
    plt.plot(x, vals, label="New")
    plt.plot(x, 1 + 0 * x)


def plot_og(c):
    x = np.linspace(0, 1, 1000)
    vals = [f(c, xx) for xx in x]
    plt.plot(x, vals, label="PE")


def approxs():
    m = 11  # TODO: Why would too high degree make it not work?
    l = 0.001
    u = 1

    fc = [
        [8.4703288, -25.10807471, 18.6292756],
        [4.18283418, -3.10870111, 0.58060668],
        [3.96185728, -2.95406375, 0.56297612],
        [3.28658622, -2.46472013, 0.50735769],
        [2.27374999, -1.64466037, 0.41619093],
    ]
    guesses = roots(fc, l, u)

    guesses = [l] + list(guesses) + [u]
    guess = np.array([(-0.99) ** i for i in range(m**2 + 4 * m + 2)])

    c = gn(fc, 1e-8, 50, guesses, guess, m)
    print(c)
    plot_pol(c, m)
    plot_og(fc)
    plt.show()


def main():
    # TODO: Some issues with the convergence of newton when the tol is too high, for large polynomials it does not converge.
    # TODO: Have some issues when using degree 3, since only one point it is not working correctly, change to just have exact solution when it is of degree 3.
    approxs()
    # test_polar()


if __name__ == "__main__":
    main()
