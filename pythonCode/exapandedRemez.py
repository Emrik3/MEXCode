import time
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np
import torch
from evalPol import eval3, eval5, eval9, eval17, sastre8
from sympy.series.approximants import approximants


def p(c, x):
    out = 0
    for i in range(len(c)):
        out += c[i] * x ** (2 * i + 1)
    return out


# Derivative of polynomial
def pp(c, x):
    out = 0
    for i in range(len(c)):
        out += c[i] * (2 * i + 1) * x ** (2 * i)
    return out


# Second derivative of polynomial
def ppp(c, x):
    out = 0
    for i in range(1, len(c)):
        out += c[i] * (2 * i + 1) * (2 * i) * x ** (2 * i - 1)
    return out


def odd_remez(q, l, u, tol):
    x = np.zeros(q + 2)
    f = np.ones(q + 2)
    n = q + 2

    # Calculate initial guess of points as Chebyshev points
    for i in range(n):
        x[i] = 0.5 * (l + u) + 0.5 * (u - l) * np.cos((2 * i + 1) * np.pi / (2 * n))
    x = np.array(sorted(x))
    old_E = np.inf
    E = 1000

    while np.abs(old_E - E) > tol:
        old_E = E
        A = np.zeros((n, n))
        for j in range(n):
            for i in range(n - 1):
                A[j, i] = x[j] ** (2 * i + 1)
        A[:, -1] = (-1) ** np.arange(n)
        c = np.linalg.solve(A, f)

        x_new = []
        coeffs_for_roots = derivative_coeffs(c[:-1])
        root_guess = np.roots(coeffs_for_roots)
        candidates = []
        for r in root_guess:
            if np.isreal(r):
                r = r.real
                if r > 0:
                    candidates.append(r)

        for guess in candidates:  # If they are too close we might have problems
            x_new.append(newton_pol(guess, c[:-1], tol))

        # Always include endpoints
        x_new = [l] + x_new + [u]

        # Sort for consistency
        x_new = np.array(sorted(x_new))

        if len(x_new) != n:
            raise ValueError(f"Expected {n} extremal points, got {len(x_new)}")

        x = x_new

        # Make sure all unique points were found
        if len(x_new) == len(set(x_new)):
            x = np.array(x_new)
        else:
            print("Error: Could not find all points")
            break

        E = c[-1]

    return c


def odd_remez_expanded(q, qOld, cOld, l, u, tol):
    x = np.zeros(q + 2 + qOld)
    f = np.ones(q + 2 + qOld)
    n = q + 2 + qOld

    # Calculate initial guess of points as Chebyshev points
    for i in range(n):
        x[i] = 0.5 * (l + u) + 0.5 * (u - l) * np.cos((2 * i + 1) * np.pi / (2 * n))
    x = np.array(sorted(x))
    print(f"init points: {x}")
    old_E = np.inf
    E = 1000

    while np.abs(old_E - E) > tol:
        old_E = E
        A = np.zeros((n, n))
        for j in range(n):
            for i in range(n - 1):
                if i <= 2:
                    A[j, i] = x[j] ** (2 * i + 1)
                else:
                    A[j, i] = p(cOld, x[j]) ** (2 * (i - 2) + 1)
        A[:, -1] = (-1) ** np.arange(n)
        c = np.linalg.solve(A, f)

        x_new = []
        root_guess = find_roots(c[:-1], cOld, l, u)

        # Always include endpoints
        x_new = [l] + list(root_guess) + [u]

        # TODO: Add newton to refine this here

        # Sort for consistency
        x_new = np.array(sorted(x_new))

        if len(x_new) != n:
            raise ValueError(f"Expected {n} extremal points, got {len(x_new)}")

        x = x_new
        # Make sure all unique points were found
        if len(x_new) == len(set(x_new)):
            x = np.array(x_new)
        else:
            print("Error: Could not find all points")
            break

        E = c[-1]

    return c


def find_roots(c, cOld, l, u):
    N = 1000
    xs = np.linspace(l, u, N)
    vals = [
        (
            pp(c[0:3], x)
            + c[3] * 3 * pp(cOld, x) * p(cOld, x) ** 2
            + c[4] * 5 * pp(cOld, x) * p(cOld, x) ** 4
        )
        for x in xs
    ]

    def f(y):
        return (
            p(c[0:3], np.array(y))
            + c[3] * p(cOld, np.array(y)) ** 3
            + c[4] * p(cOld, np.array(y) ** 5)
        )

    guesses = []

    for i in range(len(xs) - 1):
        if vals[i] * vals[i + 1] < 0:
            a, b = xs[i], xs[i + 1]
            guesses.append((a + b) / 2)

    guesses = np.array(guesses)

    while len(guesses) > len(c) - 1:
        guesses = np.delete(guesses, np.where(f(guesses) == min(f(guesses))))

    return guesses


def derivative_coeffs(c):
    """
    Construct coefficients of p'(x) for an odd polynomial
    p(x) = sum c[i] x^(2i+1)

    Returns coefficients in descending powers for np.roots
    """
    q = len(c) - 1
    coeffs = np.zeros(2 * q + 1)

    for i, ci in enumerate(c):
        power = 2 * i
        coeffs[2 * q - power] = ci * (2 * i + 1)

    return coeffs


def newton_pol(x, c, tol):
    err = 100.0
    while err > tol:
        x_new = x - pp(c, x) / ppp(c, x)

        err = np.abs(x_new - x)
        x = x_new

    return x


def plot_pol(c):
    x = np.linspace(0, 1, 100)
    plt.plot(x, p(c, x))
    plt.plot(x, 1 + 0 * x)
    plt.show()


# Does this only best for degree 5, the cushioning and such?
# Maybe just do any degree not using this?
def get_all_coeffs_different_degrees(q_list, T, l=0.001):
    cushion = 0.02407327424182761
    cushion = 0
    u = 1
    all_coeffs = []
    eps = 1e-10

    for i in range(T):
        q = q_list[i]
        print(i)

        c = odd_remez(q, max(l, cushion * u), u, 1e-8)  # Make  more exact?

        if cushion * u > l:
            pl = p(c[:-1], l)
            pu = p(c[:-1], u)
            rescalar = 2 / (pl + pu)
            for i in range(len(c[:-1])):
                c[i] *= rescalar

        l = p(c[:-1], l)
        u = 2 - l
        all_coeffs.append(c[:-1])
    return all_coeffs


def get_all_coeffs_different_degrees_new(q_list, T, l=0.001):
    cushion = 0.02407327424182761
    cushion = 0
    u = 1
    all_coeffs = []
    eps = 1e-10

    for i in range(T):
        q = q_list[i]
        print(i)
        if i == 0:
            c = odd_remez(q, max(l, cushion * u), u, 1e-8)  # Make  more exact?
        else:
            c = odd_remez_expanded(q, q_list[i - 1], all_coeffs[-1], l, u, 1e-8)
        if cushion * u > l:
            pl = p(c[:-1], l)
            pu = p(c[:-1], u)
            rescalar = 2 / (pl + pu)
            for i in range(len(c[:-1])):
                c[i] *= rescalar

        all_coeffs.append(c[:-1])
    return all_coeffs


def test_approximation_new(q, l=0.001):
    T = len(q)
    coeffs17 = get_all_coeffs_different_degrees_new(q, T, l)
    x_plt = np.linspace(l, 1, 10000)

    x = np.linspace(l, 1, 10000)
    tot_degree = 1
    for i in range(T):
        if i == 0:
            x = p(coeffs17[i], x)
        else:
            x = (
                p(coeffs17[i][0:3], x_plt)
                + coeffs17[i][3] * x**3
                + coeffs17[i][4] * x**5
            )

        print(x)
        tot_degree *= 2 * q[i] + 1
    print(f" min: {min(x)},  max: {max(x)}")

    plt.plot(
        x_plt,
        x,
        label=f"Total degree = {tot_degree}, d = {2 * np.array(q) + 1}",
    )

    plt.legend()


def test_approximation(q, l=0.001):
    T = len(q)
    coeffs17 = get_all_coeffs_different_degrees(q, T, l)

    x_plt = np.linspace(l, 1, 10000)

    x = np.linspace(l, 1, 10000)
    tot_degree = 1
    for i in range(T):
        x = p(coeffs17[i], x)
        tot_degree *= 2 * q[i] + 1
    print(f" min: {min(x)},  max: {max(x)}")

    plt.plot(x_plt, x, label=f"Total degree = {tot_degree}, d = {2 * np.array(q) + 1}")

    plt.legend()


def approxs():
    test_approximation([2, 2], 0.001)
    test_approximation_new([2, 2], 0.001)
    plt.show()


def main():
    # TODO: Some issues with the convergence of newton when the tol is too high, for large polynomials it does not converge.
    # TODO: Have some issues when using degree 3, since only one point it is not working correctly, change to just have exact solution when it is of degree 3.
    approxs()


if __name__ == "__main__":
    main()
