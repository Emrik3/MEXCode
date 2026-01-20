from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np
import torch


def odd_remezGen(q, func, l, u, tolNewton, alpha=1.0):
    x = np.zeros(q + 1)
    f = np.zeros(q + 1)
    n = q + 1

    # Calculate initial guess of points as Chebyshev points
    for i in range(n):
        x[i] = 0.5 * (l + u) + 0.5 * (u - l) * np.cos((2 * i) * np.pi / (2 * n))
    err = 1000.0
    c = None
    old_E = np.inf
    E = 1000

    while np.abs(old_E - E) > 1e-15:
        old_E = E
        A = np.zeros((q + 1, q + 1))
        for j in range(q + 1):
            for i in range(q):
                A[j, i] = x[j] ** (2 * i)
        A[:, -1] = (-1) ** np.arange(q + 1)

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
            x_new.append(newton_pol(guess, c[:-1], tolNewton))

        # Always include endpoints
        x_new = [l] + x_new + [u]

        # Sort for consistency
        x_new = np.array(sorted(x_new))

        if len(x_new) != q + 1:
            raise ValueError(f"Expected {q + 1} extremal points, got {len(x_new)}")

        x = x_new

        # Make sure all unique points were found
        if len(x_new) == len(set(x_new)):
            x = np.array(x_new)
        else:
            print("Error: Counld not find all points")
            break

        for i in range(len(x)):
            f[i] = func(x[i])

        E = c[-1]
    return c


def get_all_coeffsGen(q, T):
    l = 0.001
    cushion = 0.02407327424182761
    u = 1
    all_coeffs = []

    f = lambda x: x ** (-1 / 2)

    for i in range(T):
        print(i)
        c = odd_remezGen(q, f, max(l, cushion * u), u, 1e-10)  # Make  more exact
        pl = p(c[:-1], l)
        pu = p(c[:-1], u)
        rescalar = 2 / (pl + pu)
        for i in range(len(c[:-1])):
            c[i] *= rescalar

        l = p(c[:-1], l)
        u = 2 - l
        all_coeffs.append(c[:-1])
    return all_coeffs


def p(c, x):
    out = 0
    for i in range(len(c)):
        out += c[i] * x ** (2 * i + 1)
    return out


def derivative_coeffs(c):
    """
    Construct coefficients of p'(x) for an odd polynomial
    p(x) = sum c[i] x^(2i+1)

    Returns coefficients in descending powers for np.roots
    """
    q = len(c) - 1
    coeffs = np.zeros(2 * q)

    for i, ci in enumerate(c):
        power = 2 * i
        coeffs[2 * q - 1 - power] = ci * (2 * i)

    return coeffs


def test_general_approximation():
    T = 5
    q = 2

    coeffs17 = get_all_coeffsGen(q, T)

    x_plt = np.linspace(0, 1, 1000)

    x = np.linspace(0, 1, 1000)
    for i in range(T):
        x = p(coeffs17[i], x)
    plt.plot(x_plt, x * x_plt, label="New")
    plt.legend()
    plt.show()


def main():
    test_general_approximation()


if __name__ == "__main__":
    main()
