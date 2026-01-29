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

    return c, x


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


# Does this only best for degree 5, the cushioning and such?
# Maybe just do any degree not using this?
def get_all_coeffs_different_degrees(q_list, T, l=0.001):
    cushion = 0.02407327424182761
    u = 1
    all_coeffs = []
    equiList = []
    eps = 1e-10

    for i in range(T):
        q = q_list[i]
        print(i)
        c, equi = odd_remez(q, max(l, cushion * u), u, 1e-8)  # Make  more exact?
        pl = p(c[:-1], l)
        pu = p(c[:-1], u)
        rescalar = 2 / (pl + pu)
        print(f"Rescalar: {rescalar}")
        for i in range(len(c[:-1])):
            c[i] *= rescalar

        l = p(c[:-1], l)
        u = 2 - l
        all_coeffs.append(c[:-1])
        equiList.append(equi)
    return all_coeffs, equiList


@torch.compile
def PolarExpress(G: torch.Tensor, steps: int, coeffs_list) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()  # for speed
    if G.size(-2) > G.size(-1):
        X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    for a, b, c in coeffs_list:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX ˆ3 + cX ˆ5
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile
def NewPolarExpress(G: torch.Tensor, steps: int, coeffs_list) -> torch.Tensor:
    # TODO: accumulate in 32 bult mult in 16
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    for c in coeffs_list:
        if len(c) == 2:
            X = eval3(X, c)
        elif len(c) == 3:
            X = eval5(X, c)
        elif len(c) == 5:
            X = eval9(X, c)
        elif len(c) == 9:
            X = eval17(X, c)
        else:
            raise NotImplementedError("This mult not impl!")

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


def test_approximation(q, l=0.001):
    T = len(q)
    coeffs17, equiList = get_all_coeffs_different_degrees(q, T, l)

    x_plt = np.linspace(l, 1, 10000)

    x = np.linspace(l, 1, 10000)
    tot_degree = 1
    for i in range(T):
        x = p(coeffs17[i], x)
        tot_degree *= 2 * q[i] + 1

    print(f" min: {min(x)},  max: {max(x)}")

    # plt.plot(x_plt, x, label=f"Total degree = {tot_degree}, d = {2 * np.array(q) + 1}")

    plt.legend()


def interpolate(x, e, eps):
    A = np.array(
        [[x[i] ** (2 * j + 1) for j in range(len(x) - 1)] for i in range(len(x))]
    )

    f = np.zeros(len(x))
    f[0] = min(e)
    for i in range(1, len(x) - 1, 2):
        f[i] = max(e) + eps
        f[i + 1] = min(e) - eps
    f[-1] = max(e)
    c, residuals, rank, s = np.linalg.lstsq(A, f, rcond=None)
    return c


def plot_all(q, l=0.001):
    T = len(q)
    coeffs17, equiList = get_all_coeffs_different_degrees(q, T, l)
    x_plt = np.linspace(l, 1, 10000)
    u = 1
    x = np.linspace(l, 1, 10000)
    tot_degree = 1
    # Also maybe eps have ot be different at dfferent iters
    eps = 0.01  # This one should not be needed, should jsut make sure interpolation is accurate in the way that it makes sure the polynomial goes to these points, maybe change the rescalar thing as well

    for i in range(len(coeffs17) - 1):
        plt.plot(
            np.linspace(l, u, 10000),
            p(coeffs17[i], np.linspace(l, u, 10000)),
            label=f"old P{i + 1}",
        )

        coeffs17[i] = interpolate(
            equiList[i], [p(coeffs17[i], l), 2 - p(coeffs17[i], l)], eps
        )
        plt.plot(
            np.linspace(l, u, 10000),
            p(coeffs17[i], np.linspace(l, u, 10000)),
            label=f"new P{i + 1}",
        )
        l = p(coeffs17[i], l)
        u = 2 - l
        plt.plot(np.linspace(0, 2), np.linspace(0, 2) * 0 + l)
        plt.plot(np.linspace(0, 2), np.linspace(0, 2) * 0 + u)

        print(coeffs17[i])
        plt.legend()
        plt.show()

    for i in range(T):
        x = p(coeffs17[i], x)
        tot_degree *= 2 * q[i] + 1

    print(
        f"New min: {min(x)}, new max: {max(x)}"
    )  # It seems to work, i get larger l and smaller u? Double check these results


def test_polar():
    q = [2, 2, 2, 8]
    qPE = [2, 2, 2, 2, 2]
    T = len(q)
    TPE = len(qPE)
    coeffs17 = get_all_coeffs_different_degrees(q, T)
    coeffsPE = get_all_coeffs_different_degrees(qPE, TPE)
    print(coeffs17)

    for i in range(len(coeffsPE)):
        coeffsPE[i] /= 1.01 ** (2 * i + 1)

    A = torch.abs(torch.randn(5000, 70))
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    polarFactor = U @ Vh

    sastreCoeffs = []

    for i in range(len(coeffs17)):
        if len(coeffs17[i]) == 9:
            sastreCoeffs.append(sastre8(coeffs17[i]))
        else:
            sastreCoeffs.append(coeffs17[i])

    polarFactorNew = NewPolarExpress(A, T, sastreCoeffs)

    polarFactorPE = PolarExpress(A, TPE, coeffsPE)

    diffPE = polarFactor - polarFactorPE
    diffNew = polarFactor - polarFactorNew
    normFactor = polarFactor.norm(dim=(-2, -1), keepdim=True)

    errPE = diffPE.norm(dim=(-2, -1), keepdim=True) / normFactor
    errNew = diffNew.norm(dim=(-2, -1), keepdim=True) / normFactor

    print(f"Polar Express error: {errPE}")
    print(f"New Express error: {errNew}")


def approxs():
    test_approximation([2, 2, 2, 2, 2], 0.001)
    test_approximation([4, 4, 8])


def main():
    # TODO: Some issues with the convergence of newton when the tol is too high, for large polynomials it does not converge.
    # TODO: Have some issues when using degree 3, since only one point it is not working correctly, change to just have exact solution when it is of degree 3.
    approxs()
    # plot_all([2, 2, 2, 2, 2])


if __name__ == "__main__":
    main()
