import time
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np
import torch


def p(c, x):
    out = 0
    for i in range(len(c)):
        out += c[i] * x ** (2 * i + 1)
    return out


def ptest(c, x):
    out = 0
    m = max(c)
    for i in range(len(c)):
        c[i] /= 0.01 * m
        out += c[i] * x ** (2 * i + 1)
    return out * 0.01 * m


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


def odd_remezOne(q, l, u, tolNewton, alpha=1.0):
    x = np.zeros(q + 2)
    f = np.ones(q + 2)
    n = q + 2

    # Calculate initial guess of points as Chebyshev points
    for i in range(n):
        x[i] = 0.5 * (l + u) + 0.5 * (u - l) * np.cos((2 * i + 1) * np.pi / (2 * n))
    err = 1000.0
    c = None
    old_E = np.inf
    E = 1000

    while np.abs(old_E - E) > 1e-15:
        old_E = E
        A = np.zeros((q + 2, q + 2))
        for j in range(q + 2):
            for i in range(q + 1):
                A[j, i] = x[j] ** (2 * i + 1)
        A[:, -1] = (-1) ** np.arange(q + 2)

        c = np.linalg.solve(A, f)
        # c = [8.28721201814563, -23.595886519098837, 17.300387312530933, 1] # Optimal in PE
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

        if len(x_new) != q + 2:
            raise ValueError(f"Expected {q + 2} extremal points, got {len(x_new)}")

        x = x_new

        # Make sure all unique points were found
        if len(x_new) == len(set(x_new)):
            x = np.array(x_new)
        else:
            print("Error: Counld not find all points")
            break

        E = c[-1]
    return c


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


# Does thsi only work for degree 5, the cushioning and such?
# Maybe just do any degree not using this?
def get_all_coeffs(q, T):
    l = 0.001
    cushion = 0.02407327424182761
    u = 1
    all_coeffs = []

    for i in range(T):
        print(i)
        c = odd_remezOne(q, max(l, cushion * u), u, 1e-10)  # Make  more exact
        pl = p(c[:-1], l)
        pu = p(c[:-1], u)
        rescalar = 2 / (pl + pu)
        for i in range(len(c[:-1])):
            c[i] *= rescalar

        l = p(c[:-1], l)
        u = 2 - l
        all_coeffs.append(c[:-1])
    return all_coeffs


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
    X = G.float()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    for c in coeffs_list:
        c_torch = torch.tensor(c, dtype=torch.float32, device=X.device)
        I = torch.eye(X.size(-2), dtype=torch.float32, device=X.device)

        A = X @ X.mT
        B = A @ A
        C = B @ A
        D = C @ C

        # PS evaluation, check what happens with stability when using better eval scheme. Also maybe do the math on the cancellation?
        X = (
            c_torch[0] * I
            + c_torch[1] * A
            + c_torch[2] * B
            + (c_torch[3] * I + c_torch[4] * A + c_torch[5] * B) @ C
            + (c_torch[6] * I + c_torch[7] * A + c_torch[8] * B) @ D
        ) @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


def test_approximation():
    T = 3
    TPE = 5
    q = 8
    qPE = 2

    coeffs17 = get_all_coeffs(q, T)
    coeffsPE = get_all_coeffs(qPE, TPE)

    x_plt = np.linspace(0, 1, 1000)
    x = np.linspace(0, 1, 1000)
    for i in range(TPE):
        x = p(coeffsPE[i], x)
    plt.plot(x_plt, x, label="PolarExpress (q = 2, T = 5)")

    x = np.linspace(0, 1, 1000)
    for i in range(T):
        x = p(coeffs17[i], x)
    plt.plot(x_plt, x, label="Degree 17 (q = 8, T = 3)")

    """x = np.linspace(0, 1, 1000)
    for i in range(T2):
        x = p(coeffs2[i], x)
    plt.plot(x_plt, x, label="q = 8, T = 2")"""

    plt.legend()
    plt.show()


def time_test():
    T = 3
    TPE = 5
    q = 8
    qPE = 2
    coeffs17 = get_all_coeffs(q, T)
    coeffsPE = get_all_coeffs(qPE, TPE)
    print(coeffs17)

    for i in range(len(coeffsPE)):
        coeffsPE[i] /= 1.01 ** (2 * i + 1)

    A = torch.abs(torch.randn(50257, 768))

    start = time.perf_counter()
    polarFactorNew = NewPolarExpress(A, T, coeffs17)
    end = time.perf_counter()
    elapsed = end - start
    print(f"Time new version: {elapsed:.6f} seconds")

    start = time.perf_counter()
    polarFactorPE = PolarExpress(A, TPE, coeffsPE)
    end = time.perf_counter()
    elapsed = end - start
    print(f"Time PE: {elapsed:.6f} seconds")


def test_polar():
    T = 3
    TPE = 5
    q = 8
    qPE = 2
    coeffs17 = get_all_coeffs(q, T)
    coeffsPE = get_all_coeffs(qPE, TPE)
    print(coeffs17)

    for i in range(len(coeffsPE)):
        coeffsPE[i] /= 1.01 ** (2 * i + 1)

    A = torch.abs(torch.randn(5000, 70))
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    polarFactor = U @ Vh

    polarFactorNew = NewPolarExpress(A, T, coeffs17)

    polarFactorPE = PolarExpress(A, TPE, coeffsPE)

    diffPE = polarFactor - polarFactorPE
    diffNew = polarFactor - polarFactorNew
    normFactor = polarFactor.norm(dim=(-2, -1), keepdim=True)

    errPE = diffPE.norm(dim=(-2, -1), keepdim=True) / normFactor
    errNew = diffNew.norm(dim=(-2, -1), keepdim=True) / normFactor

    print(f"Polar Express error: {errPE}")
    print(f"New Express error: {errNew}")


"""
q=10, T = 3:
    [array([ 3.13422195e+01, -1.40818285e+03,  2.65294365e+04, -2.46146558e+05,
            1.28959429e+06, -4.11482454e+06,  8.29014494e+06, -1.05943865e+07,
            8.32597575e+06, -3.66978039e+06,  6.94272387e+05]), array([ 1.29332197e+01, -1.49932190e+02,  7.28824194e+02, -1.74480720e+03,
            2.35866197e+03, -1.94188205e+03,  1.00946804e+03, -3.32862742e+02,
            6.74968396e+01, -7.67621794e+00,  3.74710650e-01]), array([ 4.67319285e+00, -2.27857582e+01,  8.05002795e+01, -1.81091626e+02,
            2.65646905e+02, -2.59338586e+02,  1.69450202e+02, -7.31059510e+01,
            1.99610970e+01, -3.12226706e+00,  2.13017610e-01])]
"""


def main():
    test_polar()


if __name__ == "__main__":
    main()
