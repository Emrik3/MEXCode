import time
from itertools import repeat
from types import LambdaType
from unittest import TestProgram

import matplotlib.pyplot as plt
import numpy as np
import torch
from evalPol import eval3, eval5, eval9, eval17, sastre8

coeffs_dir = "coeffs/"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.5
plt.rc("text", usetex=False)
plt.rc("legend", fontsize=10)


def f(c, x):
    for cc in c:
        out = 0
        for i in range(len(cc)):
            out += cc[i] * x ** (2 * i + 1)
        x = out
    return x


def p(allc, m, x):
    out = np.zeros(m + 2)

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
    c = allc[idx : idx + m + 2]
    # [-0.30990275  4.5558329  -0.13127587  1.          0.        ]

    out[0] = 1
    out[1] = x**2
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
    c = c0.copy()
    err = (1 - 0.001) ** (2**m)

    err2 = 1000
    while True:
        res = r(zs, c, m, fc)
        jac = J(zs, c, m)
        F = 0.5 * np.dot(res, res)
        gr = jac.T @ res

        p, *_ = np.linalg.lstsq(jac, -res, rcond=None)
        if gr @ p >= 0:
            raise RuntimeError("Direction is not descent.")

        # Armijo
        alpha = alpha0
        i = 0
        while True:
            i += 1
            c_trial = c + alpha * p
            res_trial = r(zs, c_trial, m, fc)
            F_trial = 0.5 * np.dot(res_trial, res_trial)

            if F_trial <= F + a * alpha * (gr @ p):
                break
            alpha *= rho
        c = c_trial
        err2 = np.sum([np.abs(r(zs, c, m, fc)) ** 2 for z in zs]) / len(zs)

        if i > max_iter:
            break

        if err2 < tol:
            break

    return c, err2


def newton(fc, tol, max_iter, zs, c0, m, alpha0=1.0, rho=0.5, a=1e-4):
    """
    Newton's method version of the Gauss-Newton routine.
    Assumes number of residuals == number of variables and J is invertible.
    """
    c = c0.copy()
    err2 = 1000
    for i in range(max_iter):
        res = r(zs, c, m, fc)
        jac = J(zs, c, m)

        # Solve J p = -r (Newton step)
        try:
            p = np.linalg.solve(jac, -res)
        except np.linalg.LinAlgError:
            raise RuntimeError("Jacobian is singular or ill-conditioned.")

        # Optional: keep line search for stability
        alpha = alpha0
        while True:
            c_trial = c + alpha * p
            res_trial = r(zs, c_trial, m, fc)

            # Use residual norm instead of F
            if np.linalg.norm(res_trial) <= (1 - a * alpha) * np.linalg.norm(res):
                break
            alpha *= rho

        c = c_trial

        err2 = np.linalg.norm(r(zs, c, m, fc)) ** 2 / len(zs)
        print(err2)

        if err2 < tol:
            break

    return c, err2


def roots(c, l, u, steps=10000):
    xs = np.linspace(l, u, steps)
    guesses = []
    eps = 1e-8

    prev_x = xs[0]
    prev_val = 1 - f(c, prev_x)  # TODO: Fix this 1 to work guven the rescalar cushion

    for x in xs[1:]:
        curr_val = 1 - f(c, x)

        # Check for sign change
        if prev_val * curr_val < 0:
            guesses.append((prev_x + x) / 2)
        prev_x = x
        prev_val = curr_val

    return guesses


def rootsp(c, l, u, steps=1000):
    xs = np.linspace(l, u, steps)
    guesses = []
    eps = 1e-12

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


# Derivative of polynomial
def fp(c, x):
    out = 0
    for i in range(len(c)):
        out += c[i] * (2 * i + 1) * x ** (2 * i)
    return out


# Second derivative of polynomial
def fpp(c, x):
    out = 0
    for i in range(1, len(c)):
        out += c[i] * (2 * i + 1) * (2 * i) * x ** (2 * i - 1)
    return out


def newton_pol(x, c, tol):
    err = 100.0
    while err > tol:
        x_new = x - fp(c, x) / fpp(c, x)
        err = np.abs(x_new - x)
        x = x_new
    return x


def plot_pol(c, m, l, u):
    x = np.linspace(l, u, 1000)
    vals = [p(c, m, xx) for xx in x]
    plt.plot(x, vals, label="New")
    # plt.plot(x, 1 + 0 * x)


def plot_og(c, l, u):
    x = np.linspace(l, u, 1000)
    vals = [f(c, xx) for xx in x]
    plt.plot(x, vals, label="PE")


"""coNew = [[ 8.19006284e+00 -1.13414979e+01  5.26952866e+00 -1.13557551e+01
 -8.55755878e+00  2.99419201e-01  3.93687364e-01 -1.36299949e-01
  8.75437928e-01  5.72242006e+00 -1.33495705e+01  2.40272297e+01
 -6.90335422e+00 -8.15817768e+00  3.42806735e-02 -5.95545030e-01
 -1.49974268e+00  2.87975012e+00  1.26506513e-01  3.60109930e-03
  1.89759345e+00 -1.00000000e+00  1.00000000e+00],

]"""
co17 = [
    (
        [
            [8.19006284e00, -1.13414979e01],
            [5.26952866e00, -1.13557551e01, -8.55755878e00],
            [2.99419201e-01, 3.93687364e-01, -1.36299949e-01, 8.75437928e-01],
        ],
        [
            [5.72242006e00, -1.33495705e01],
            [2.40272297e01, -6.90335422e00, -8.15817768e00],
            [3.42806735e-02, -5.95545030e-01, -1.49974268e00, 2.87975012e00],
        ],
        [
            1.26506513e-01,
            3.60109930e-03,
            1.89759345e00,
            -1.00000000e00,
            1.00000000e00,
        ],
    ),
    (
        [
            [1.11440745e01, -6.93968288e00],
            [7.99906695e00, -1.52038107e01, -1.18648297e01],
            [4.87264211e00, -5.81189185e00, -4.14955237e00, 3.76700505e00],
        ],
        [
            [2.79206430e00, -1.21731802e01],
            [1.83302276e01, -1.25959889e01, -8.58183598e00],
            [5.45734344e-01, -6.52046697e-01, -4.88640155e-01, -3.61349369e-01],
        ],
        [
            -6.70151350e-03,
            1.32073180e-02,
            -1.32714544e00,
            -1.00000000e00,
            1.00000000e00,
        ],
    ),
    (
        [
            [1.62992507e01, -1.09165807e01],
            [3.79130325e00, -1.20918536e01, -8.44971299e00],
            [2.04438607e00, -2.30355479e00, -3.02265517e00, -1.78983106e00],
        ],
        [
            [5.24600604e00, -1.62986125e01],
            [2.41105188e01, -1.04430341e01, -7.06863589e00],
            [-7.56679623e00, 4.75054075e00, 2.06561878e00, 1.86435561e00],
        ],
        [
            -5.36337617e-03,
            2.07934129e-02,
            -5.29262448e-02,
            -1.00000000e00,
            1.00000000e00,
        ],
    ),
]


def approxs():
    # TODO: Get the old code for this with the random initial guess.

    # TODO: A way of finding a guess that kind of works, also can it find this or
    # do we need to give it a higher order? Cuase why would it not be able to find an approimation?
    m = 3  # TODO: Why would too high degree make it not work?
    l = 0.9738946167139803
    u = 2 - l

    fc = [np.load(coeffs_dir + "coeffs.npy", allow_pickle=True)[3]]
    print(fc)
    extremums = []
    coeffs_for_roots = derivative_coeffs(fc[0])
    root_guess = np.roots(coeffs_for_roots)
    candidates = []
    for r in root_guess:
        if np.isreal(r):
            r = r.real
            if r > 0:
                candidates.append(r)

    for guess in candidates:  # If they are too close we might have problems
        extremums.append(newton_pol(guess, fc[0], 1e-8))

    # Always include endpoints
    extremums = [l] + extremums + [u]

    # Sort for consistency
    extremums = np.array(sorted(extremums))
    intersects = roots(fc, l, u)
    guesses = []
    for i in range(len(intersects)):
        guesses.append(extremums[i])
        guesses.append(intersects[i])
    guesses.append(extremums[-1])
    print(len(guesses))
    plt.scatter(guesses, np.ones(len(guesses)))
    plot_og(fc, l, u)
    plt.show()

    prev_corr = np.load(coeffs_dir + "coeffsPolsetThirdIter3.npy", allow_pickle=True)
    print(prev_corr)
    plot_pol(prev_corr, m, l, u)
    plot_og(fc, l, u)
    plt.show()
    guess = prev_corr

    while True:
        c, err = gn(fc, 1e-8, 50, guesses, guess, m)
        plot_og(fc, l, u)
        plt.scatter(guesses, np.ones(len(guesses)))
        plot_pol(c, m, l, u)
        plt.show()
        print(err)
        if err < 1e-8:
            break
    np.save(coeffs_dir + "coeffsPolsetFourthCush" + str(m) + ".npy", c)
    plt.scatter(guesses, np.ones(len(guesses)))
    plot_pol(c, m, l, u)
    print(p(c, m, l))
    plot_og(fc, l, u)
    plt.show()
    # Can just solve it if i have same number of points as variables, then we solve interpolation.
    # Interpolate at the points of maximum deviation of combined poly omial?


# Have been able to find, degree 12 in 4 mults, degree 8 in 3, degree 4 in 2.
@torch.compile
def NewPolarExpress(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    for i in range(3):
        X = pX(co17[i], 3, X)

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile
def pX(allc, m, X: torch.Tensor):
    # TODO: Doulbe check this is implemented correctly, get very large error. might be because instability though...
    n = X.shape[0]

    # out[i] stores an (n x n) matrix, not a scalar
    out = torch.zeros(m + 2, n, n, dtype=X.dtype, device=X.device)

    A = []
    B = []
    idx = 0
    for i in range(m):
        k = i + 2
        A.append(allc[idx : idx + k])
        idx += k
    for i in range(m):
        k = i + 2
        B.append(allc[idx : idx + k])
        idx += k
    c = allc[idx : idx + m + 2]

    out[0] = torch.eye(n, dtype=X.dtype, device=X.device)  # "1" as identity matrix
    out[1] = X @ X.mT

    for i in range(m):
        out1 = torch.zeros(n, n, dtype=X.dtype, device=X.device)
        out2 = torch.zeros(n, n, dtype=X.dtype, device=X.device)
        for j in range(len(A[i])):
            out1 = out1 + A[i][j] * out[j]
            out2 = out2 + B[i][j] * out[j]

        out[i + 2] = c[i + 2] * (out1 @ out2)
    # out[0] = c[0] * out[0]
    # out[1] = c[1] * out[1]
    return torch.sum(out, dim=0) @ X


@torch.compile
def PolarExpress(G: torch.Tensor, steps: int, coeffs_list) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.float()  # for speed
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


co = [
    (
        [
            [8.19006284e00, -1.13414979e01],
            [5.26952866e00, -1.13557551e01, -8.55755878e00],
            [2.99419201e-01, 3.93687364e-01, -1.36299949e-01, 8.75437928e-01],
        ],
        [
            [5.72242006e00, -1.33495705e01],
            [2.40272297e01, -6.90335422e00, -8.15817768e00],
            [3.42806735e-02, -5.95545030e-01, -1.49974268e00, 2.87975012e00],
        ],
        [
            1.26506513e-01,
            3.60109930e-03,
            1.89759345e00,
            -1.00000000e00,
            1.00000000e00,
        ],
    ),
    (
        [
            [1.11440745e01, -6.93968288e00],
            [7.99906695e00, -1.52038107e01, -1.18648297e01],
            [4.87264211e00, -5.81189185e00, -4.14955237e00, 3.76700505e00],
        ],
        [
            [2.79206430e00, -1.21731802e01],
            [1.83302276e01, -1.25959889e01, -8.58183598e00],
            [5.45734344e-01, -6.52046697e-01, -4.88640155e-01, -3.61349369e-01],
        ],
        [
            -6.70151350e-03,
            1.32073180e-02,
            -1.32714544e00,
            -1.00000000e00,
            1.00000000e00,
        ],
    ),
    (
        [
            [1.62992507e01, -1.09165807e01],
            [3.79130325e00, -1.20918536e01, -8.44971299e00],
            [2.04438607e00, -2.30355479e00, -3.02265517e00, -1.78983106e00],
        ],
        [
            [5.24600604e00, -1.62986125e01],
            [2.41105188e01, -1.04430341e01, -7.06863589e00],
            [-7.56679623e00, 4.75054075e00, 2.06561878e00, 1.86435561e00],
        ],
        [
            -5.36337617e-03,
            2.07934129e-02,
            -5.29262448e-02,
            -1.00000000e00,
            1.00000000e00,
        ],
    ),
]
for i in range(len(co) - 1):
    for j in range(len(co[i][2])):
        co[i][2][j] /= 1.01 ** (j + 2)
# safety factor for numerical stability (but exclude last polynomial)
"""coeffs_list = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in coeffs_list[:-1]
] + [coeffs_list[-1]]"""


@torch.compile
def MachPolar(G: torch.Tensor) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.float()  # for speed
    if G.size(-2) > G.size(-1):
        X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    n = X.shape[0]
    t = 0
    for m in [3, 3, 3]:
        A = co[t][0]
        B = co[t][1]
        c = co[t][2]
        t += 1
        out = torch.zeros(m + 2, n, n, dtype=X.dtype, device=X.device)
        out[0] = torch.eye(n, dtype=X.dtype, device=X.device)
        out[1] = X @ X.mT
        for i in range(m):
            out1 = torch.zeros(n, n, dtype=X.dtype, device=X.device)
            out2 = torch.zeros(n, n, dtype=X.dtype, device=X.device)
            for j in range(len(A[i])):
                out1 = out1 + A[i][j] * out[j]
                out2 = out2 + B[i][j] * out[j]

            out[i + 2] = c[i] * (out1 @ out2)
        X = torch.sum(out, dim=0) @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def MachPolarScalarTest221(x):

    t = 0
    tt = [0, 1, 1]
    for m in [3, 3, 3]:
        A = co[tt[t]][0]
        B = co[tt[t]][1]
        c = co[tt[t]][2]
        t += 1
        out = np.zeros(m + 2)
        out[0] = 1
        out[1] = x * x
        for i in range(m):
            out1 = 0
            out2 = 0
            for j in range(len(A[i])):
                out1 = out1 + A[i][j] * out[j]
                out2 = out2 + B[i][j] * out[j]

            out[i + 2] = c[i] * (out1 * out2)
        x = np.sum(out) * x
    return x


def MachPolarScalarTest(x):

    t = 0
    for m in [3, 3]:
        A = co[t][0]
        B = co[t][1]
        c = co[t][2]
        t += 1
        out = np.zeros(m + 2)
        out[0] = 1
        out[1] = x * x
        for i in range(m):
            out1 = 0
            out2 = 0
            for j in range(len(A[i])):
                out1 = out1 + A[i][j] * out[j]
                out2 = out2 + B[i][j] * out[j]

            out[i + 2] = c[i] * (out1 * out2)
        x = np.sum(out) * x
    return x


def test_polar():
    # TODO: using 2,2,2,2,8 seems to be worse than 2,2,2,2,2 whcih makes no sence. might be an error in the sastre implementation
    # or the stability is present even when l is small
    m = 3
    coeffs1 = np.load(coeffs_dir + "coeffsPolsetFirstIter3.npy")
    coeffs2 = np.load(coeffs_dir + "coeffsPolsetSecondIter3.npy")
    coeffs3 = np.load(coeffs_dir + "coeffsPolsetThirdIter3.npy")

    coeffs_list = co
    x = np.linspace(0, 1, 1000)
    x_new = np.zeros(1000)
    """for i in range(len(x)):
        x_new[i] = p(coeffs1, m, x[i])
        x_new[i] = p(coeffs2, m, x_new[i])
        x_new[i] = p(coeffs3, m, x_new[i])
    plt.plot(x, x_new)
    plt.show()"""

    A = torch.randn(500, 70)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    polarFactor = U @ Vh

    T = 3
    TPE = 5

    coeffsPE = [
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
        (3.3184196573706015, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
        # (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
        # (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
        # (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
    ]

    polarFactorNew = MachPolar(A)
    polarFactorPE = PolarExpress(A, TPE, coeffsPE)

    diffPE = polarFactor - polarFactorPE
    diffNew = polarFactor - polarFactorNew

    rel = torch.linalg.matrix_norm(polarFactor, ord="fro")

    errPE = torch.linalg.matrix_norm(diffPE, ord="fro") / rel
    errNew = torch.linalg.matrix_norm(diffNew, ord="fro") / rel

    print(f"Polar Express error: {errPE}")
    print(f"New Express error: {errNew}")


def composite_gnremez(m, ii, l, u):
    # TODO: implement a thing that find the polynomial on the form of Polset and then saves them and applies polar express to them, this seems to be stable
    # Tried for one iteration and it seemed to work

    fc = [np.load(coeffs_dir + "coeffs" + str(ii) + str(m) + ".npy", allow_pickle=True)]
    extremums = []
    coeffs_for_roots = derivative_coeffs(fc[0])
    root_guess = np.roots(coeffs_for_roots)
    candidates = []
    for r in root_guess:
        if np.isreal(r):
            r = r.real
            if r > 0:
                candidates.append(r)

    for guess in candidates:  # If they are too close we might have problems
        extremums.append(newton_pol(guess, fc[0], 1e-8))

    # Always include endpoints
    extremums = [l] + extremums + [u]

    # Sort for consistency
    extremums = np.array(sorted(extremums))
    intersects = roots(fc, l, u)
    guesses = []
    for i in range(len(intersects)):
        guesses.append(extremums[i])
        # guesses.append(intersects[i])
    guesses.append(extremums[-1])
    plot_og(fc, l, u)
    plt.scatter(guesses, np.ones(len(guesses)))

    prev_corr = np.load(coeffs_dir + "coeffsPolset" + str(ii) + str(m - 1) + ".npy")

    guess = np.zeros(m**2 + 4 * m + 2)
    for i in range(((m - 1) ** 2 + 3 * (m - 1)) // 2):
        guess[i] = prev_corr[i]
        guess[i + (m**2 + 3 * m) // 2] = prev_corr[
            i + ((m - 1) ** 2 + 3 * (m - 1)) // 2
        ]
    for i in range(m + 1):
        guess[-i - 2] = prev_corr[-i - 1]
    plot_pol(guess, m, l, u)
    plt.show()
    while True:
        guess = np.zeros(m**2 + 4 * m + 2)
        for i in range(((m - 1) ** 2 + 3 * (m - 1)) // 2):
            guess[i] = prev_corr[i]
            guess[i + (m**2 + 3 * m) // 2] = prev_corr[
                i + ((m - 1) ** 2 + 3 * (m - 1)) // 2
            ]
        for i in range(m + 1):
            guess[-i - 2] = prev_corr[-i - 1]
        k = 0

        for i in range(len(guess)):
            if guess[i] == 0:
                guess[i] = -(np.random.rand()) * guess[i - 1] / np.abs(guess[i - 1])
                k += 1
            if i == (m**2 + 3 * m) // 2:
                k = 0
        guess[-1] = -guess[-2] / np.abs(guess[-2])

        c, err = gn(fc, 1e-8, 50, guesses, guess, m)
        print(err)
        if err < 1e-8:
            break
    return c


def main():
    # TODO: Some issues with the convergence of newton when the tol is too high, for large polynomials it does not converge.
    # TODO: Have some issues when using degree 3, since only one point it is not working correctly, change to just have exact solution when it is of degree 3.
    # approxs()
    # test_polar()
    # composite_gnremez(3, 1, 0.02, 2 - 0.02)
    n = 1000
    x = np.linspace(0.001, 1, n)
    y = []
    yNormal = []
    for xx in x:
        y.append(MachPolarScalarTest221(xx))
        yNormal.append(MachPolarScalarTest(xx))
    plt.plot(x, y, label=f"Total degree = {17 * 17 * 17}, Final Polynomial Twice")
    plt.plot(x, yNormal, label=f"Total degree = {17 * 17}, d=[17, 17]")
    plt.legend()
    plt.savefig("sametwice.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
