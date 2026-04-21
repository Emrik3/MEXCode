import time
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np
import torch
from evalPol import eval3, eval5, eval9, eval17, sastre8
from sympy.series.approximants import approximants

coeffs_dir = "coeffs/"


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


def roots(c, l, u, steps=100000):
    xs = np.linspace(l, u, steps)
    guesses = []

    prev_x = xs[0]
    prev_val = pp(c, prev_x)  # TODO: Fix this 1 to work guven the rescalar cushion

    for x in xs[1:]:
        curr_val = pp(c, x)

        # Check for sign change
        if prev_val * curr_val < 0:
            guesses.append((prev_x + x) / 2)
        prev_x = x
        prev_val = curr_val

    return guesses


def odd_remez(q, l, u, tol):
    x = np.zeros(q + 2)
    f = np.ones(q + 2)
    n = q + 2
    # Calculate initial guess of points as Chebyshev points
    for i in range(n):
        x[i] = 0.5 * (l + u) + 0.5 * (u - l) * np.cos((i) * np.pi / (n))
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
                    candidates.append(
                        r
                    )  # TODO: might have to change basis for high degree polynomials cause cant find all the roots.
        # Makes sence since we get like 0.001**41 and stuff like that.

        for guess in candidates:  # If they are too close we might have problems
            x_new.append(newton_pol(guess, c[:-1], tol))

        # Always include endpoints
        x_new = [l] + x_new + [u]

        # Sort for consistency
        x_new = np.array(sorted(x_new))

        if len(x_new) != n:
            # Fill the non found extremums with new points in the least dense area.
            while len(x_new) < n:
                # Compute gaps
                gaps = np.diff(x_new)

                # Index of largest gap
                idx = np.argmax(gaps)

                # Midpoint of the largest gap
                midpoint = (x_new[idx] + x_new[idx + 1]) / 2

                # Insert midpoint
                x_new = np.insert(x_new, idx + 1, midpoint)

        x = x_new

        # Make sure all unique points were found
        if len(x_new) == len(set(x_new)):
            x = np.array(x_new)
        else:
            print("Error: Could not find all points")
            break
        E = c[-1]

    return c


# --- Chebyshev helpers (mapped to [l, u]) ---


def plot_cheb(c, l, u):
    x = np.linspace(l, u)
    out = np.zeros(len(x))
    for j in range(len(x)):
        out[j] = eval_odd_cheb_poly(x[j], c, l, u)
    plt.plot(x, out)


def cheb_to_std(x, l, u):
    """Map x in [l,u] to t in [0,1]."""
    return (x - l) / (u - l)


def std_to_cheb(t, l, u):
    """Map t in [0,1] to x in [l,u]."""
    return (u - l) * t + l


def eval_odd_cheb_poly(x, coeffs, l, u):
    """
    Evaluate  p(x) = sum_i  coeffs[i] * T_{2i+1}(t)
    where t = cheb_to_std(x, l, u) and i = 0, ..., len(coeffs)-1.
    """
    t = cheb_to_std(x, l, u)
    # Build T_1, T_3, T_5, ... via recurrence
    # T_0=1, T_1=t, T_{n+1}=2t*T_n - T_{n-1}
    q = len(coeffs)  # number of odd terms
    max_order = 2 * q  # we need T_1 … T_{2q-1}
    T = np.zeros(max_order + 1)
    T[0] = 1.0
    if max_order >= 1:
        T[1] = t
    for k in range(2, max_order + 1):
        T[k] = 2.0 * t * T[k - 1] - T[k - 2]
    return sum(coeffs[i] * T[2 * i + 1] for i in range(q))


def eval_odd_cheb_deriv(x, coeffs, l, u):
    """
    Derivative  p'(x)  using  T_n'(t) = n * U_{n-1}(t)
    and the chain-rule factor  dt/dx = 2/(u-l).
    U_n satisfies  U_0=1, U_1=2t, U_{n+1}=2t*U_n - U_{n-1}.
    """
    t = cheb_to_std(x, l, u)
    q = len(coeffs)
    max_order = 2 * q
    # Build U_0 … U_{2q-2}
    U = np.zeros(max_order)
    U[0] = 1.0
    if max_order >= 2:
        U[1] = 2.0 * t
    for k in range(2, max_order):
        U[k] = 2.0 * t * U[k - 1] - U[k - 2]
    # d p / d t
    dp_dt = sum(coeffs[i] * (2 * i + 1) * U[2 * i] for i in range(q))
    # chain rule
    return dp_dt * (2.0 / (u - l))


def cheb_poly_extrema(coeffs, l, u, tol=1e-12):
    """
    Find interior extrema of  p(x) = sum_i coeffs[i]*T_{2i+1}(t(x))
    in (l, u) by companion-matrix roots of p'(t) in the T basis,
    then polish with Newton in x-space.

    p'(t) lives in the U basis:  p'(t) = sum_i coeffs[i]*(2i+1)*U_{2i}(t).
    We convert to the T basis (U_n = sum T) and use numpy's Chebyshev roots.
    """
    from numpy.polynomial.chebyshev import Chebyshev

    q = len(coeffs)
    # Degree of p'(t) in T-basis: highest term is T_{2q-2}  (from (2q-1)*U_{2q-2})
    # Build coefficient array in T-basis for p'(t)
    # U_n(t) = sum_{k=0}^{n} c_k T_k(t)  with T having same parity as n:
    #   U_{2m}(t) = T_0 + 2*T_2 + 2*T_4 + ... + 2*T_{2m}   (even terms)
    # So p'(t) = sum_i coeffs[i]*(2i+1) * U_{2i}(t)  — all even-order T basis
    max_T_order = 2 * (q - 1)  # highest T index in p'
    T_coeffs = np.zeros(max_T_order + 1)
    for i in range(q):
        n = 2 * i  # U_{2i} involves T_0,T_2,...,T_{2i}
        weight = coeffs[i] * (2 * i + 1)
        T_coeffs[0] += weight * 1.0  # T_0 coefficient in U_{2i}
        for k in range(1, i + 1):
            T_coeffs[2 * k] += weight * 2.0

    # Use numpy Chebyshev companion matrix to find roots in [0,1]
    cheb_poly = Chebyshev(T_coeffs)
    roots_t = cheb_poly.roots()

    interior = []
    for r in roots_t:
        if np.isreal(r):
            r = r.real
            if 0 < r < 1.0:
                x_guess = std_to_cheb(r, l, u)
                interior.append(x_guess)

    # Newton polish in x-space
    polished = []
    for x0 in interior:
        x_cur = x0
        for _ in range(50):
            fx = eval_odd_cheb_poly(x_cur, coeffs, l, u)
            dfx = eval_odd_cheb_deriv(x_cur, coeffs, l, u)
            if abs(dfx) < 1e-10:
                break
            step = fx / dfx
            x_cur -= step
            if abs(step) < tol:
                break
        if l < x_cur < u:
            polished.append(x_cur)

    return polished


# --- Main Remez loop ---


def odd_remez_cheb(q, l, u, tol):
    """
    Chebyshev-basis Remez for best odd approximation to sign(x) on [l, u].
    Uses basis  { T_1, T_3, ..., T_{2q-1} }  (q terms) plus error E.
    """
    n = q + 2  # number of interpolation points

    # Initial Chebyshev reference points
    x = np.array(
        sorted(
            0.5 * (l + u) + 0.5 * (u - l) * np.cos((2 * i + 1) * np.pi / (2 * n))
            for i in range(n)
        )
    )

    f = np.ones(n)  # sign(x) = 1 for x in [l, u] > 0
    old_E = np.inf
    E = 1000.0

    while abs(old_E - E) > tol:
        old_E = E

        # Build collocation matrix  A[j, i] = T_{2i+1}( t(x_j) )
        A = np.zeros((n, n))
        for j in range(n):
            t_j = cheb_to_std(x[j], l, u)
            # T values via recurrence
            T_prev, T_curr = 1.0, t_j
            A[j, 0] = T_curr  # T_1
            for i in range(1, n - 1):
                T_next = 2.0 * t_j * T_curr - T_prev
                T_prev = T_curr
                T_curr = T_next
                T_next = 2.0 * t_j * T_curr - T_prev
                T_prev = T_curr
                T_curr = T_next
                A[j, i] = T_curr  # T_{2i+1}
        A[:, -1] = (-1.0) ** np.arange(n)
        c = np.linalg.solve(A, f)
        poly_coeffs = c[:-1]  # Chebyshev coefficients

        # Find interior extrema of the error  e(x) = sign(x) - p(x)
        # Since sign(x)=1 on [l,u], extrema of e are extrema of p.
        x_new = cheb_poly_extrema(poly_coeffs, l, u, tol)
        x_new = [l] + x_new + [u]
        x_new = np.array(sorted(set(np.round(x_new, decimals=14))))
        # Pad with midpoints of largest gaps if too few points found
        while len(x_new) < n:
            gaps = np.diff(x_new)
            idx = np.argmax(gaps)
            midpoint = (x_new[idx] + x_new[idx + 1]) / 2.0
            x_new = np.insert(x_new, idx + 1, midpoint)

        x = x_new[:n]  # safety trim (shouldn't be needed)
        E = c[-1]

    return c  # last element is E; poly coefficients are c[:-1] in T_{1,3,...} basis


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


def plot_pol(c, l, u):
    x = np.linspace(l, u, 100000)
    plt.plot(x, p(c, x))
    plt.plot(x, 1 + 0 * x)


# Does this only best for degree 5, the cushioning and such?
# Maybe just do any degree not using this?
def get_all_coeffs_different_degrees(q_list, T, l=0.001):
    cushion = 0.02407327424182761
    cushion = 0
    u = 1
    all_coeffs = []
    eps = 1e-10
    llist = [l]
    k = 1

    for i in range(T):
        q = q_list[i]
        c = odd_remez(q, max(l, cushion * u), u, 1e-4)  # Make  more exact?
        if cushion * u > l:
            pl = p(c[:-1], l)
            pu = p(c[:-1], u)
            rescalar = 2 / (pl + pu)
            for i in range(len(c[:-1])):
                c[i] *= rescalar

        for i in range(len(c) - 1):
            c[i] /= (1.01) ** (2 * i + 1)

        l = p(c[:-1], l)
        llist.append(l)
        x = np.linspace(l, u, 1000)

        u = np.max(p(c[:-1], x))

        all_coeffs.append(c[:-1])
        if len(llist) > 1:
            print(l)
            # print((1 - llist[k]) / (1 - llist[k - 1]) ** (q + 1))
        k += 1
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
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    S = torch.linalg.svdvals(X.float())
    print(f"Singular value range after normalization: [{S.min():.4f}, {S.max():.4f}]")

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
        S = torch.linalg.svdvals(X.float())
        print(f"Singular value range at iter: [{S.min():.4f}, {S.max():.4f}]")
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


@torch.compile
def PolarTest(G: torch.Tensor, steps: int, coeffs_list) -> torch.Tensor:
    # TODO: accumulate in 32 bult mult in 16
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    for c in coeffs_list:
        for i in range(len(c) - 1):
            c[i] /= 1.01 ** (2 * i + 1)
    for a, b, c, d, e, f, g, h, i in coeffs_list:
        A = X @ X.mT
        B = (
            b * A
            + c * A @ A
            + d * A @ A @ A
            + e * A @ A @ A @ A
            + f * A @ A @ A @ A @ A
            + g * A @ A @ A @ A @ A @ A
            + h * A @ A @ A @ A @ A @ A @ A
            + i * A @ A @ A @ A @ A @ A @ A @ A
        )
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


def test_approximation(q, l=0.001):
    T = len(q)
    coeffs17 = get_all_coeffs_different_degrees(q, T, l)
    print(coeffs17)
    # np.save(coeffs_dir + "coeffs.npy", np.array(coeffs17, dtype=object))
    x_plt = np.linspace(0, 1, 10000)

    x = np.linspace(0, 1, 10000)
    tot_degree = 1
    for i in range(T):
        x = p(coeffs17[i], x)
        tot_degree *= 2 * q[i] + 1
        l = p(coeffs17[i], l)

    plt.plot(x_plt, x, label=f"Total degree = {tot_degree}, d = {2 * np.array(q) + 1}")

    plt.legend()


def get_all_coeffs_different_degrees_cheb(q_list, T, l=0.001):
    cushion = 0.02407327424182761
    cushion = 0
    u = 1
    all_coeffs = []
    eps = 1e-10
    llist = [l]
    k = 1

    for i in range(T):
        q = q_list[i]
        c = odd_remez_cheb(q, max(l, cushion * u), u, 1e-8)  # Make  more exact?
        if cushion * u > l:
            pl = eval_odd_cheb_poly(l, c[:-1], l, u)
            pu = eval_odd_cheb_poly(u, c[:-1], l, u)
            rescalar = 2 / (pl + pu)
            for i in range(len(c[:-1])):
                c[i] *= rescalar

        for i in range(len(c) - 1):
            c[i] /= (1.01) ** (2 * i + 1)
        plot_cheb(c, l, u)
        plt.show()
        l = eval_odd_cheb_poly(l, c[:-1], l, u)
        llist.append(l)
        x = np.linspace(l, u, 1000)

        u = 2 - l
        all_coeffs.append(np.array(c[:-1]))
        if len(llist) > 1:
            print(l)
            # print((1 - llist[k]) / (1 - llist[k - 1]) ** (q + 1))
        k += 1
    return all_coeffs


def test_approximation_cheb(q, l=0.001):
    T = len(q)
    coeffs17 = get_all_coeffs_different_degrees_cheb(q, T, l)
    print(coeffs17)
    # np.save(coeffs_dir + "coeffs.npy", np.array(coeffs17, dtype=object))
    x_plt = np.linspace(0, 1, 1000)

    x = np.linspace(0, 1, 1000)
    tot_degree = 1
    u = 1
    for i in range(T):
        tot_degree *= 2 * q[i] + 1
        l = eval_odd_cheb_poly(l, coeffs17[i], l, u)
        u = 2 - l

        for j in range(len(x)):
            x[j] = eval_odd_cheb_poly(x[j], coeffs17[i], l, u)

    plt.plot(x_plt, x, label=f"Total degree = {tot_degree}, d = {2 * np.array(q) + 1}")

    plt.legend()


def interpolateTest(x, y):
    A = np.array([[x[i] ** (2 * j + 1) for j in range(3)] for i in range(4)])

    f = np.array(y)
    c, residuals, rank, s = np.linalg.lstsq(A, f, rcond=None)
    return c


def plot_all(q, l=0.001):
    T = len(q)
    coeffs17 = get_all_coeffs_different_degrees(q, T, l)
    x_plt = np.linspace(l, 1, 10000)

    x = np.linspace(l, 1, 10000)
    tot_degree = 1
    eps = 0.01

    ep2 = np.array([0.04794705, 0.74936796, 1.63851199, 1.99171281])
    y = np.array([min(ep2), max(ep2) + eps, min(ep2) - eps, max(ep2)])
    ep1 = np.array([0.02407327, 0.37624298, 0.82266478, 1])

    coeffs17[0] = interpolateTest(ep1, y)

    for i in range(T):
        x = p(coeffs17[i], x)
        # x = x + l * (x - 1.06)
        tot_degree *= 2 * q[i] + 1

    ep = [0.04794705, 0.74936796, 1.63851199, 1.99171281]

    xplt2 = np.linspace(l, 2, 10000)

    for i in range(len(ep)):
        plt.plot(xplt2, 0 * xplt2 + ep[i])
    print(
        f"New min: {min(x)}, new max: {max(x)}"
    )  # It seems to work, i get larger l and smaller u? Double check these results
    plt.plot(x_plt, p(coeffs17[0], x_plt), label="P1")
    plt.plot(xplt2, p(coeffs17[1], xplt2), label="P2")
    plt.plot(x_plt, x, label=f"Total degree = {tot_degree}, d = {2 * np.array(q) + 1}")

    plt.legend()
    plt.show()


def test_polar():
    # TODO: using 2,2,2,2,8 seems to be worse than 2,2,2,2,2 whcih makes no sence. might be an error in the sastre implementation
    # or the stability is present even when l is small
    q = [8, 8, 8]  # TODO: Order seems to matter for stability.
    qPE = [2, 2, 2, 2, 2]
    T = len(q)
    TPE = len(qPE)
    coeffs17 = get_all_coeffs_different_degrees(q, T)
    coeffsPE = get_all_coeffs_different_degrees(qPE, TPE)

    A = torch.abs(torch.randn(500, 70))  # Varför funkar det inte för stora matriser?
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

    errPE = torch.linalg.matrix_norm(diffPE, ord=2)
    errNew = torch.linalg.matrix_norm(diffNew, ord=2)

    print(f"Polar Express error: {errPE}")
    print(f"New Express error: {errNew}")


def approxs():
    test_approximation_cheb([2, 2, 2, 2], l=0.001)
    plt.show()


def main():
    # TODO: Some issues with the convergence of newton when the tol is too high, for large polynomials it does not converge.
    # TODO: Have some issues when using degree 3, since only one point it is not working correctly, change to just have exact solution when it is of degree 3.
    approxs()
    # test_polar()


if __name__ == "__main__":
    main()
