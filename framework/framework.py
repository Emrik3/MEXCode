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
    """Find roots of odd poynomial with coefficients c."""
    xs = np.linspace(l, u, steps)
    guesses = []

    prev_x = xs[0]
    prev_val = pp(c, prev_x)

    for x in xs[1:]:
        curr_val = pp(c, x)

        # Check for sign change
        if prev_val * curr_val < 0:
            guesses.append((prev_x + x) / 2)
        prev_x = x
        prev_val = curr_val

    return guesses




def odd_remez(q, l, u, tol):
    """Remez algorithm for approximation of the sign function of any degree."""
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
        try:
            c = np.linalg.solve(A, f)
        except np.linalg.LinAlgError:
            if q == 2:
                return [1.875, -1.25, 0.375, 0]
            else:
                raise np.linalg.LinAlgError("Singular Matrix")

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

        # Make sure all unique points were found
        if len(x_new) == len(set(x_new)):
            x = np.array(x_new)
        else:
            print("Error: Could not find all points")
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
    """Newton's method the refine root search"""
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


def get_all_coeffs_different_degrees(q_list, T, l=0.001):
    """Main function using Remez to get the coefficents for total composition of polynomials of different degrees."""
    cushion = 0.02407327424182761 # Set cushion to 0 if it is not wished to be used.
    #cushion = 0
    u = 1
    all_coeffs = []
    eps = 1e-10
    llist = [l]
    k = 1

    for i in range(T):
        if 1-l <= 1e-9:
            all_coeffs.append( [1.875, -1.25, 0.375])
            continue
        q = q_list[i]
        c = odd_remez(
            q, max(l, cushion * u), u, 1e-8
        )  # Make  more exact?
        if cushion * u > l:
            pl = p(c[:-1], l)
            pu = p(c[:-1], u)
            rescalar = 2 / (pl + pu)
            for i in range(len(c[:-1])):
                c[i] *= rescalar

        #for i in range(len(c) - 1):
        #    c[i] /= (1.01) ** (2 * i + 1)

        l = p(c[:-1], l)
        llist.append(l)
        x = np.linspace(l, u, 1000)

        u = 2-l

        all_coeffs.append(c[:-1])
        if len(llist) > 1:
            print(1 - l)
            # print((1 - llist[np.max(p(c[:-1], x))k]) / (1 - llist[k - 1]) ** (q + 1))
        k += 1

    return all_coeffs


@torch.compile
def PolarExpress(G: torch.Tensor, steps: int, coeffs_list) -> torch.Tensor:
    """Degree 5 five times."""
    assert G.ndim >= 2
    X = G.double()  # This is used for testing, should use bfloat16 in machine learning setting.
    if G.size(-2) > G.size(-1):
        X = X.mT  # this reduces FLOPs

    for a, b, c in coeffs_list:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX ˆ3 + cX ˆ5
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile
def NewPolarExpress(G: torch.Tensor, steps: int, coeffs_list) -> torch.Tensor:
    """Total evaluation of the polynomial composition using."""
    # TODO: accumulate in 32 bult mult in 16
    assert G.ndim >= 2
    X = G.double() # This is used for testing, should use bfloat16 in machine learning setting.
    if G.size(-2) > G.size(-1):
        X = X.mT

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


"""Compare answer."""
@torch.compile
def PolarTest(G: torch.Tensor, steps: int, coeffs_list) -> torch.Tensor:
    # TODO: accumulate in 32 bult mult in 16
    assert G.ndim >= 2
    X = G.double() # This is used for testing, should use bfloat16 in machine learning setting.
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

    # TODO: Some issues with the convergence of newton when the tol is too high, for large polynomials it does not converge.


colormap = {2: "#6298D2", 8: "#004791"}


def test_approximation(q, l=0.001):
    """Test function, get coefficents of all polynomials in composition and plot the resulting approximation."""
    T = len(q)
    coeffs17 = get_all_coeffs_different_degrees(q, T, l)
    print(coeffs17)
    np.save(coeffs_dir + "coeffs.npy", np.array(coeffs17, dtype=object))
    x_plt = np.linspace(0, 1, 10000)

    x = np.linspace(0, 1, 10000)
    tot_degree = 1
    for i in range(T):
        x = p(coeffs17[i], x)
        tot_degree *= 2 * q[i] + 1
        l = p(coeffs17[i], l)

    real_plots(coeffs17, q)
    """plt.plot(
        x_plt,
        x,
        linewidth=1.5,
        label=r"Total degree $= {}$, $d = {}$".format(
            int(tot_degree), [int(v) for v in 2 * np.array(q) + 1]
        ),
        # color=colormap.get(q[0], None),
    )
    plt.legend(fontsize=10)"""


def test_polar():
    """Test accuracy in matrix setting"""
    q = [8, 8, 8]  
    qPE = [2, 2, 2, 2, 2]
    T = len(q)
    TPE = len(qPE)
    coeffs17 = get_all_coeffs_different_degrees(q, T)
    coeffsPE = get_all_coeffs_different_degrees(qPE, TPE)

    A = torch.abs(torch.randn(500, 70))  
    A = A / (A.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
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


def real_plots(coeffs, q_list):
    """Plot relative Frobenius error of a gradient matrix from machine learning setting."""
    A = torch.load("h3_c_attn_grads.pt", map_location="cpu")
    if not torch.is_tensor(A):
        # in case it's saved as a state_dict / dict of tensors
        raise ValueError("Loaded object is not a tensor — inspect its structure first")
    A = A.double()
    if A.ndim > 2:
        # flatten any leading batch/head dims down to a single 2D matrix
        A = A.reshape(-1, A.shape[-1])

    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    polarFactor = U @ Vh

    A17 = A.mT.clone()
    APE = A.mT.clone()

    A17 = A17 / (A17.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    APE = APE / (APE.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    spec17 = [1]
    specPE = [1]

    for i in range(len(coeffs)):
        APE = APE*0
        for j in range(len(coeffs[i])):

            APE = APE + coeffs[i][j] * torch.linalg.matrix_power(A17@A17.mT, j) @ A17
        A17 = APE.clone()
        specPE.append(torch.linalg.matrix_norm(A17.mT - polarFactor, ord="fro") / torch.linalg.matrix_norm(polarFactor, ord="fro"))


    q_to_mult = {1: 2, 2: 3, 4: 4, 8: 5, 12: 6}

    mults = [0]
    for q in q_list:
        mults.append(mults[-1] + q_to_mult[q])

    if q_list[0] == 2:
        lab = "Polar Express"
    else:
        lab = "Proposed New Method"
    plt.plot(
        mults,
        specPE,
        linewidth=3,
        marker="o",
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=1.5,
        label=lab,
    )

    plt.xlabel("Matrix-Matrix Multiplications", fontsize=14)
    plt.ylabel("Relative Frobenius Error", fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(True, alpha=0.25)
    plt.ylim(0, 1.05)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.legend(frameon=False, fontsize=11)
    plt.tight_layout()
    plt.savefig("gradient_matrix_error.pdf", format="pdf", bbox_inches="tight")



def approxs():
    plt.figure(figsize=(6.2, 4.0), dpi=150)

    test_approximation([2,2,2,2,2,2, 2,2,2,2,2,2])
    test_approximation([8,8,8,8,2,2,2,2,2])

    # plt.savefig("degree17lasttwice.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def main():
    approxs()


if __name__ == "__main__":
    main()
