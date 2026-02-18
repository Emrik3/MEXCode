import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# Define symbolic variables
z = sp.Symbol("z")
c = sp.symbols("c0:41")  # creates c0, c1, ..., c40


def p_sym(z, c):
    return c[0] * z + c[1] * z**3 + c[2] * z**5


def k_sym(z, c):
    return c[0] * z + c[1] * z**3 + c[2] * z**5


def g(z, c):
    q1 = k_sym(z, c[0:3]) + c[3] * p_sym(z, c[5:8]) ** 3 + c[4] * p_sym(z, c[5:8]) ** 5
    q2 = (
        k_sym(z, c[8:11])
        + k_sym(p_sym(z, c[5:8]), c[11:14])
        + c[14] * q1**3
        + c[15] * q1**5
    )
    q3 = (
        k_sym(z, c[16:19])
        + k_sym(p_sym(z, c[5:8]), c[19:22])
        + k_sym(q1, c[22:25])
        + c[25] * q2**3
        + c[26] * q2**5
    )
    q4 = (
        k_sym(z, c[27:30])
        + k_sym(p_sym(z, c[5:8]), c[30:33])
        + k_sym(q1, c[33:36])
        + k_sym(q2, c[36:39])
        + c[39] * q3**3
        + c[40] * q3**5
    )
    return q4


# Build symbolic expression once
g_expr = g(z, c)

# Differentiate symbolically w.r.t. each coefficient
dg_dc = [sp.diff(g_expr, ci) for ci in c]

# Lambdify everything for fast numerical evaluation
# lambdify turns sympy expressions into numpy functions
args = [z] + list(c)
g_func = sp.lambdify(args, g_expr, "numpy")
dg_funcs = [sp.lambdify(args, dgi, "numpy") for dgi in dg_dc]


def g_numerical(z_val, c_val):
    return g_func(z_val, *c_val)


def J(zs, c_val):
    """Jacobian using symbolic derivatives - much more stable than finite diff."""
    return np.array(
        [[dg_funcs[i](z_val, *c_val) for i in range(len(c_val))] for z_val in zs]
    )


def r(zs, c):
    """
    Added penalty term at the start to make it grow quickley, but it is hard to do gauss newton given that we want bounds on maximum error.
    """
    return np.array([(g_numerical(z, c) - 1) for z in zs]).T


def chebyshev(n, l, u, alpha=2.0):
    """
    Includes the endpoints l and u.
    """

    k = np.arange(n)
    x = np.cos((k / (n - 1)) ** alpha * np.pi)
    return (l + u) / 2 + (u - l) / 2 * x


def gn(
    tol,
    max_iter,
    zs,
    c=np.array([0, 0, 0, -3, 0.5, 8, -25, 17]).T,
):
    """
    Very sensitive to starting guess, should have starting guess be the value that is from remez maybe.
    """
    err = 1
    i = 0
    print(g(0.5, c))
    while err > tol and i < max_iter:
        print(err)
        c_old = c
        try:
            c = c - np.linalg.inv(J(zs, c).T @ J(zs, c)) @ J(zs, c).T @ r(zs, c)
        except np.linalg.LinAlgError:
            print("ERROR")
            break
        err = np.linalg.norm(c_old - c) / np.linalg.norm(c)
        i += 1
    return c


def main():
    c_init = np.array(
        [
            4.107059111542203 * 8.28721201814563,
            4.107059111542203 * -23.595886519098837,
            4.107059111542203 * 17.300387312530933,
            -2.9478499167379106,
            0.5448431082926601,
            8.28721201814563,
            -23.595886519098837,
            17.300387312530933,
            0,
            0,
            0,
            4.107059111542203 * 3.9486908534822946,
            -2.9478499167379106 * 3.9486908534822946,
            0.5448431082926601 * 3.9486908534822946,
            -2.908902115962949,
            0.5518191394370137,
            0,
            0,
            0,
            0,
            0,
            0,
            3.9486908534822946 * 3.3184196573706015,
            -2.908902115962949 * 3.3184196573706015,
            0.5518191394370137 * 3.3184196573706015,
            -2.488488024314874,
            0.51004894012372,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            3.3184196573706015 * 2.300652019954817,
            -2.488488024314874 * 2.300652019954817,
            0.51004894012372 * 2.300652019954817,
            -1.6689039845747493,
            0.4188073119525673,
        ]
    ).T
    l = 0.001
    zs = chebyshev(1000, l, 1, alpha=2)
    c = gn(5e-5, 100, zs, c_init)
    x = np.linspace(0, 1, 1000)
    print(f"g(l) = {g(l, c)}")
    plt.plot(x, g(x, c))
    plt.show()


if __name__ == "__main__":
    main()
