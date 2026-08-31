import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch


def SP8_coeffs(b):
    """Get coefficients for SP8 evaluation. See "Recursive expansion of the matrix step function using polynomials of degree eight" by E. Rubensson, E. Jarlebring and G. Lorentzon (2026)."""
    f4 = b[8]
    c1 = b[7] / (2 * f4)
    t2 = b[6] / f4 - c1**2
    t1 = b[5] / f4 - c1 * t2
    d0 = 0.25 * (1 - t2**2 + 4 * b[4] / f4 - 4 * c1 * t1)
    e2 = 0.5 * (t2 + 1)
    d2 = 0.5 * (t2 - 1)
    e1 = c1 * d0 + t1 * e2 - b[3] / f4
    d1 = t1 - e1
    f2 = b[2] - f4 * (d0 * e2 + d1 * e1)
    f1 = b[1] - f4 * d0 * e1
    f0 = b[0]

    r1 = d1 - 0.5 * c1 * (d2 - 0.25 * c1**2)
    r2 = d2 - 0.25 * c1**2
    r3 = e1 - d1 - 0.5 * c1
    r4 = f1 - f2 * (e1 - d1)

    return f0, f1, f2, f4, e1, d0, d1, d2, c1, r1, r2, r3, r4

@torch.compile
def SP8_eval(X, b):
    """Evaluation using coefficients from above. See "Recursive expansion of the matrix step function using polynomials of degree eight" by E. Rubensson, E. Jarlebring and G. Lorentzon (2026)."""
    f0, f1, f2, f4, e1, d0, d1, d2, c1, r1, r2, r3, r4 = b
    M1 = X @ X.mT
    I = torch.eye(M1.size(0), dtype=X.dtype, device=X.device)
    M2 = M1 @ M1
    M2 = M2 + 0.5 * c1 * M1
    M3 = M2 @ M2
    M3 = M3 + r1 * M1
    M3 = M3 + r2 * M2
    M2 = M2 + r3 * M1
    M1 = r4 * M1 + f2 * M2 + f0 * I
    M2 = M2 + M3
    M3 = M3 + d0 * I
    return (f4 * M2 @ M3 + M1) @ X


def sastre8(b):
    """See Sastre Efficient evaluation of matrix polynomials"""
    assert len(b) == 9
    assert b[-1] > 1e-8

    c4 = np.sqrt(b[8])  # Two solutions from root

    c3 = b[7] / (2 * c4)

    de2 = (b[6] - c3**2) / c4

    d1 = (b[5] - c3 * de2) / c4

    e2 = (
        ((c3 / c4) * de2 - d1)
        + np.sqrt(
            (d1 - (c3 / c4) * de2) ** 2
            + 4 * (c3 / c4) * (b[3] + (c3**2 / c4) * d1 - (c3 / c4) * b[4])
        )
    ) / (2 * (c3 / c4))  # Two solutions from root

    d2 = de2 - e2

    e0 = (b[4] - c3 * d1 - de2 * e2 + e2**2) / c4

    f2 = b[2]
    f1 = b[1]
    f0 = b[0]

    return [f0, f1, f2, e0, e2, d1, d2, c3, c4]


def sastreEval17Scalar(X, b):
    """Evaluate degree 17 odd polynomail for scalar input."""
    f0, f1, f2, e0, e2, d1, d2, c3, c4 = b
    A = X * X
    AA = A * A

    y02 = AA * (c4 * AA + c3 * A)
    y12 = (y02 + d2 * AA + d1 * A) * (y02 + e2 * AA) + e0 * y02 + f2 * AA + f1 * A

    return f0 * X + y12 * X


@torch.compile
def eval17(X: torch.Tensor, b):
    """Evaluate odd degree 17 matrix polynomial."""
    f0, f1, f2, e0, e2, d1, d2, c3, c4 = b
    A = X @ X.mT
    AA = A @ A

    y02 = AA @ (c4 * AA + c3 * A)
    y12 = (y02 + d2 * AA + d1 * A) @ (y02 + e2 * AA) + e0 * y02 + f2 * AA + f1 * A

    return f0 * X + y12 @ X


@torch.compile
def eval9(X: torch.Tensor, b):
    """Evaluate odd degree 9 matrix polynomial."""
    A = X @ X.mT
    AA = A @ A
    return b[0] * X + (b[1] * A + b[2] * AA + AA @ (b[3] * A + b[4] * AA)) @ X


@torch.compile
def eval5(X: torch.tensor, b):
    """Evaluate odd degree 5 matrix polynomial."""
    A = X @ X.mT
    AA = A @ A
    return b[0] * X + (b[1] * A + b[2] * AA) @ X


@torch.compile
def eval3(X: torch.tensor, b):
    """Evaluate odd degree 3 matrix polynomial."""
    return b[0] * X + b[1] * (X @ X.mT) @ X


def p(c, x):
    out = 0
    for i in range(len(c)):
        out += c[i] * x ** (2 * i + 1)
    return out


def main():
    pass


if __name__ == "__main__":
    main()
