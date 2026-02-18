import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch


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
    f0, f1, f2, e0, e2, d1, d2, c3, c4 = b
    A = X * X
    AA = A * A

    y02 = AA * (c4 * AA + c3 * A)
    y12 = (y02 + d2 * AA + d1 * A) * (y02 + e2 * AA) + e0 * y02 + f2 * AA + f1 * A

    return f0 * X + y12 * X


@torch.compile
def eval17(X: torch.Tensor, b):
    """Make sure length of b is correct, len(b) == 9 == dof"""
    f0, f1, f2, e0, e2, d1, d2, c3, c4 = b
    A = X @ X.mT
    AA = A @ A

    y02 = AA @ (c4 * AA + c3 * A)
    y12 = (y02 + d2 * AA + d1 * A) @ (y02 + e2 * AA) + e0 * y02 + f2 * AA + f1 * A

    return f0 * X + y12 @ X


@torch.compile
def eval9(X: torch.Tensor, b):
    """Make sure length of b is correct, len(b) == 5 == dof"""
    A = X @ X.mT
    AA = A @ A
    return b[0] * X + (b[1] * A + b[2] * AA + AA @ (b[3] * A + b[4] * AA)) @ X


@torch.compile
def eval5(X: torch.tensor, b):
    """Make sure length of b is correct, len(b) == 3 == dof"""
    A = X @ X.mT
    AA = A @ A
    return b[0] * X + (b[1] * A + b[2] * AA) @ X


@torch.compile
def eval3(X: torch.tensor, b):
    """Make sure length of b is correct, len(b) == 2 == dof"""
    return b[0] * X + b[1] * (X @ X.mT) @ X


def testSastre():
    # Given form new polar degree 17 three times
    c = np.array(
        [
            np.array(
                [
                    2.59913611e01,
                    -7.92828499e02,
                    9.80249362e03,
                    -5.80808762e04,
                    1.87004843e05,
                    -3.45623379e05,
                    3.66229363e05,
                    -2.06759009e05,
                    4.81953758e04,
                ]
            ),
            np.array(
                [
                    1.13780641e01,
                    -8.90676835e01,
                    2.82604469e02,
                    -4.29711692e02,
                    3.55057277e02,
                    -1.68403157e02,
                    4.57932885e01,
                    -6.63459879e00,
                    3.96878112e-01,
                ]
            ),
            np.array(
                [
                    4.81741599,
                    -21.93954932,
                    62.52382524,
                    -100.98701338,
                    96.18014484,
                    -55.02848716,
                    18.55832927,
                    -3.39555863,
                    0.25972273,
                ]
            ),
        ]
    )

    x = np.linspace(0, 1, 1000)

    for i in range(3):
        b = sastre8(c[i])
        x = sastreEval17Scalar(x, b)

    x_plt = np.linspace(0, 1, 1000)
    plt.plot(x_plt, x)
    plt.show()


def main():
    testSastre()


if __name__ == "__main__":
    main()
