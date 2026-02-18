import math

import matplotlib.pyplot as plt
import numpy as np

coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    # (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    # (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    # (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]

coeffs_list_no_cushion = [
    [8.4703288, -25.10807471, 18.6292756],
    [4.18283418, -3.10870111, 0.58060668],
    [3.96185728, -2.95406375, 0.56297612],
    [3.28658622, -2.46472013, 0.50735769],
    [2.27374999, -1.64466037, 0.41619093],
]


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


def error_comp(c, l, u, x):
    n = len(c[0]) * 2 + 1
    return (
        np.sqrt(
            np.abs(
                (l - x) * (u - x) * (pp(c[0], x) * pp(c[1], p(c[0], x))) ** 2
                + n**2 * p(c[1], p(c[0], x)) ** 2
            )
        )
        / n
    )


def error(c, l, u, x):
    n = len(c) * 2 + 1
    return np.sqrt((l - x) * (u - x) * pp(c, x) ** 2 + n**2 * (p(c, x)) ** 2) / n


def plot_comp():
    c = coeffs_list
    l = 0.001
    u = 1
    x = np.linspace(l, u, 1000)
    plt.plot(x, np.abs(error_comp(c, l, u, x) - 1), label="Approx error")
    plt.plot(x, np.abs(p(c[1], p(c[0], x)) - 1), label="Actual error")
    plt.legend()
    plt.show()


def plot_non_comp():
    c = coeffs_list[0]
    l = 0.001
    u = 1
    x = np.linspace(l, u, 1000)
    plt.plot(x, np.abs(error(c, l, u, x) - 1), label="Approx error")
    plt.plot(x, np.abs(p(c, x) - 1), label="Actual error")
    plt.legend()
    plt.show()


def main():
    plot_non_comp()


if __name__ == "__main__":
    main()
