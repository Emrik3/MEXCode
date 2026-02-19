import matplotlib.pyplot as plt
import numpy as np


def p(z, c):
    return c[0] * z + c[1] * z**3 + c[2] * z**5


def k(z, c):
    return c[0] * z + c[1] * z**3 + c[2] * z**5


def g(z, c):
    q1 = k(z, c[0:3]) + c[3] * p(z, c[5:8]) ** 3 + c[4] * p(z, c[5:8]) ** 5
    q2 = k(z, c[8:11]) + k(p(z, c[5:8]), c[11:14]) + c[14] * q1**3 + c[15] * q1**5
    q3 = (
        k(z, c[16:19])
        + k(p(z, c[5:8]), c[19:22])
        + k(q1, c[22:25])
        + c[25] * q2**3
        + c[26] * q2**5
    )
    q4 = (
        k(z, c[27:30])
        + k(p(z, c[5:8]), c[30:33])
        + k(q1, c[33:36])
        + k(q2, c[36:39])
        + c[39] * q3**3
        + c[40] * q3**5
    )
    return q4


def r(zs, c):
    """
    Added penalty term at the start to make it grow quickley, but it is hard to do gauss newton given that we want bounds on maximum error.
    """
    return np.array(
        np.concatenate(
            [[(g(z, c) - 1) for z in zs]]
        )  # Add the lambda term to make it important to be good at l=0.001
    ).T


def J(zs, c):
    h = 1e-10

    return np.array(
        [
            [
                (g(z, c + h * np.eye(len(c))[i]) - g(z, c - h * np.eye(len(c))[i]))
                / (2 * h)
                for i in range(len(c))
            ]
            for z in zs
        ]
    )


def chebyshev(n, l, u, alpha=2.0):
    """
    Includes the endpoints l and u.
    """

    k = np.arange(n)
    x = np.cos((k / (n - 1)) ** alpha * np.pi)
    return (l + u) / 2 + (u - l) / 2 * x


def gn(tol, max_iter, zs, c):
    """
    Very sensitive to starting guess, should have starting guess be the value that is from remez maybe.
    """
    err = 1
    i = 0
    lam = 1e-2
    while err > tol and i < max_iter:
        print(err)
        c_old = c
        try:
            jacobian = J(zs, c)
            cDiff = -np.linalg.solve(
                jacobian.T @ jacobian + lam * np.eye(len(c)), jacobian.T @ r(zs, c)
            )
        except np.linalg.LinAlgError:
            print("ERROR")
            break
        c = cDiff + c_old
        err = max(abs(g(zs, c) - 1))
        i += 1
    return c


def findGN():
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
    zs = chebyshev(2000, l, 1, alpha=2)
    h = 1e-5
    # If you want more points around l.
    zs = np.concatenate([zs, np.linspace(0.001 - h, 0.001 + 10 * h, 1000)])

    c = gn(0.1, 10, zs, c_init)
    np.save("coeffs.npy", c)


def plot():

    c = np.load("coeffs.npy")
    l = 0.001
    x = np.linspace(0, 1, 10000)
    print(f"g(l) = {g(l, c)}")
    plt.plot(x, g(x, c), label="Gauss-Newton")

    def pPE(c, x):
        out = 0
        for i in range(len(c)):
            out += c[i] * x ** (2 * i + 1)
        return out

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

    x = np.linspace(0, 1, 10000)
    for i in range(5):
        x = pPE(coeffs_list[i], x)
        l = pPE(coeffs_list[i], l)
    print(f"PE(l) = {l}")

    plt.plot(np.linspace(0, 1, 10000), x, label="PE")
    plt.legend()
    plt.show()


def main():
    findGN()
    plot()


if __name__ == "__main__":
    main()
