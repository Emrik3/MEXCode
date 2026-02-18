import matplotlib.pyplot as plt
import numpy as np


def p(c, x):
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

coeffs_list2 = [
    [5, -20, 16],
    [5, -20, 16],
    [5, -20, 16],
    [5, -20, 16],
    [5, -20, 16],
]


# TODO: Fix this to be like the other plots.
l = 0.001
u = 1
x = np.linspace(l, 2, 100)
for c in coeffs_list:
    plt.plot(
        np.linspace(l, u, 10000),
        p(c, np.linspace(l, u, 10000)),
    )

    l = p(c, l)
    u = 2 - l
    plt.plot(x, 0 * x + l)
    plt.plot(x, 0 * x + u)
    plt.legend()
plt.show()

l = 0.001
u = 1
for c in coeffs_list2:
    plt.plot(
        np.linspace(l, u, 10000),
        p(c, np.linspace(l, u, 10000)),
    )

    l = p(c, l)
    u = 2 - l
    plt.plot(x, 0 * x + l)
    plt.plot(x, 0 * x + u)
    plt.legend()
plt.show()

l = 0.001
u = 1
x = np.linspace(l, 1, 10000)

for i in range(3):
    x = p(coeffs_list[i], x)

plt.plot(np.linspace(0, 1, 10000), x, label="Old")

x = np.linspace(l, 1, 10000)

for i in range(3):
    x = p(coeffs_list2[i], x)

plt.plot(np.linspace(0, 1, 10000), x, label="New")
plt.legend()
plt.show()
