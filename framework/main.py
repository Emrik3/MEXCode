import time
from itertools import repeat
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from evalPol import eval3, eval5, eval9, eval17, sastre8
from gnremez import composite_gnremez, p, plot_pol

from framework import get_all_coeffs_different_degrees, odd_remez

coeffs_dir = "coeffs/"
pe_coeffs = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]


def create_file(i, m, l, u):
    print(f"Number of mults: {m}")
    if m == 1:
        c = pe_coeffs[i]
        np.save(coeffs_dir + "coeffs" + str(i) + str(m) + ".npy", c)
        coeffsPolset = np.array(
            [1, 1, 1, 1, c[0], c[1] / 2, c[2]]
        )  # TODO: This is wrong, fix
        np.save(coeffs_dir + "coeffsPolset" + str(i) + str(m) + ".npy", coeffsPolset)
        x = np.linspace(l, u)
        plot_pol(coeffsPolset, m, l, u)
        plt.show()
        create_file(i, m + 1, l, u)

        return
    file_path = Path(coeffs_dir + "coeffsPolset" + str(i) + str(m - 1) + ".npy")
    if not file_path.exists():
        create_file(i, m - 1, l, u)
    file_path_og = Path(coeffs_dir + "coeffs" + str(i) + str(m) + ".npy")
    if not file_path_og.exists():
        c = odd_remez(2**m, l, u, 1e-8)
        np.save(coeffs_dir + "coeffs" + str(i) + str(m) + ".npy", c)
    c = composite_gnremez(m, i, l, u)
    np.save(coeffs_dir + "coeffsPolset" + str(i) + str(m) + ".npy", c)


def get_all_polset(m_list):
    l = 0.001
    u = 1
    for i, m in enumerate(m_list):
        create_file(i, m, l, u)
        c = np.load(coeffs_dir + "coeffsPolset" + str(i) + str(m) + ".npy")
        l = p(c, m, l)
        u = 2 - l


def main():
    get_all_polset([3, 3, 3])


if __name__ == "__main__":
    main()
