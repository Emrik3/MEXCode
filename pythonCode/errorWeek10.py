import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def E(l, u):

    x = np.sqrt((l**2 + l * u + u**2) / 3)  # This is the maximizer of the error
    return (l - x) * (x - u) * (l + x + u) / ((l + x) * (x + u) * (l - x + u))


def E3(l, u):

    T = 1
    xs = []

    for i in range(T):
        l = 1 - max5(l, u)
        u = 2 - l

        xs.append(l)

        print(f"l = {l}")

        if len(xs) >= 3:
            e_k1 = abs(xs[-1] - 1)
            e_k = abs(xs[-2] - 1)
            e_km1 = abs(xs[-3] - 1)

            p = np.log(e_k1 / e_k) / np.log(e_k / e_km1)

            print(f"estimated order p ≈ {p}")


def symbolicDerivative():
    l = sp.symbols("l")

    u = 2 - l

    # symbolic x
    x = sp.sqrt((l**2 + l * u + u**2) / 3)

    # symbolic E
    E = ((l - x) * (x - u) * (l + x + u)) / ((l + x) * (x + u) * (l - x + u))

    # iteration map
    F = 1 - E

    # derivatives
    F1 = sp.diff(F, l)
    F2 = sp.diff(F1, l)
    F3 = sp.diff(F2, l)

    # evaluate at fixed point l=1
    print("F'(1) =", sp.simplify(F1.subs(l, 1)))
    print("F''(1) =", sp.simplify(F2.subs(l, 1)))
    print("F'''(1) =", sp.simplify(F3.subs(l, 1)))


def pPE(c, x):
    out = 0
    for i in range(len(c)):
        out += c[i] * x ** (2 * i + 1)
    return out


def intersects(n, T):

    coeffs_list_no_cushion = [
        [8.4703288, -25.10807471, 18.6292756],
        [4.18283418, -3.10870111, 0.58060668],
        [3.96185728, -2.95406375, 0.56297612],
        [3.28658622, -2.46472013, 0.50735769],
        [2.27374999, -1.64466037, 0.41619093],
    ]

    x = np.linspace(0, 1, n)

    for i in range(T):
        x = pPE(coeffs_list_no_cushion[i], x)

    xs = np.linspace(0, 1, n)
    ys = x

    # Compute difference from y = 1
    diff = ys - 1

    # Detect sign changes
    sign_changes = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]

    intersections = []

    for idx in sign_changes:
        x0, x1 = xs[idx], xs[idx + 1]
        y0, y1 = diff[idx], diff[idx + 1]

        # Linear interpolation
        root = x0 - y0 * (x1 - x0) / (y1 - y0)
        intersections.append(root)

    # Optional: include exact hits
    exact_hits = xs[np.isclose(ys, 1)]
    intersections.extend(exact_hits)

    return np.array(intersections)


def product(x, l, u):
    zeros = intersects(10, 1)
    out = 1
    print(zeros)
    for point in zeros:
        out *= x - point
    return out


def lagrangeBound(q, l):
    xi_x = l  # This is the problem to getting a large error
    # I mean the (1-l^2) part is also not really great.
    double_fact = 1
    for k in range(2 * q + 1, 0, -2):
        double_fact *= k

    # Denominator: 2^(2q+1) * (2q+2)!
    denom = (2**q) * math.factorial(q + 1)

    # xi_x^( (4q+3)/2 )
    xi_term = xi_x ** (q + 1 / 2)

    prod = product(l, l, 1)

    # Note that double_fact \approx denom. and behaves well.
    # xi_term is the problem...
    return l * prod * double_fact / (denom * xi_term)


def bernsteinBound(q, l):
    return (
        2 * (1 - np.sqrt(l)) * ((1 - np.sqrt(l)) / (1 + np.sqrt(l))) ** q
    ) / np.sqrt(l)


def tanhBound(n, l, T):
    return math.tanh((np.sqrt(n)) ** T * math.atanh(np.sqrt(l))) ** 2


# When l is around 0.7 we can start using this bound, otherwise it is too loose of a bound
def whenDoesLagrangeWork(l, q):
    x = np.linspace(l, 1, 1000)
    double_fact = 1
    for k in range(2 * q + 1, 0, -2):
        double_fact *= k

    # Denominator: 2^(2q+1) * (2q+2)!
    denom = (2**q) * math.factorial(q + 1)

    new_l = np.abs(
        1 - double_fact * (2 - x - x**2) ** (q + 1) / (denom * x ** (q + 1 / 2))
    )
    for i in range(len(x)):
        if new_l[i] < x[i]:
            print(new_l[i])
            print(x[i])
            break


# Too loose at small l
def chebApproxError(l, q):
    double_fact = 1
    for k in range(2 * q + 1, 0, -2):
        double_fact *= k

    # Denominator: 2^(2q+1) * (2q+2)!
    denom = (4**q) * math.factorial(q + 1) * l ** (q + 1 / 2)

    return double_fact * (1 - l) ** (q + 1) / denom


# ----- Complete homogeneous symmetric polynomial h_k -----
def h_k(xs, k):
    if k == 0:
        return 1
    total = 0
    n = len(xs)
    for comb in itertools.combinations_with_replacement(range(n), k):
        prod = 1
        for i in comb:
            prod *= xs[i]
        total += prod
    return total


# ----- Schur polynomial via Jacobi–Trudi -----
def schur(xs, partition):
    l = len(partition)
    M = np.zeros((l, l), dtype=float)

    for i in range(l):
        for j in range(l):
            k = partition[i] - i + j
            if k >= 0:
                M[i, j] = h_k(xs, k)
            else:
                M[i, j] = 0

    return np.linalg.det(M)


# ----- Vandermonde product -----
def vandermonde(xs):
    prod = 1
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            prod *= xs[j] - xs[i]
    return prod


# ----- Denominator term -----
def denominator(xs):
    n = len(xs)
    total = 0

    for i in range(n):
        prod1 = 1
        for j in range(n):
            if j != i:
                prod1 *= xs[j]

        prod2 = 1
        idx = [j for j in range(n) if j != i]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                j, k = idx[a], idx[b]
                prod2 *= (xs[k] + xs[j]) * (xs[k] - xs[j])

        total += prod1 * prod2

    return total


# ----- Main error function -----
def error_value(xs, q):
    xs = list(xs)

    if len(xs) != q + 2:
        raise ValueError("Need q+2 x values")

    partition = list(range(q, 0, -1))  # (q, q-1, ..., 1)

    xx = np.linspace(0.001, 2)
    out = []
    for x in xx:
        out.append(schur([0.001, x, x, 2], partition))
    plt.plot(xx, out)
    plt.show()

    # Will wait with the schur one until i have a good way of calculationg or approximating it.
    num = (
        vandermonde(xs) * schur(xs, partition)
    )  # The schur is now 20 if we input the value 1 only, i.e. the max, then the bound is too loose
    den = denominator(xs)

    return abs(num / den)


def V2(xs, k):
    prod = 1
    for i in range(len(xs)):
        if i == k:
            continue
        for j in range(i + 1, len(xs)):
            if j == k:
                continue
            prod *= xs[j] ** 2 - xs[i] ** 2
    return prod


def V2All(xs):
    prod = 1
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            prod *= xs[j] - xs[i]
    return prod


# ----- No schur error -----
def V(x, i):
    prod = 1
    for k in range(len(x)):
        if k == i:
            continue
        prod *= x[k] ** 2 - x[i] ** 2
    return abs(prod)


def betterErrorSimple(x, q):
    num = 0
    for i in range(0, q + 2, 2):
        num += 1 / (x[i] * V(x, i))
    den = 0
    for i in range(q + 2):
        den += 1 / (x[i] * V(x, i))

    return 2 * num / den - 1


def betterError(x, q):
    num = 0
    # Should this not be even?

    for i in range(0, q + 2, 2):
        num += V2(x, i) / x[i]
    den = 0
    for i in range(q + 2):
        den += V2(x, i) / x[i]

    return 2 * num / den - 1


def plotBetter():
    l = 0.001
    u = 1
    q = 1
    d = (1 - l) / (2 * q + 1) ** 2
    print(d)

    xplt = np.linspace(l + d, u - d)
    E = []
    for x in xplt:
        E.append(betterError([l, x, u], q))
    plt.plot(xplt, E)
    plt.show()


def proportional(x, q):
    odd = 0
    for i in range(0, q + 2, 2):
        odd += 1 / x[i]
    even = 0
    for i in range(1, q + 2, 2):
        even += 1 / x[i]
    return (odd - even) / (odd + even)


# Without the cushioning and division by 0.01 in the remez composition that gives the same error as this one.
def max5(l, u):
    q = 2
    d = (1 - l) / (2 * q + 1) ** 2

    xplt = np.linspace(l + d, u - d)
    E = []
    for x in xplt:
        for y in xplt:
            if x >= y:
                continue
            E.append(betterError([l, x, y, u], q))
            # Ok so simple and normal give the same answer but the bound is a bit too good?
            if len(E) > 5:
                if E[-1] == max(E):
                    print(x, y)
    print(1 - max(E))
    return max(E)


def error5test(x, q):
    num = V2All(x) * sum(x)

    den = 0
    for i in range(q + 2):
        den += V2(x, i) / x[i]

    return num / den


def maxProp(l, q):
    d = (1 - l) / (2 * q + 1) ** 2
    x = [l]
    for i in range(q, -1, -1):
        x.append(1 - d * i)
    odd = 0
    for i in range(0, q + 2, 2):
        odd += 1 / x[i]
    even = 0
    for i in range(1, q + 2, 2):
        even += 1 / x[i]
    return (odd - even) / (odd + even)


def main():
    max5(0.001, 1)


main()
