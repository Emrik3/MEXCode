import numpy as np


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


def odd_remez(q, l, u, tolRemez, tolNewton, alpha=1.0):
    """
    TODO: It seems that after just a few guesses the coefficients stio changing at all.
    Need some way of getting out of local minimum
    Do a bit of math about this and find better source on algorithm...
    """
    x = np.zeros(q + 2)
    f = np.ones(q + 2)
    n = q + 2

    # Calculate initial guess of points as Chebyshev points
    for i in range(n):
        x[i] = 0.5 * (l + u) + 0.5 * (u - l) * np.cos((2 * i + 1) * np.pi / (2 * n))
    err = 1000.0
    c = None

    while err > tolRemez:
        print(f"x: {x}")
        A = np.zeros((q + 2, q + 2))
        for j in range(q + 2):
            for i in range(q + 1):
                A[j, i] = x[j] ** (2 * i + 1)
        A[:, -1] = (-1) ** np.arange(q + 2)

        print(A)

        c = np.linalg.solve(A, f)
        # c = [8.28721201814563, -23.595886519098837, 17.300387312530933, 1] # Optimal in PE
        # Compute all extremes of error function, use newton here?
        x_new = []
        coeffs_for_roots = derivative_coeffs(c[:-1])
        root_guess = np.roots(coeffs_for_roots)
        print(f"coeffs: {c[:-1]}")
        candidates = []
        for r in root_guess:
            if np.isreal(r):
                r = r.real
                print(r)
                if r > 0:
                    candidates.append(r)

        for guess in candidates:  # If they are too close we might have problems
            x_new.append(newton_pol(guess, c[:-1], tolNewton))

        # Always include endpoints
        x_new = [l] + x_new + [u]

        # Sort for consistency
        x_new = np.array(sorted(x_new))

        if len(x_new) != q + 2:
            raise ValueError(f"Expected {q + 2} extremal points, got {len(x_new)}")

        print("Extremal points:", x_new)
        x = x_new

        # Make sure all unique points were found
        if len(x_new) == len(set(x_new)):
            print("Unique points found")
            x = np.array(x_new)
        else:
            print("Implement way to find all points...")

        errList = []
        for point in x:
            errList.append(np.abs(p(c, point) - 1))
        err = np.max(errList) - alpha * np.min(errList)
        print(f"Errors:  {errList}")
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
    err = 100.0
    while err > tol:
        x_new = x - pp(c, x) / ppp(c, x)
        err = np.abs(x_new - x)
        x = x_new
    return x


def main():
    c = odd_remez(2, 0.001, 1, 1e-10, 1e-10, 1.05)
    print(c)


if __name__ == "__main__":
    main()
