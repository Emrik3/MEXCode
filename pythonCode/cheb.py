import sympy as sp


def chebApprox(n, l, u):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return 2 * x * chebApprox(n - 1, l, u) + chebApprox(n - 2, l, u)


print(chebApprox(5, 0.001, 1))
