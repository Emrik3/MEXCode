import math


def compute_term(q, xi_x):
    """
    Computes (4q+3)!! / (2^(2q+1) * (2q+2)! * xi_x^((4q+3)/2))

    Parameters:
        q (int): non-negative integer
        xi_x (float): positive number

    Returns:
        float: computed value
    """
    # Double factorial (4q+3)!!
    double_fact = 1
    for k in range(2 * q + 1, 0, -2):
        double_fact *= k

    # Denominator: 2^(2q+1) * (2q+2)!
    denom = (2 ** (q)) * math.factorial(q + 1)

    # xi_x^( (4q+3)/2 )
    xi_term = xi_x ** (q + 1 / 2)

    # Note that double_fact \approx denom. and behaves well.
    # xi_term is the problem...
    return double_fact / (
        denom * xi_term * 2**q
    )  # Need some bound for the point disrubution as well, remember they are squared.


def main():
    q = 2
    xi_x = 1  # Should this be l?  # At least one distinct zero? Where? Also better bound on the product possible giver equioscillation, probably yes?
    # The error in the last approximation is what matters, Then l is large, maybe try to combine to derive what happens to l.
    # In the end, want a formula saying that a combination of degrees leads to something, that is first ones doing some thing to l and satisfying equi and then
    #  the error of the last approx...
    #
    # It seems what we want from these error bounds is that it pulls quickly up and then high degree in the end
    # Need some analysis on how to pull values up quickly, is equioscilation even the answer in the first iterations?
    # Yes probably because dont want any other values to land outside of interval, that is that values like 0.8 can go to 3.
    # But if 0.001 goes to higher error than other points?
    result = compute_term(q, xi_x)
    print(result)


if __name__ == "__main__":
    main()
