"""Rewritten by AI, not verified
Did not want to spend time on this since remez is very slow for degree 31, if Remez can be sped up i will spend time on this"""

def sastre15(b16, b15, b14, b13, b12, b11, b10, b9,
                     b8, b7, b6, b5, b4, b3, b2, b1, b0):
    """
    Python/SymPy rewrite of MATLAB code fragments 4.1 and 4.2 from
    Sastre & Ibanez (2021).

    Input:
        b16, ..., b0 : symbolic or numeric coefficients

    Output:
        dict with symbolic expressions for
        c2,c3,c4,c5,c6,c7,c8,d1,d2,e0,e1,f0,g0,h2,h1,h0I
    """

    # ------------------------------------------------------------------
    # Symbols
    # ------------------------------------------------------------------
    A = sp.symbols('A')

    c2, c3, c4, c5, c6, c7, c8 = sp.symbols('c2 c3 c4 c5 c6 c7 c8')
    d1, d2 = sp.symbols('d1 d2')
    e0, e1 = sp.symbols('e0 e1')
    f0, g0 = sp.symbols('f0 g0')
    h2, h1, h0I = sp.symbols('h2 h1 h0I')

    # coefficient vectors
    c = sp.Matrix([c2, c3, c4, c5, c6, c7, c8])
    b = sp.Matrix([
        b16, b15, b14, b13, b12, b11, b10, b9,
        b8, b7, b6, b5, b4, b3, b2, b1, b0
    ])

    # ------------------------------------------------------------------
    # Polynomial construction
    # ------------------------------------------------------------------
    y0 = A**2 * (sp.sqrt(c8)*A**2 + c7/(2*sp.sqrt(c8))*A)

    y1 = sum(c[i] * A**(i+2) for i in range(7))

    y2 = (
        (y1 + d2*A**2 + d1*A)
        * (y1 + e0*y0 + e1*A)
        + f0*y1 + g0*y0
        + h2*A**2 + h1*A + h0I
    )

    # ------------------------------------------------------------------
    # Match coefficients with target polynomial
    # y2 = c8^2 A^16 + sum_{i=0}^{15} b_i A^i
    # ------------------------------------------------------------------
    poly = sp.Poly(y2, A)
    coeffs = poly.all_coeffs()     # from A^16 down to A^0

    # remove leading A^16 term (equals c8^2)
    eqs = [
        coeffs[i+1] - b[i+1]
        for i in range(16)
    ]

    # ------------------------------------------------------------------
    # Elimination sequence (matches MATLAB)
    # ------------------------------------------------------------------
    c7s = sp.solve(eqs[0], c7)[0]
    eqs = [eq.subs(c7, c7s) for eq in eqs]

    c6s = sp.solve(eqs[1], c6)[0]
    eqs = [eq.subs(c6, c6s) for eq in eqs]

    c5s = sp.solve(eqs[2], c5)[0]
    eqs = [eq.subs(c5, c5s) for eq in eqs]

    e0s = sp.solve(eqs[3], e0)[0]
    eqs = [eq.subs(e0, e0s) for eq in eqs]

    c3s = sp.solve(eqs[4], c3)[0]
    eqs = [eq.subs(c3, c3s) for eq in eqs]

    d2s = sp.solve(eqs[5], d2)[0]
    eqs = [eq.subs(d2, d2s) for eq in eqs]

    d1s = sp.solve(eqs[6], d1)[0]
    eqs = [eq.subs(d1, d1s) for eq in eqs]

    f0s = sp.solve(eqs[7], f0)[0]
    eqs = [eq.subs(f0, f0s) for eq in eqs]

    c8s = sp.solve(eqs[8], c8)[0]
    eqs = [eq.subs(c8, c8s) for eq in eqs]

    # ---- fragment 4.2 ----
    c4s = sp.solve(eqs[9], c4)[0]   # first branch
    eqs = [eq.subs(c4, c4s) for eq in eqs]

    e1s = sp.solve(eqs[10], e1)[0]
    eqs = [eq.subs(e1, e1s) for eq in eqs]

    g0s = sp.solve(eqs[11], g0)[0]
    eqs = [eq.subs(g0, g0s) for eq in eqs]

    # ------------------------------------------------------------------
    # Return all coefficients
    # ------------------------------------------------------------------
    return {
        "c2": sp.solve(eqs[12], c2)[0],
        "c3": c3s,
        "c4": c4s,
        "c5": c5s,
        "c6": c6s,
        "c7": c7s,
        "c8": c8s,
        "d1": d1s,
        "d2": d2s,
        "e0": e0s,
        "e1": e1s,
        "f0": f0s,
        "g0": g0s,
        "h2": h2,
        "h1": h1,
        "h0I": h0I,
    }


def sastreEval31Scalar(X, b):
    [c2, c3, c4, c5, c6, c7, c8, d1, d2, e0, e1, f0, g0, h2, h1, h0] = b
    A = X * X
    A2 = A * A
    

    y02 = A2 * (np.sqrt(c8) * A2 + c7 / (2*np.sqrt(c8)) * A)

    # TODO: Use degree 8 eval on this
    y12 = c2 * A*A + c3 * A**3 + c4*A**4 + c5*A**5 + c6*A**6 + c7*A**7 + c8*A**8
    
    return ((y12 + d2 * A2 + d1 * A) * (y12 + e0 * y02 + e1 * A) + f0 * y12 + g0 * y02 + h2 * A2 + h1 * A + h0 ) * X
