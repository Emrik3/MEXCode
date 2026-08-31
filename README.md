# MEXCode

Code for generating the polynomial approximations and some of the plots in "Compositional Polynomial Approximation of the Matrix Sign Function for the Muon Optimizer" (E. Erikson, 2026).

Main components:

- `framework/framework.py`
  - Remez-based coefficient generation (`odd_remez`, `get_all_coeffs_different_degrees`)
  - scalar approximation experiments and plotting
  - matrix experiments (`PolarExpress`, `NewPolarExpress`)
- `framework/evalPol.py`
  - efficient evaluation kernels for odd polynomials of degree 3, 5, 9, and 17
- `framework/gnremez.py`
  - Gauss-Newton coefficient fitting and additional polynomial evaluation experiments
- `framework/matrix_testing.py`
  - matrix-level testing/benchmarking utilities for different polynomial compositions
