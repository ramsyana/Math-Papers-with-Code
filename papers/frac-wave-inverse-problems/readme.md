# Python Implementation: Fractional Wave Equation Inverse Problems

This directory contains a Python implementation for solving direct and inverse source problems associated with a 1D time-fractional wave equation, based on the work by D.K. Durdiev: [https://arxiv.org/abs/2503.17404v1](https://arxiv.org/abs/2503.17404v1)

\[
\partial_t^\alpha u - L u = f(t) h(x), \quad 1 < \alpha < 2
\]

where \( L = -\frac{d^2}{dx^2} \) is the negative Laplacian with homogeneous Dirichlet boundary conditions on the interval \( [0, L] \).

**Source Paper:**  
Durdiev, D. K. (2024). *Inverse Source Problems for a Multidimensional Time-Fractional Wave Equation with Integral Overdetermination Conditions*. [arXiv:2503.17404v1](https://arxiv.org/abs/2503.17404v1)

## Overview

The code provides tools to:

1. **Solve the Direct Problem (DP)**: Compute the solution \( u(t, x) \) given the source components \( f(t) \), \( h(x) \), initial conditions \( u(0,x) = \phi(x) \), \( u_t(0,x) = \psi(x) \), and boundary conditions.
2. **Solve Inverse Problem 1 (IP1)**: Recover the time-dependent source component \( f(t) \) given \( h(x) \), initial/boundary conditions, and the spatial integral overdetermination data \( g(t) = \int_0^L h(x) u(t,x) \, dx \).
3. **Solve Inverse Problem 2 (IP2)**: Recover the space-dependent source component \( h(x) \) given \( f(t) \), initial/boundary conditions, and the time-averaged velocity data \( \omega(x) = \int_0^T f(t) \partial_t u(t,x) \, dt \).

## Features

- **Direct Problem Solver**: Uses the Fourier method with eigenfunction expansion and Mittag-Leffler functions for time-dependent coefficients.
- **Inverse Problem 1 Solver**: Implements the second-kind Volterra integral equation approach derived from the overdetermination condition, solved numerically using the Product Trapezoidal Rule.
- **Inverse Problem 2 Solver**: Provides an optimization framework using `scipy.optimize.minimize` to find the Fourier coefficients of \( h(x) \) by minimizing a least-squares cost function with Tikhonov regularization.
- **Fractional Derivative Approximation**: Uses the L1 scheme for approximating the Caputo fractional derivative \( \partial_t^{\alpha-1} \) required in the IP1 solver.
- **Mittag-Leffler Function**: Custom implementation with caching (`functools.lru_cache`) for efficient repeated evaluations.
- **Robust Error Handling**: Comprehensive error checking, NaN detection, and warning systems throughout the solvers.
- **Testing Framework**: Includes `test_solver.py` for verifying the implementation with defined test functions and detailed error metrics.
- **Visualization**: Generates plots comparing true vs. estimated functions for IP1 and IP2, and contour plots for the DP solution.

## File Structure

```
/papers/frac-wave-inverse-problems/
├── /python/
│   ├── main_solver.py        # Main implementation with all solvers
│   ├── test_solver.py        # Test framework for all solvers
│   └── requirements.txt      # Python dependencies
└── README.md                 # Readme file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ramsyana/math-papers-with-code.git
   cd papers/frac-wave-inverse-problems/python
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   deactivate  # To exit the virtual environment

   # Or using conda
   # conda create --name frac_env python=3.9 -y
   # conda activate frac_env
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the main solver script**:
   This script executes the DP, IP1, and IP2 using predefined example functions and parameters set within the script.
   ```bash
   python main_solver.py
   ```
   Output includes console logs of the process, error metrics (L2 relative, Max absolute), and plots showing the DP solution and comparisons for IP1 and IP2 results.

2. **Run the test suite**:
   This script executes predefined test cases for DP, IP1, and IP2, reporting errors and showing comparison plots.
   ```bash
   python test_solver.py
   ```

## Configuration

Key parameters can be adjusted directly within `main_solver.py` and `test_solver.py`:

Key parameters can be adjusted in both main_solver.py and test_solver.py :

### Main Parameters
- ALPHA : Fractional order ((1 < \alpha < 2))
- L_DOMAIN : Length of the spatial domain ( [0, L] )
- T_FINAL : Final time ( T )
- N_TERMS : Number of Fourier series terms used in solvers
- NX , NT : Number of spatial and temporal grid points

### Numerical Control Parameters
- SERIES_TOL : Tolerance for early stopping of infinite series summations
- ML_MAX_TERMS , ML_SERIES_TOL : Controls for Mittag-Leffler function computation
- DP_QUAD_LIMIT , DP_QUAD_EPSABS , DP_QUAD_EPSREL : Integration controls for the DP solver
- IP2_REG_LAMBDA : Tikhonov regularization parameter for IP2 optimization
- IP2_MAX_ITER : Maximum iterations for IP2 optimizer

## Implementation Details
### Direct Problem
The solution is obtained via eigenfunction expansion:
[
u(t,x) = \sum_{n=1}^{\infty} u_n(t) X_n(x)
]
where ( X_n(x) ) are the eigenfunctions of ( L ) (normalized sine functions for the 1D case), and ( u_n(t) ) are time-dependent coefficients computed using Mittag-Leffler functions.

### Inverse Problem 1
The time component ( f(t) ) is recovered by solving a second-kind Volterra integral equation:
[
f(t) = G(t) + \int_0^t K(t,s) f(s) ds
]
where ( G(t) ) and ( K(t,s) ) are computed from the overdetermination data ( g(t) ) and the known spatial component ( h(x) ).

### Inverse Problem 2
The space component ( h(x) ) is recovered by minimizing:
[
\min_{h} \left| \omega(x) - \int_0^T f(t) \partial_t u(t,x) dt \right|^2 + \lambda |h|^2
]
where ( \omega(x) ) is the overdetermination data, and ( \lambda ) is the regularization parameter.

## Limitations and Considerations

- Numerical Stability : The Mittag-Leffler function implementation uses series summation which can be unstable for certain parameter ranges. Consider using specialized libraries for production use.

- Computational Cost : The DP solver can be computationally intensive for large N_TERMS or fine grids due to repeated Mittag-Leffler function evaluations.

- IP2 Convergence : The optimization for IP2 uses Nelder-Mead by default, which may converge slowly or to local minima. Consider alternative optimization methods for complex problems.

- Fractional Derivative Accuracy : The L1 scheme for fractional derivatives has accuracy ( O(\Delta t^{2-\gamma}) ) where ( \gamma = \alpha-1 ).

- Memory Usage : The caching of Mittag-Leffler function values can consume significant memory for long simulations.

## References

- [D.K. Durdiev, *Inverse Source Problems for a Multidimensional Time-Fractional Wave Equation with Integral Overdetermination Conditions*](https://arxiv.org/abs/2503.17404v1)

**Abstract**: In this paper, we consider two linear inverse problems for the time-fractional wave equation, assuming that its right-hand side takes the separable form \( f(t)h(x) \), where \( t \geq 0 \) and \( x \in \Omega \subset \mathbb{R}^N \). The objective is to determine the unknown function \( f(t) \) (Inverse Problem 1) and \( h(x) \) (Inverse Problem 2), given that the other function is known. 

- Podlubny, I. (1999). Fractional differential equations. Academic Press .

*Keywords*: Fractional wave equation, Dirichlet boundary condition, overdetermination condition, Mittag-Leffler function, Fourier method, existence, uniqueness.

- **L1 Scheme Reference**: Lin, Y., & Xu, C. (2007). Finite difference/spectral approximations for the time-fractional diffusion equation. *Journal of Computational Physics*, 225(2), 1533-1552.

## License

This project is licensed under the MIT License.

---

*This project is intended for research and educational purposes. Contributions via GitHub Issues or Pull Requests are welcome!*
