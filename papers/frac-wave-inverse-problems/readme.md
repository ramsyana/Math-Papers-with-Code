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

- **Direct Problem Solver**: Uses the Fourier method with eigenfunction expansion.
- **Inverse Problem 1 Solver**: Implements the second-kind Volterra integral equation approach derived from the overdetermination condition, solved numerically using the Product Trapezoidal Rule.
- **Inverse Problem 2 Solver**: Provides a basic framework using optimization (`scipy.optimize.minimize` with Nelder-Mead) to find the Fourier coefficients of \( h(x) \) by minimizing a least-squares cost function with optional Tikhonov regularization.
- **Fractional Derivative Approximation**: Uses the L1 scheme for approximating the Caputo fractional derivative \( \partial_t^{\alpha-1} \) required in the IP1 solver.
- **Mittag-Leffler Function**: Utilizes the efficient `mittaglet` library with caching (`functools.lru_cache`) for repeated evaluations.
- **Testing Suite**: Includes `test_cases.py` for verifying the implementation of DP, IP1, and IP2 with defined test functions and error metrics.
- **Visualization**: Generates plots comparing true vs. estimated functions for IP1 and IP2, and contour plots for the DP solution.

## File Structure

```
/papers/frac-wave-inverse-problems/
├── /python/main_solver.py        # Main script with solver implementations
├── /python/tests/test_cases.py   # Script for running test cases
├── /python/tests/requirements.txt # Python dependencies
└── README.md                     # This file
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
   python tests/test_cases.py
   ```

## Configuration

Key parameters can be adjusted directly within `main_solver.py` and `test_cases.py`:

- `ALPHA`: Fractional order (\(1 < \alpha < 2\)).
- `L_DOMAIN`: Length of the spatial domain \( [0, L] \).
- `T_FINAL`: Final time \( T \).
- `N_TERMS`: Number of Fourier series terms used in final solvers.
- `NX`, `NT`: Number of spatial and temporal grid points. Higher values increase accuracy but also computation time.
- `SERIES_TOL`: Tolerance for early stopping of infinite series summations.
- `IP2_REG_LAMBDA`: Tikhonov regularization parameter for IP2 optimization (helps stabilize the inverse problem).
- `IP2_N_H_OPTIMIZE`: Number of Fourier coefficients of \( h(x) \) to optimize for in IP2.
- `IP2_DP_N_TERMS_OPTIM`: Number of terms used for the internal DP solver during IP2 optimization (can be lower than `N_TERMS` for speed).

## Implementation Details

- **Direct Problem**: Solution obtained via eigenfunction expansion (sine series for \( L = -d^2/dx^2 \)) and solving the resulting fractional ODEs for time-dependent coefficients using Mittag-Leffler functions (Eq. 15 in the paper).
- **Fractional Derivative (IP1)**: The term \( G_0(t) = \partial_t^{\alpha-1} G'(t) \) is computed using the L1 scheme, suitable for \( 0 < \alpha-1 < 1 \).
- **Volterra Equation (IP1)**: The second-kind equation (Eq. 29) is solved step-by-step using the Product Trapezoidal Rule.
- **Optimization (IP2)**: A least-squares cost function comparing the computed \( \int f(t) u_t dt \) with the target \( \omega(x) \) is minimized using `scipy.optimize.minimize`. Regularization is included.

## Limitations

- **1D Spatial Domain**: The current implementation is specific to \( L = -d^2/dx^2 \) on a 1D interval \( [0, L] \). Extension to higher dimensions or different operators \( L \) would require implementing appropriate eigensolvers or different numerical methods (e.g., finite differences/elements).
- **Uniform Time Grid**: Assumes a uniform time grid `dt`. The L1 scheme and Product Trapezoidal Rule used are implemented for uniform grids. Non-uniform grids (often beneficial near \( t=0 \) for fractional problems) would require adapting these numerical methods.
- **IP2 Convergence**: The optimization for IP2 uses a gradient-free method (Nelder-Mead by default), which can be slow and may converge to local minima, especially without a good initial guess or appropriate regularization.
- **Numerical Approximations**: Accuracy depends on grid sizes (`NX`, `NT`), number of terms (`N_TERMS`), and the tolerance (`SERIES_TOL`). The L1 scheme accuracy is typically \( O(\Delta t^{2-\gamma}) \).

## References

[D.K. Durdiev, *Inverse Source Problems for a Multidimensional Time-Fractional Wave Equation with Integral Overdetermination Conditions*](https://arxiv.org/abs/2503.17404v1)  
**Abstract**: In this paper, we consider two linear inverse problems for the time-fractional wave equation, assuming that its right-hand side takes the separable form \( f(t)h(x) \), where \( t \geq 0 \) and \( x \in \Omega \subset \mathbb{R}^N \). The objective is to determine the unknown function \( f(t) \) (Inverse Problem 1) and \( h(x) \) (Inverse Problem 2), given that the other function is known.  

*Keywords*: Fractional wave equation, Dirichlet boundary condition, overdetermination condition, Mittag-Leffler function, Fourier method, existence, uniqueness.

- **L1 Scheme Reference**: Lin, Y., & Xu, C. (2007). Finite difference/spectral approximations for the time-fractional diffusion equation. *Journal of Computational Physics*, 225(2), 1533-1552.

## License

This project is licensed under the MIT License.

---

*This project is intended for research and educational purposes. Contributions via GitHub Issues or Pull Requests are welcome!*


