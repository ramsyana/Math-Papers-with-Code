import numpy as np
from scipy.special import gamma
# Fix the import for mpmath
from mpmath import mp
print("Successfully imported mpmath:", mp)
from scipy import linalg
import matplotlib.pyplot as plt
from functools import lru_cache
from scipy.optimize import minimize

# Define our own implementation of the Mittag-Leffler function
def mittag_leffler(alpha, beta, z, dps=15, max_terms=1000, tolerance=1e-15):
    """
    Implementation of the Mittag-Leffler function using mpmath's core functionality.
    
    Parameters:
    -----------
    alpha : float
        First parameter of the Mittag-Leffler function
    beta : float
        Second parameter of the Mittag-Leffler function
    z : float or complex
        Argument of the Mittag-Leffler function
    dps : int
        Decimal places of precision (default: 15)
    max_terms : int
        Maximum number of series terms (default: 1000)
    tolerance : float
        Convergence tolerance (default: 1e-15)
        
    Returns:
    --------
    float
        Value of the Mittag-Leffler function
    """
    # Set precision
    mp.dps = dps
    
    # Convert inputs to mpmath precision
    z_mp = mp.mpf(z)
    alpha_mp = mp.mpf(alpha)
    beta_mp = mp.mpf(beta)
    
    # Convert tolerance to mpmath precision
    tolerance_mp = mp.mpf(str(tolerance))
    
    # Initialize sum
    result = mp.mpf(0)
    term = mp.mpf(1) / mp.gamma(beta_mp)  # First term (k=0)
    result += term
    
    # Compute the series
    for k in range(1, max_terms):
        term = (z_mp**k) / mp.gamma(alpha_mp*k + beta_mp)
        result += term
        
        # Check for convergence
        if abs(term) < tolerance_mp:
            break
    
    # Convert back to Python float
    return float(result)

class ImprovedFractionalPDEInverseSolver:
    """
    Solves the fractional PDE:
    D_t^α u - Lu = f(t)h(x), with initial conditions u(0,x) = phi(x) and u_t(0,x) = psi(x)
    
    Where:
    - L is a general elliptic operator (default is Laplacian Δ)
    - Supports 1D and 2D spatial domains
    - 1 < α < 2 requires two initial conditions (wave-like behavior)
    """
    def __init__(self, M=100, K=100, T=1.0, alpha=1.5, Nmax=10, operator_L=None):
        # Store grid parameters
        self.M = M
        self.K = K
        self.T = T
        self.alpha = alpha
        self.Nmax = Nmax
        
        # Spatial grid (1D or 2D)
        self.x = np.linspace(0, np.pi, M)
        self.dx = self.x[1] - self.x[0] if M > 1 else 0
        
        # Temporal grid
        self.t = np.linspace(0, T, K)
        self.dt = self.t[1] - self.t[0] if K > 1 else 0

        # Operator configuration
        if operator_L is None:
            # Default to Laplacian eigenfunctions
            self.eigenvalues = np.array([n**2 for n in range(1, Nmax+1)])
            self.eigenfunctions = np.array([np.sin(n*self.x) for n in range(1, Nmax+1)])
        else:
            # Allow custom operator implementation
            self.eigenvalues, self.eigenfunctions = operator_L

    def solve_direct_problem(self, phi, psi, h, f):
        """
        Solve direct problem with TWO initial conditions (for 1 < α < 2)
        Implements:
        u(0,x) = phi(x)
        u_t(0,x) = psi(x)
        """
        # Combined initial conditions
        init_condition = phi(self.x) + psi(self.x)*self.t[0]
        
        # Store both initial conditions for inverse problems
        self.phi = phi
        self.psi = psi
        
        # Modify source term to be multiplicative
        source_term = np.outer(f, h)  # f(t)h(x)
        
        # ... (modified solution implementation would go here)