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
    Solver for fractional PDE inverse problems.
    
    Solves the fractional PDE:
    D_t^α u - Δu = h(x) + f(t), u(0,x) = phi(x), u(t,0) = u(t,π) = 0
    
    Note: Differs from paper (2503.17404v1.pdf) form D_t^α u - Δu = f(t)h(x), which may imply 1 < α < 2.
    
    where:
    - D_t^α is the Caputo fractional derivative of order α (0 < α < 1)
    - Δ is the Laplacian operator
    - h(x) is the spatial source term
    - f(t) is the temporal source term
    
    The class can solve both direct and inverse problems:
    1. Direct problem: Given h(x), f(t), and phi(x), find u(t,x)
    2. Inverse problem 1: Given u(t,x_0), find f(t)
    3. Inverse problem 2: Given u(T,x), find h(x)
    
    The solution method uses spectral decomposition with eigenfunctions of the Laplacian
    and the Mittag-Leffler function for the time evolution.
    """
    def __init__(self, N=1, M=100, K=100, T=1.0, alpha=0.5, Nmax=10):
        """
        Initialize the solver for fractional PDE inverse problems.
        
        Parameters:
        -----------
        N : int
            Dimensionality (1 or 2)
        M : int
            Number of spatial grid points
        K : int
            Number of time grid points
        T : float
            Time horizon
        alpha : float
            Fractional order (0 < alpha < 1)
        Nmax : int
            Number of eigenmodes to use
        """
        # Input validation
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if N not in [1, 2]:
            raise ValueError("N must be 1 or 2")
        if M <= 0 or K <= 0 or Nmax <= 0:
            raise ValueError("M, K, and Nmax must be positive integers")
        if T <= 0:
            raise ValueError("T must be positive")
        
        self.N = N
        self.M = M
        self.K = K
        self.T = T
        self.alpha = alpha
        self.Nmax = Nmax
        
        # Create spatial grid (0 to π)
        self.x = np.linspace(0, np.pi, M)
        self.dx = self.x[1] - self.x[0]
        
        # Create temporal grid (0 to T)
        self.t = np.linspace(0, T, K+1)
        self.dt = self.t[1] - self.t[0]
        
        # Generate eigenfunctions and eigenvalues
        self.eigenfunctions, self.eigenvalues = self._generate_eigenfunctions()
        
        # Precompute Mittag-Leffler kernel for efficiency
        self._ml_kernel = self._precompute_ml_kernel()
        
        # Precompute weights for L1 scheme of Caputo derivative
        self._precompute_caputo_weights()
        
        # Cache for optimization
        self._cache = {}
    
    def _generate_eigenfunctions(self):
        """
        Generate eigenfunctions and eigenvalues for the Laplacian operator
        with Dirichlet boundary conditions on [0, π].
        """
        if self.N == 1:
            # 1D case: phi_n(x) = sqrt(2/π) * sin((n+1)x)
            eigenfunctions = []
            eigenvalues = []
            
            for n in range(self.Nmax):
                phi_n = np.sqrt(2/np.pi) * np.sin((n+1) * self.x)
                lambda_n = (n+1)**2
                eigenfunctions.append(phi_n)
                eigenvalues.append(lambda_n)
            
            return np.array(eigenfunctions), np.array(eigenvalues)
        
        elif self.N == 2:
            # 2D case: phi_{n,m}(x,y) = (2/π) * sin((n+1)x) * sin((m+1)y)
            # Compute meshgrid once for efficiency
            X, Y = np.meshgrid(self.x, self.x)
            eigenfunctions = np.zeros((self.Nmax * self.Nmax, self.M, self.M))
            eigenvalues = []
            idx = 0
            
            for n in range(self.Nmax):
                for m in range(self.Nmax):
                    eigenfunctions[idx] = (2/np.pi) * np.sin((n+1) * X) * np.sin((m+1) * Y)
                    eigenvalues.append((n+1)**2 + (m+1)**2)
                    idx += 1
            
            return eigenfunctions, np.array(eigenvalues)
        
        else:
            raise ValueError("Only N=1 (1D) or N=2 (2D) is supported.")
    
    @lru_cache(maxsize=128)
    def _mittag_leffler_cached(self, alpha, z, dps=15, max_terms=1000, tolerance=1e-15):
        """Cached version of Mittag-Leffler function using mpmath"""
        # Use beta=1 for standard E_alpha(z), convert mpf to float for NumPy compatibility
        return mittag_leffler(alpha, 1, z, dps=dps, max_terms=max_terms, tolerance=tolerance)
    
    def _precompute_ml_kernel(self):
        """
        Precompute the Mittag-Leffler kernel E_α(-λ_n * t^α) for each eigenvalue
        and time point for efficiency using vectorization.
        """
        # Use broadcasting to compute all z values at once
        z = -self.eigenvalues[:, None] * self.t[None, :]**self.alpha
        
        # Apply the Mittag-Leffler function to each element
        ml_kernel = np.zeros(z.shape)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                ml_kernel[i, j] = self._mittag_leffler_cached(self.alpha, z[i, j])
        
        return ml_kernel
    
    def _precompute_caputo_weights(self):
        """
        Precompute weights for the L1 scheme of Caputo fractional derivative
        using the standard formula.
        """
        K = self.K
        alpha = self.alpha
        
        # Weights for the standard L1 scheme
        self._caputo_weights = np.zeros(K+1)
        for j in range(K+1):
            if j < K:
                self._caputo_weights[j] = (j+1)**(1-alpha) - j**(1-alpha)
            else:
                # Fix for j = K: set to 0 for boundary consistency
                self._caputo_weights[j] = 0
        
        # Scale by gamma factor
        self._caputo_weights /= gamma(2-alpha) * self.dt**alpha
    
    def _caputo_fractional_derivative(self, f):
        """
        Compute the Caputo fractional derivative using the L1 scheme with differences.
        
        Parameters:
        -----------
        f : array_like
            Function values at time points
            
        Returns:
        --------
        df : array_like
            Fractional derivative at each time point
        """
        df = np.zeros_like(f)
        
        # Implement the L1 scheme explicitly using differences
        for k in range(1, len(f)):
            sum_term = 0
            for j in range(1, k+1):
                # Calculate difference: f(t_{k-j+1}) - f(t_{k-j})
                diff = f[k-j+1] - f[k-j]
                sum_term += self._caputo_weights[j] * diff
            
            df[k] = sum_term
        
        return df
    
    # Update solve_direct_problem docstring
    def solve_direct_problem(self, phi, psi=None, h=None, f=None):
        """
        Solve the direct problem: D_t^α u - Δu = h(x) + f(t), u(0,x) = phi(x), u(t,0) = u(t,π) = 0.
        
        Uses spectral method with solution:
        u(t,x) = Σ_n [phi_n E_α(-λ_n t^α) + h_n t^α E_α(-λ_n t^α) + ∫_0^t (t-τ)^(α-1) E_{α,α}(-λ_n (t-τ)^α) f(τ) dτ / Γ(α)] φ_n(x)
        
        Parameters:
        -----------
        phi : array_like
            Initial condition u(0,x)
        psi : array_like, optional
            Not used (for α < 1)
        h : array_like, optional
            Spatial source term h(x)
        f : array_like, optional
            Temporal source term f(t)
            
        Returns:
        --------
        u : array_like
            Solution u(t,x) at grid points
        """
        if self.N == 1:
            # Initialize solution array
            u = np.zeros((len(self.t), len(self.x)))
            
            # Set initial condition
            u[0, :] = phi
            
            # Default source terms to zero if not provided
            if h is None:
                h = np.zeros_like(self.x)
            if f is None:
                f = np.zeros_like(self.t)
            
            # Ensure shapes match
            if len(h) != len(self.x):
                raise ValueError(f"Spatial source h must have shape ({len(self.x)},), got {h.shape}")
            if len(f) != len(self.t):
                raise ValueError(f"Temporal source f must have shape ({len(self.t)},), got {f.shape}")
            
            # Compute eigenfunction coefficients for initial condition and source
            phi_coeffs = np.array([np.sum(phi * ef) * self.dx for ef in self.eigenfunctions])
            h_coeffs = np.array([np.sum(h * ef) * self.dx for ef in self.eigenfunctions])
            
            # Pre-allocate arrays for source terms
            homogeneous_terms = np.zeros((self.Nmax, len(self.t)))
            spatial_source_terms = np.zeros((self.Nmax, len(self.t)))
            temporal_source_terms = np.zeros((self.Nmax, len(self.t)))
            
            # Compute homogeneous terms and spatial source terms
            for n in range(self.Nmax):
                homogeneous_terms[n, :] = phi_coeffs[n] * self._ml_kernel[n, :]
                
                if np.any(h):
                    spatial_source_terms[n, :] = h_coeffs[n] * self.t**self.alpha * self._ml_kernel[n, :]
            
            # Correct temporal source term computation
            if np.any(f):
                for n in range(self.Nmax):
                    for k in range(1, len(self.t)):
                        t_k = self.t[k]
                        # Pre-compute time points for integration
                        tau_points = self.t[:k+1]
                        # Compute kernel values
                        time_diff = t_k - tau_points
                        kernel = (time_diff)**(self.alpha - 1) * np.array([
                            mittag_leffler(
                                self.alpha, 
                                self.alpha, 
                                -self.eigenvalues[n] * (t_diff)**self.alpha
                            ) for t_diff in time_diff
                        ])
                        # Compute integrand
                        integrand = f[:k+1] * kernel
                        # Perform numerical integration
                        temporal_source_terms[n, k] = np.trapz(integrand, x=tau_points) / gamma(self.alpha)
            
            # Assemble solution
            for k in range(1, len(self.t)):
                for n, ef in enumerate(self.eigenfunctions):
                    u[k, :] += (homogeneous_terms[n, k] + spatial_source_terms[n, k] + temporal_source_terms[n, k]) * ef
            
            return u
            
        elif self.N == 2:
            # Initialize solution array
            u = np.zeros((len(self.t), self.M, self.M))
            
            # Set initial condition
            u[0, :, :] = phi
            
            # Default source terms to zero if not provided
            if h is None:
                h = np.zeros((self.M, self.M))
            if f is None:
                f = np.zeros_like(self.t)
            
            # Compute eigenfunction coefficients
            phi_coeffs = np.array([np.sum(phi * ef) * self.dx**2 for ef in self.eigenfunctions])
            h_coeffs = np.array([np.sum(h * ef) * self.dx**2 for ef in self.eigenfunctions])
            
            # Pre-allocate arrays for source terms
            homogeneous_terms = np.zeros((len(self.eigenvalues), len(self.t)))
            spatial_source_terms = np.zeros((len(self.eigenvalues), len(self.t)))
            temporal_source_terms = np.zeros((len(self.eigenvalues), len(self.t)))
            
            # Compute homogeneous and spatial source terms
            for n in range(len(self.eigenvalues)):
                homogeneous_terms[n, :] = phi_coeffs[n] * self._ml_kernel[n, :]
                
                if np.any(h):
                    spatial_source_terms[n, :] = h_coeffs[n] * self.t**self.alpha * self._ml_kernel[n, :]
            
            # Correct temporal source term computation for 2D case
            if np.any(f):
                for n in range(len(self.eigenvalues)):
                    for k in range(1, len(self.t)):
                        t_k = self.t[k]
                        # Pre-compute time points for integration
                        tau_points = self.t[:k+1]
                        # Compute kernel values
                        time_diff = t_k - tau_points
                        kernel = (time_diff)**(self.alpha - 1) * np.array([
                            mittag_leffler(
                                self.alpha, 
                                self.alpha, 
                                -self.eigenvalues[n] * (t_diff)**self.alpha
                            ) for t_diff in time_diff
                        ])
                        # Compute integrand
                        integrand = f[:k+1] * kernel
                        # Perform numerical integration
                        temporal_source_terms[n, k] = np.trapz(integrand, x=tau_points) / gamma(self.alpha)
            
            # Assemble solution
            for k in range(1, len(self.t)):
                for n, ef in enumerate(self.eigenfunctions):
                    u[k] += (homogeneous_terms[n, k] + spatial_source_terms[n, k] + temporal_source_terms[n, k]) * ef
            
            return u

    # Update solve_inverse_problem_1 docstring
    def solve_inverse_problem_1(self, g, lambda_reg=1e-4, method='tikhonov', x_obs_idx=None):
        K = self.K
        alpha = self.alpha
        
        # Set default observation point to middle of domain if not specified
        if x_obs_idx is None:
            x_obs_idx = self.M // 2
        
        # Construct the Volterra kernel matrix with correct time differences
        K_matrix = np.zeros((K+1, K+1))
        for i in range(K+1):
            for j in range(i):  # Only j < i contributes
                kernel_sum = 0
                for n in range(self.Nmax):
                    ef_val = self.eigenfunctions[n][x_obs_idx]
                    t_diff = self.t[i] - self.t[j]
                    kernel = t_diff**(self.alpha - 1) * mittag_leffler(
                        self.alpha, 
                        self.alpha, 
                        -self.eigenvalues[n] * t_diff**self.alpha
                    )
                    kernel_sum += ef_val**2 * kernel
                K_matrix[i, j] = kernel_sum * self.dt / gamma(self.alpha)
        
        # Make matrix symmetric (only needed for some solvers)
        K_matrix = K_matrix + K_matrix.T - np.diag(np.diag(K_matrix))
        
        if method == 'l1':
            def ista(A, b, lambda_reg, max_iter=1000, tol=1e-6):
                # Compute Lipschitz constant for step size
                L = np.linalg.norm(A.T @ A, ord=2)
                step_size = 1.0 / L
                
                x = np.zeros(A.shape[1])
                x_prev = x.copy()
                
                for _ in range(max_iter):
                    # Gradient step
                    grad = A.T @ (A @ x - b)
                    x_new = x - step_size * grad
                    
                    # Soft thresholding
                    x_new = np.sign(x_new) * np.maximum(
                        np.abs(x_new) - step_size * lambda_reg, 0
                    )
                    
                    # Check convergence
                    if np.linalg.norm(x_new - x_prev) < tol:
                        break
                        
                    x_prev = x.copy()
                    x = x_new
                
                return x
            
            f = ista(K_matrix, g, lambda_reg)
            
        elif method == 'tikhonov':
            # Add Tikhonov regularization
            A = K_matrix + lambda_reg * np.eye(K+1)
            f = linalg.solve(A, g, assume_a='sym')
            
        elif method == 'tsvd':
            # Truncated SVD regularization
            U, s, Vh = linalg.svd(K_matrix)
            s_filtered = np.where(s > lambda_reg * s[0], s, 0)
            s_inv = np.where(s_filtered > 0, 1.0 / s_filtered, 0)
            f = Vh.T @ (s_inv * (U.T @ g))
            
        else:
            raise ValueError("Regularization method must be 'tikhonov', 'tsvd', or 'l1'")
            
        return f

    # Update solve_inverse_problem_2 docstring
    def solve_inverse_problem_2(self, omega, f, lambda_reg=1e-4, method='tikhonov'):
        """
        Solve inverse problem 2: recover h(x) from spatial measurement omega(x) = u(T,x).
        
        Solves the equation:
        omega(x) = Σ_n [phi_n E_α(-λ_n T^α) + h_n T^α E_α(-λ_n T^α) + ∫_0^T (T-τ)^(α-1) E_{α,α}(-λ_n (T-τ)^α) f(τ) dτ / Γ(α)] φ_n(x)
        
        Parameters:
        -----------
        omega : array_like
            Spatial measurement omega(x) = u(T,x)
        f : array_like
            Known temporal source term f(t)
        lambda_reg : float
            Regularization parameter
        method : str
            Regularization method ('tikhonov', 'tsvd', or 'l1')
            
        Returns:
        --------
        h : array_like
            Recovered spatial source term h(x)
        """
        if self.N == 1:
            # Compute eigenfunction coefficients for omega(x)
            omega_coeffs = np.array([np.sum(omega * ef) * self.dx for ef in self.eigenfunctions])
            
            # Compute the effect of f(t) with correct time differences
            f_effects = np.zeros(self.Nmax)
            for n in range(self.Nmax):
                integrand = np.zeros(self.K+1)
                for j in range(self.K+1):
                    tau = self.t[j]
                    t_T = self.T
                    if t_T > tau:  # Only consider valid time differences
                        kernel = (t_T - tau)**(self.alpha - 1) * mittag_leffler(
                            self.alpha, 
                            self.alpha, 
                            -self.eigenvalues[n] * (t_T - tau)**self.alpha
                        )
                        integrand[j] = f[j] * kernel
                f_effects[n] = np.trapz(integrand, x=self.t) / gamma(self.alpha)
            
            if method == 'tikhonov':
                # Compute coefficients for h(x) with Tikhonov regularization
                h_coeffs = np.zeros(self.Nmax)
                for n in range(self.Nmax):
                    denominator = self.T**self.alpha * self._ml_kernel[n, -1] + lambda_reg * self.eigenvalues[n]
                    h_coeffs[n] = (omega_coeffs[n] - f_effects[n]) / denominator
                    
            elif method == 'tsvd':
                # Construct system matrix
                A = np.diag(self.T**self.alpha * self._ml_kernel[:, -1])
                b = omega_coeffs - f_effects
                
                # SVD decomposition
                U, s, Vh = linalg.svd(A)
                s_filtered = np.where(s > lambda_reg * s[0], s, 0)
                s_inv = np.where(s_filtered > 0, 1.0 / s_filtered, 0)
                h_coeffs = Vh.T @ (s_inv * (U.T @ b))
                
            elif method == 'l1':
                # Use the same ISTA implementation as in solve_inverse_problem_1
                A = np.diag(self.T**self.alpha * self._ml_kernel[:, -1])
                b = omega_coeffs - f_effects
                
                def ista(A, b, lambda_reg, max_iter=1000, tol=1e-6):
                    L = np.linalg.norm(A.T @ A, ord=2)
                    step_size = 1.0 / L
                    
                    x = np.zeros(A.shape[1])
                    x_prev = x.copy()
                    
                    for _ in range(max_iter):
                        grad = A.T @ (A @ x - b)
                        x_new = x - step_size * grad
                        x_new = np.sign(x_new) * np.maximum(
                            np.abs(x_new) - step_size * lambda_reg, 0
                        )
                        
                        if np.linalg.norm(x_new - x_prev) < tol:
                            break
                            
                        x_prev = x.copy()
                        x = x_new
                    
                    return x
                
                h_coeffs = ista(A, b, lambda_reg)
                
            else:
                raise ValueError("Regularization method must be 'tikhonov', 'tsvd', or 'l1'")
                
            # Reconstruct h(x) from its coefficients
            h = np.zeros_like(self.x)
            for n, coeff in enumerate(h_coeffs):
                h += coeff * self.eigenfunctions[n]
            
            return h
            
        elif self.N == 2:
            # Compute eigenfunction coefficients for omega(x,y)
            omega_coeffs = []
            for ef in self.eigenfunctions:
                omega_coeff = np.sum(omega * ef) * self.dx**2
                omega_coeffs.append(omega_coeff)
            
            omega_coeffs = np.array(omega_coeffs)
            
            # Compute the effect of f(t) on each mode
            f_effects = np.zeros(len(self.eigenvalues))
            
            for n in range(len(self.eigenvalues)):
                # Compute the effect of f(t) on this mode
                f_effect = 0
                for k in range(self.K+1):
                    t_k = self.t[k]
                    for j in range(k+1):
                        t_j = self.t[j]
                        if k > j:
                            f_effect += f[j] * (t_k - t_j)**(self.alpha-1) * self._ml_kernel[n, k-j] * self.dt
                
                f_effects[n] = f_effect * (1 / gamma(self.alpha))
            
            if method == 'tikhonov':
                # Compute coefficients for h(x,y) with Tikhonov regularization
                h_coeffs = np.zeros(len(self.eigenvalues))
                
                for n in range(len(self.eigenvalues)):
                    denominator = self.T**self.alpha * self._ml_kernel[n, -1] + lambda_reg * self.eigenvalues[n]
                    h_coeffs[n] = (omega_coeffs[n] - f_effects[n]) / denominator
            
            elif method in ['tsvd', 'l1']:
                # Similar implementation as 1D case
                A = np.diag(self.T**self.alpha * self._ml_kernel[:, -1])
                b = omega_coeffs - f_effects
                
                if method == 'tsvd':
                    # SVD decomposition
                    U, s, Vh = linalg.svd(A)
                    
                    # Determine truncation level
                    s_filtered = np.where(s > lambda_reg * s[0], s, 0)
                    s_inv = np.where(s_filtered > 0, 1.0 / s_filtered, 0)
                    
                    # Reconstruct solution
                    h_coeffs = Vh.T @ (s_inv * (U.T @ b))
                
                else:  # L1 regularization
                    def objective(h_coeffs_flat):
                        h_coeffs_reshaped = h_coeffs_flat.reshape(-1)
                        residual = A @ h_coeffs_reshaped - b
                        return 0.5 * np.sum(residual**2) + lambda_reg * np.sum(np.abs(h_coeffs_reshaped))
                    
                    # Initial guess
                    h0 = np.zeros(len(self.eigenvalues))
                    
                    # Solve optimization problem
                    result = minimize(objective, h0, method='L-BFGS-B')
                    h_coeffs = result.x
            
            else:
                raise ValueError("Regularization method must be 'tikhonov', 'tsvd', or 'l1'")
            
            # Reconstruct h(x,y) from its coefficients
            h = np.zeros((self.M, self.M))
            for n, coeff in enumerate(h_coeffs):
                h += coeff * self.eigenfunctions[n]
            
            return h
    
    def estimate_optimal_regularization(self, g, method='l_curve', param_range=None):
        """
        Estimate optimal regularization parameter using L-curve or GCV with improved efficiency.
        
        Parameters:
        -----------
        g : array_like
            Observation data
        method : str
            Method to use ('l_curve' or 'gcv')
        param_range : array_like, optional
            Range of regularization parameters to test
            
        Returns:
        --------
        lambda_opt : float
            Optimal regularization parameter
        """
        if param_range is None:
            param_range = np.logspace(-8, 0, 20)
        
        # Compute K_matrix once for efficiency
        K = self.K
        K_matrix = np.zeros((K+1, K+1))
        x_obs_idx = self.M // 2
        
        # Build kernel matrix efficiently
        for i in range(K+1):
            for j in range(i):
                kernel_sum = 0
                t_diff = self.t[i] - self.t[j]
                
                # Vectorize eigenfunction computation
                ef_vals = np.array([ef[x_obs_idx] for ef in self.eigenfunctions])
                kernels = t_diff**(self.alpha - 1) * np.array([
                    mittag_leffler(
                        self.alpha,
                        self.alpha,
                        -eigenval * t_diff**self.alpha
                    ) for eigenval in self.eigenvalues[:self.Nmax]
                ])
                
                kernel_sum = np.sum(ef_vals**2 * kernels)
                K_matrix[i, j] = kernel_sum * self.dt / gamma(self.alpha)
        
        # Make matrix symmetric
        K_matrix = K_matrix + K_matrix.T - np.diag(np.diag(K_matrix))
        
        if method == 'l_curve':
            residual_norms = []
            solution_norms = []
            
            # Compute SVD once for efficiency
            U, s, Vh = linalg.svd(K_matrix)
            
            for lambda_reg in param_range:
                # Solve using pre-computed SVD
                s_filtered = s / (s**2 + lambda_reg)
                f = Vh.T @ (s_filtered * (U.T @ g))
                
                # Compute residual and solution norms
                residual = K_matrix @ f - g
                residual_norms.append(np.linalg.norm(residual))
                solution_norms.append(np.linalg.norm(f))
            
            # Convert to numpy arrays for vectorized operations
            residual_norms = np.array(residual_norms)
            solution_norms = np.array(solution_norms)
            
            # Compute curvature using vectorized operations
            x = np.log(residual_norms)
            y = np.log(solution_norms)
            
            # First derivatives using numpy gradient
            dx = np.gradient(x)
            dy = np.gradient(y)
            
            # Second derivatives
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            
            # Curvature formula (vectorized)
            curvature = np.abs(dx * d2y - d2x * dy) / (dx**2 + dy**2)**(1.5)
            
            # Find maximum curvature
            idx_max = np.argmax(curvature)
            lambda_opt = param_range[idx_max]
            
        elif method == 'gcv':
            gcv_values = []
            
            # Compute SVD once
            U, s, Vh = linalg.svd(K_matrix)
            
            for lambda_reg in param_range:
                # Compute filtered singular values
                s_filtered = s / (s**2 + lambda_reg)
                
                # Solve using pre-computed SVD
                f = Vh.T @ (s_filtered * (U.T @ g))
                
                # Compute residual
                residual = K_matrix @ f - g
                residual_norm_sq = np.sum(residual**2)
                
                # Compute trace term
                trace = np.sum(s_filtered)
                
                # Compute GCV function
                gcv = residual_norm_sq / (K+1 - trace)**2
                gcv_values.append(gcv)
            
            # Find minimum GCV value
            idx_min = np.argmin(gcv_values)
            lambda_opt = param_range[idx_min]
            
        else:
            raise ValueError("Method must be 'l_curve' or 'gcv'")
        
        return lambda_opt

# Demonstration script
def demo_improved_solver():
    # Parameters
    M = 100  # Spatial grid points
    K = 100  # Temporal grid points
    T = 1.0  # Time horizon
    
    # Test for different alpha values
    alpha_values = [0.5, 0.7, 0.9]
    
    # Define multiple test functions
    def create_test_functions(solver):
        x = solver.x
        t = solver.t
        
        h_true_options = [
            np.sin(2*x),                      # Smooth sinusoidal
            np.where(x < np.pi/2, 1, 0)       # Step function
        ]
        
        f_true_options = [
            t * np.exp(-2*t),                 # Decaying exponential
            np.sin(5*t)                       # Oscillatory function
        ]
        
        return h_true_options, f_true_options
    
    # Test different noise levels
    noise_levels = [0.001, 0.01, 0.05]
    
    for alpha in alpha_values:
        print(f"Testing with alpha = {alpha}")
        
        # Initialize solver
        solver = ImprovedFractionalPDEInverseSolver(N=1, M=M, K=K, T=T, alpha=alpha)
        
        # Get test functions
        h_true_options, f_true_options = create_test_functions(solver)
        
        for h_idx, h_true in enumerate(h_true_options):
            for f_idx, f_true in enumerate(f_true_options):
                print(f"  Testing with h_true[{h_idx}] and f_true[{f_idx}]")
                
                # Initial condition
                phi = np.zeros_like(solver.x)
                
                # Solve direct problem
                u = solver.solve_direct_problem(phi, h=h_true, f=f_true)
                
                # Extract observation data for inverse problems
                g = u[:, M//2]  # u(t, x_0) at middle point
                omega = u[-1, :]  # u(T, x) at final time
                
                for noise_level in noise_levels:
                    print(f"    Testing with noise level = {noise_level}")
                    
                    # Add noise
                    g_noisy = g + noise_level * np.max(np.abs(g)) * np.random.randn(len(g))
                    omega_noisy = omega + noise_level * np.max(np.abs(omega)) * np.random.randn(len(omega))
                    
                    # Estimate optimal regularization parameter
                    lambda_opt = solver.estimate_optimal_regularization(g_noisy, method='l_curve')
                    print(f"    Estimated optimal regularization parameter: {lambda_opt:.6e}")
                    
                    # Solve inverse problems with different methods
                    f_recovered_tikhonov = solver.solve_inverse_problem_1(g_noisy, lambda_reg=lambda_opt, method='tikhonov')
                    f_recovered_tsvd = solver.solve_inverse_problem_1(g_noisy, lambda_reg=lambda_opt*10, method='tsvd')
                    f_recovered_l1 = solver.solve_inverse_problem_1(g_noisy, lambda_reg=lambda_opt*0.1, method='l1')
                    
                    # Solve inverse problem 2: recover h(x)
                    h_recovered = solver.solve_inverse_problem_2(omega_noisy, f_true, lambda_reg=lambda_opt)
                    
                    # Plot results
                    plt.figure(figsize=(15, 12))
                    
                    # Plot direct solution
                    plt.subplot(2, 3, 1)
                    plt.imshow(u.T, aspect='auto', extent=[0, T, 0, np.pi], origin='lower')
                    plt.colorbar(label='u(t,x)')
                    plt.xlabel('t')
                    plt.ylabel('x')
                    plt.title(f'Direct Solution (α = {alpha})')
                    
                    # Plot recovered vs true f(t) using different methods
                    plt.subplot(2, 3, 2)
                    plt.plot(t, f_true, 'k-', label='True f(t)')
                    plt.plot(t, f_recovered_tikhonov, 'r--', label='Tikhonov')
                    plt.plot(t, f_recovered_tsvd, 'g-.', label='TSVD')
                    plt.plot(t, f_recovered_l1, 'b:', label='L1')
                    plt.xlabel('t')
                    plt.ylabel('f(t)')
                    plt.title('Inverse Problem 1: Recovering f(t)')
                    plt.legend()
                    
                    # Plot recovered vs true h(x)
                    plt.subplot(2, 3, 3)
                    plt.plot(x, h_true, 'k-', label='True h(x)')
                    plt.plot(x, h_recovered, 'r--', label='Recovered h(x)')
                    plt.xlabel('x')
                    plt.ylabel('h(x)')
                    plt.title('Inverse Problem 2: Recovering h(x)')
                    plt.legend()
                    
                    # Plot error metrics for different regularization methods
                    plt.subplot(2, 3, 4)
                    f_error_tikhonov = np.linalg.norm(f_recovered_tikhonov - f_true) / np.linalg.norm(f_true)
                    f_error_tsvd = np.linalg.norm(f_recovered_tsvd - f_true) / np.linalg.norm(f_true)
                    f_error_l1 = np.linalg.norm(f_recovered_l1 - f_true) / np.linalg.norm(f_true)
                    h_error = np.linalg.norm(h_recovered - h_true) / np.linalg.norm(h_true)
                    
                    methods = ['Tikhonov', 'TSVD', 'L1', 'h(x)']
                    errors = [f_error_tikhonov, f_error_tsvd, f_error_l1, h_error]
                    plt.bar(methods, errors)
                    plt.ylabel('Relative L2 Error')
                    plt.title('Error Metrics')
                    
                    # Plot L-curve for regularization parameter selection
                    plt.subplot(2, 3, 5)
                    param_range = np.logspace(-8, 0, 10)
                    residual_norms = []
                    solution_norms = []
                    
                    for lambda_reg in param_range:
                        f_test = solver.solve_inverse_problem_1(g_noisy, lambda_reg=lambda_reg, method='tikhonov')
                        
                        # Build kernel matrix (simplified)
                        K_matrix = np.zeros((K+1, K+1))
                        for i in range(K+1):
                            for j in range(i+1):
                                if i > j:
                                    kernel_sum = 0
                                    for n, ef in enumerate(solver.eigenfunctions):
                                        x_obs_idx = solver.M // 2
                                        ef_val = ef[x_obs_idx]
                                        kernel_sum += ef_val**2 * (solver.t[i] - solver.t[j])**(solver.alpha-1) * solver._ml_kernel[n, i-j]
                                    
                                    K_matrix[i, j] = kernel_sum * solver.dt / gamma(solver.alpha)
                        
                        residual = K_matrix @ f_test - g_noisy
                        residual_norm = np.linalg.norm(residual)
                        solution_norm = np.linalg.norm(f_test)
                        
                        residual_norms.append(residual_norm)
                        solution_norms.append(solution_norm)
                    
                    plt.loglog(residual_norms, solution_norms, 'bo-')
                    for i, lambda_reg in enumerate(param_range):
                        plt.annotate(f'{lambda_reg:.1e}', (residual_norms[i], solution_norms[i]), 
                                     textcoords="offset points", xytext=(0,10), ha='center')
                    plt.xlabel('Residual Norm ||Af-g||')
                    plt.ylabel('Solution Norm ||f||')
                    plt.title('L-curve for Parameter Selection')
                    
                    # Plot convergence of error vs regularization parameter
                    plt.subplot(2, 3, 6)
                    lambda_range = np.logspace(-8, 0, 20)
                    errors = []
                    
                    for lambda_reg in lambda_range:
                        f_test = solver.solve_inverse_problem_1(g_noisy, lambda_reg=lambda_reg, method='tikhonov')
                        error = np.linalg.norm(f_test - f_true) / np.linalg.norm(f_true)
                        errors.append(error)
                    
                    plt.loglog(lambda_range, errors, 'ro-')
                    plt.axvline(lambda_opt, color='g', linestyle='--', label=f'λ_opt = {lambda_opt:.1e}')
                    plt.xlabel('Regularization Parameter λ')
                    plt.ylabel('Relative Error')
                    plt.title('Error vs. Regularization Parameter')
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig(f'fractional_pde_inverse_alpha_{alpha}_improved.png')
                    plt.show()
                    
                    # Print error metrics
                    print(f"Relative L2 error for f(t) (Tikhonov): {f_error_tikhonov:.4f}")
                    print(f"Relative L2 error for f(t) (TSVD): {f_error_tsvd:.4f}")
                    print(f"Relative L2 error for f(t) (L1): {f_error_l1:.4f}")
                    print(f"Relative L2 error for h(x): {h_error:.4f}")
                    print(f"Optimal regularization parameter: {lambda_opt:.6e}")
                    print("-" * 50)

if __name__ == "__main__":
    demo_improved_solver()