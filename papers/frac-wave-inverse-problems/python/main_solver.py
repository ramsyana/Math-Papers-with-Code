# -*- coding: utf-8 -*-
import numpy as np
import scipy.integrate as spi
import scipy.special as sps
import scipy.optimize as spo
import matplotlib.pyplot as plt
import warnings
import functools # For caching
import time
import math # For isinf/isnan if needed
import traceback # For detailed error printing

# --- Module Description ---
"""
Numerical implementation for solving direct and inverse source problems
for a 1D time-fractional wave equation:
  d^alpha u/dt^alpha - L u = f(t) h(x),  1 < alpha < 2
where L = -d^2/dx^2 with Dirichlet boundary conditions.

Based on the work by D.K. Durdiev.

Includes:
- Direct Problem solver using Fourier method (Eq. 15), with improved integration.
- Inverse Problem 1 (IP1) solver: Recovers f(t) given h(x) and integral data g(t),
  using a second-kind Volterra equation solved via Product Trapezoidal rule (Eq. 29).
- Inverse Problem 2 (IP2) basic structure: Recovers h(x) given f(t) and integral data w(x),
  using optimization (least-squares).

Key Features:
- Improved DP solver using weighted integration for singularity handling.
- L1 scheme for approximating fractional derivatives required in IP1.
- Mittag-Leffler function computed via series summation (with caching).
  WARNING: Series summation has known accuracy/stability limitations.
- Product Trapezoidal rule for solving the second-kind Volterra equation (IP1).
- Basic optimization structure (Nelder-Mead) for IP2 with regularization.
- Error handling, NaN checks, validation metrics, and visualization.
"""

# --- Configuration ---
ALPHA = 1.5         # Fractional order (1 < alpha < 2)
L_DOMAIN = np.pi    # Domain length [0, L] (1D)
T_FINAL = 1.0       # Final time
N_TERMS = 40        # Max Fourier terms; higher improves accuracy but increases computation time
NX = 51             # Spatial points; odd ensures center point, affects resolution
NT = 101            # Time points; odd ensures midpoint, affects temporal resolution
SERIES_TOL = 1e-10  # Tolerance for early stopping of infinite series summations (DP, Kernels)
ML_MAX_TERMS = 250  # Max terms specifically for ML series summation
ML_SERIES_TOL = 1e-14 # Tolerance specifically for ML series summation
CHECK_RESIDUAL_IP1 = True # Optional: Check & Plot Volterra solver residual for IP1
IP2_REG_LAMBDA = 1e-9     # Regularization parameter for IP2 (increase for noisy data; typical 1e-9 to 1e-3)
IP2_DP_N_TERMS_OPTIM = 20 # Fewer terms for DP solver inside IP2 optimization loop for speed
IP2_MAX_ITER = 100        # Max iterations for IP2 optimizer

# --- Mittag-Leffler Function (Series Implementation) ---

def mittag_leffler_series(z, alpha, beta, max_terms=ML_MAX_TERMS, tol=ML_SERIES_TOL):
    """
    Computes the Mittag-Leffler function E_{alpha,beta}(z) using series definition.

    WARNING: This series summation can be numerically unstable or slow for
             large |z| or certain alpha/beta values. Use with caution.
             High-precision libraries or specialized algorithms are recommended
             for production use or challenging parameter ranges.

    Args:
        z (complex or float): Argument of the function.
        alpha (float): First parameter (alpha > 0).
        beta (float): Second parameter.
        max_terms (int): Maximum number of terms to sum.
        tol (float): Relative tolerance for early stopping.

    Returns:
        complex or float: Value of E_{alpha,beta}(z) or NaN on failure.
    """
    if not np.all(np.isfinite([alpha, beta])) or alpha <= 0:
        warnings.warn(f"Invalid alpha ({alpha}) or beta ({beta}) in mittag_leffler_series.", RuntimeWarning)
        return np.nan
    # Handle complex/real z and validate finiteness
    if np.iscomplexobj(z):
        if not np.all(np.isfinite(z)):
            warnings.warn("mittag_leffler_series called with non-finite complex z.", RuntimeWarning)
            return np.nan
        dtype = np.complex128
    else:
        if not np.isfinite(z):
            warnings.warn("mittag_leffler_series called with non-finite float z.", RuntimeWarning)
            return np.nan
        dtype = np.float64

    series_sum = dtype(0.0)
    z_pow_k = dtype(1.0) # z^0

    # k=0 term: 1 / Gamma(beta)
    try:
        log_gamma_beta = sps.loggamma(beta)
        # Handle potential overflow/underflow in exp
        term_k0 = np.exp(-log_gamma_beta) if np.isfinite(log_gamma_beta) else np.nan
        if not np.isfinite(term_k0):
            if beta <= 0 and beta == math.floor(beta): # Handle poles
                warnings.warn(f"Pole in Gamma(beta={beta}) at k=0. Returning 0.", RuntimeWarning)
                return dtype(0.0) # Result should be 0 at pole if defined by limit
            else:
                warnings.warn(f"Cannot compute 1/Gamma(beta={beta}) reliably at k=0. Returning NaN.", RuntimeWarning)
                return np.nan
        series_sum = dtype(term_k0)
    except ValueError:
        warnings.warn(f"Invalid argument beta={beta} for loggamma at k=0. Returning NaN.", RuntimeWarning)
        return np.nan
    except OverflowError:
         warnings.warn(f"Overflow calculating 1/Gamma(beta={beta}) at k=0. Returning NaN.", RuntimeWarning)
         return np.nan


    # Sum for k = 1, 2, ...
    last_term_mag = np.abs(term_k0) if np.isfinite(term_k0) else np.inf
    for k in range(1, max_terms):
        try:
            z_pow_k *= z
        except OverflowError:
             warnings.warn(f"Overflow computing z^k at k={k}. Series truncated.", RuntimeWarning)
             break # Stop summation
        if not np.isfinite(z_pow_k):
             warnings.warn(f"z^k non-finite at k={k}. Series truncated.", RuntimeWarning)
             break

        arg_gamma = alpha * k + beta
        # Check for poles in Gamma function
        if arg_gamma <= 0 and arg_gamma == math.floor(arg_gamma):
             warnings.warn(f"Pole encountered in Gamma({arg_gamma:.2f}) at k={k}. Series truncated.", RuntimeWarning)
             break

        try:
            log_gamma_term = sps.loggamma(arg_gamma)
        except ValueError:
             warnings.warn(f"Invalid argument {arg_gamma:.2f} for loggamma at k={k}. Series truncated.", RuntimeWarning)
             series_sum = dtype(np.nan); break # Mark sum as invalid and stop
        if not np.isfinite(log_gamma_term):
             warnings.warn(f"loggamma({arg_gamma:.2f}) resulted in non-finite value at k={k}. Series truncated.", RuntimeWarning)
             series_sum = dtype(np.nan); break

        try:
            # Using np.exp handles complex log_gamma_term correctly
            inv_gamma = np.exp(-log_gamma_term)
            term_val = z_pow_k * inv_gamma
        except OverflowError:
             warnings.warn(f"Overflow calculating term value at k={k}. Series truncated.", RuntimeWarning)
             break
        except Exception as e_term: # Catch unexpected math errors
             warnings.warn(f"Error calculating term value at k={k}: {e_term}. Series truncated.", RuntimeWarning)
             break

        if not np.isfinite(term_val):
            warnings.warn(f"Non-finite term value encountered at k={k}. Series truncated.", RuntimeWarning)
            break

        # Check for potential underflow in summation before adding
        if np.abs(term_val) < 1e-300 and np.abs(series_sum) < 1e-300 and k > 10:
            warnings.warn(f"Potential underflow, term magnitude ~{np.abs(term_val):.1e} at k={k}. Stopping early.", RuntimeWarning)
            break

        series_sum += term_val
        term_mag = np.abs(term_val)

        # Robust Convergence Check: Compare term magnitude to sum magnitude
        # Avoid division by zero or near-zero sum issues
        sum_mag = np.abs(series_sum)
        if k > 5 and term_mag < tol * (sum_mag + tol): # Relative check + absolute floor tol
             break

        last_term_mag = term_mag

    else: # Loop finished without break (max_terms reached)
         # Check if last term was still significant
         sum_mag = np.abs(series_sum)
         if last_term_mag > tol * (sum_mag + tol) and np.isfinite(last_term_mag):
             warnings.warn(f"ML series max_terms ({max_terms}) reached before convergence criterion met. "
                           f"|z|={np.abs(z):.1e}, a={alpha}, b={beta}. Last term mag: {last_term_mag:.1e}", RuntimeWarning)

    if not np.isfinite(series_sum):
         warnings.warn(f"Final ML sum is non-finite. |z|={np.abs(z):.1e}, a={alpha}, b={beta}. Returning NaN.", RuntimeWarning)
         return np.nan

    return series_sum


# --- Caching Wrapper for Mittag-Leffler ---
@functools.lru_cache(maxsize=None) # Unlimited cache size
def mittag_leffler(alpha, beta, z):
    """
    Cached wrapper for computing the Mittag-Leffler function E_{alpha,beta}(z).

    WARNING: Uses a potentially inaccurate/unstable series summation implementation.
             See mittag_leffler_series docstring for details and limitations.
    """
    # Ensure z is hashable for caching (convert NumPy types if necessary)
    if isinstance(z, (np.complex128, np.complex64)):
        z_hashable = complex(z)
    elif isinstance(z, (np.float64, np.float32)):
         z_hashable = float(z)
    else:
         z_hashable = z # Assume already hashable (like standard complex or float)

    # Ensure alpha and beta are standard floats for hashing
    alpha_f = float(alpha)
    beta_f = float(beta)

    return mittag_leffler_series(z_hashable, alpha_f, beta_f)


# --- Helper Functions (Eigenvalues, Eigenfunctions, Fourier Coeffs) ---
def eigenvalues(n_array, L_domain):
    """Calculates eigenvalues lambda_n = (n*pi/L)^2."""
    if L_domain <= 0: raise ValueError("L_domain must be positive")
    return (n_array * np.pi / L_domain)**2

def eigenfunctions(n, x_array, L_domain):
    """Calculates normalized eigenfunctions X_n(x) = sqrt(2/L)*sin(n*pi*x/L)."""
    if L_domain <= 0: raise ValueError("L_domain must be positive")
    # Ensure n is integer for the formula
    n_int = int(n)
    if n_int != n or n_int <= 0:
        raise ValueError(f"Eigenfunction index n must be a positive integer, got {n}")
    return np.sqrt(2.0 / L_domain) * np.sin(n_int * np.pi * x_array / L_domain)

def fourier_coeffs(func, max_n, L_domain, x_integration_points=None):
    """Calculates Fourier coefficients f_n = integral(f(x)*X_n(x) dx) from 0 to L."""
    if x_integration_points is None:
        # Use a reasonable number of points for integration grid if not provided
        num_int_points = max(int(4 * L_domain / np.pi * max_n), 201) # More points for higher n
        x_integration_points = np.linspace(0, L_domain, num_int_points)
        print(f"  (Fourier using {num_int_points} integration points)")

    coeffs = np.zeros(max_n)
    integration_warnings = 0
    nan_coeffs = 0

    for n in range(1, max_n + 1):
        try:
            # Define integrand for this n
            integrand = lambda x, n_local=n: func(x) * eigenfunctions(n_local, x, L_domain)

            # Evaluate integrand at integration points to check for issues
            y_integrand = integrand(x_integration_points)
            if np.any(np.isnan(y_integrand)) or np.any(np.isinf(y_integrand)):
                 warnings.warn(f"NaN/Inf detected in integrand for Fourier n={n}. Setting coeff to NaN.", RuntimeWarning)
                 coeffs[n-1] = np.nan
                 nan_coeffs += 1
                 continue # Skip integration if integrand is bad

            # Perform numerical integration using quad
            coeff_val, int_err = spi.quad(integrand, 0, L_domain,
                                          limit=250, # Default limit is 50, increase slightly
                                          epsabs=1e-10, epsrel=1e-10)

            if not np.isfinite(coeff_val):
                 warnings.warn(f"Integration result non-finite for Fourier n={n}. Setting coeff to NaN.", RuntimeWarning)
                 coeffs[n-1] = np.nan
                 nan_coeffs += 1
                 continue

            coeffs[n-1] = coeff_val

            # Check integration error against a threshold
            if abs(int_err) > 1e-5: # Threshold for high error warning
                warnings.warn(f"High integration error ({int_err:.1e}) encountered for Fourier coefficient n={n}", RuntimeWarning)
                integration_warnings += 1

        except ValueError as ve: # Catch specific errors like from eigenfunctions
            warnings.warn(f"ValueError during Fourier calc n={n}: {ve}. Setting coeff to NaN.", RuntimeWarning)
            coeffs[n-1] = np.nan; nan_coeffs += 1
        except Exception as e: # Catch any other unexpected errors during integration
            warnings.warn(f"Unexpected error during Fourier integration n={n}: {e}. Setting coeff to NaN.", RuntimeWarning)
            coeffs[n-1] = np.nan; nan_coeffs += 1
            traceback.print_exc() # Print stack trace for unexpected errors

    if integration_warnings > 0:
        print(f"  (Fourier coeffs: {integration_warnings} high error warnings)")
    if nan_coeffs > 0:
        print(f"  (Fourier coeffs: {nan_coeffs}/{max_n} coefficients resulted in NaN)")
        # Decide if this should be a fatal error
        raise ValueError(f"{nan_coeffs}/{max_n} Fourier coefficients resulted in NaN. Cannot proceed.")

    return coeffs


# --- Fractional Derivative (L1 Scheme Approximation) ---
def frac_deriv_L1(t_points, y, order):
    """
    Approximates Caputo fractional derivative (0 < order < 1) using L1 scheme.
    Requires uniform time grid. Raises ValueError on failure or NaN result.
    """
    nt = len(t_points)
    if nt < 2: raise ValueError("Need at least 2 time points for L1 derivative.")
    dt = t_points[1] - t_points[0]
    # Check for uniform grid with tolerance
    if not np.allclose(np.diff(t_points), dt, atol=1e-9*dt + 1e-12): # Added abs tolerance
        raise ValueError("L1 scheme requires a uniform time grid.")
    if not (0 < order < 1):
        raise ValueError(f"L1 scheme requires order between 0 and 1 (exclusive), got {order:.3f}")
    if len(y) != nt:
        raise ValueError(f"Input array y has length {len(y)}, expected {nt}.")
    if np.any(np.isnan(y)):
        warnings.warn("Input array y contains NaN values. Result will likely be NaN.", RuntimeWarning)

    deriv = np.zeros(nt)
    try:
        # Use log-gamma for better stability with potentially large factorials
        log_gamma_factor = sps.loggamma(2.0 - order)
        prefactor = dt**(-order) * np.exp(-log_gamma_factor)
        if not np.isfinite(prefactor):
             raise ValueError(f"Prefactor calculation resulted in non-finite value (dt={dt:.2e}, order={order:.2f}).")
    except (ValueError, OverflowError) as e:
        raise ValueError(f"Error calculating L1 prefactor: {e}") from e

    # Precompute weights b_k = (k+1)^(1-order) - k^(1-order)
    k_vals = np.arange(nt)
    # Use np.power for potential non-integer exponents
    term1 = np.power(k_vals + 1.0, 1.0 - order)
    term0 = np.power(k_vals, 1.0 - order)
    b = term1 - term0 # b[0] = 1, b[1] = 2^(1-o) - 1, ...
    if np.any(np.isnan(b)):
        raise ValueError("NaN encountered during L1 weight calculation.")

    # Calculate derivative using convolution-like sum
    deriv[0] = 0.0 # Caputo derivative approximation at t=0
    for i in range(1, nt):
        # Difference y[j] - y[j-1] for j=1 to i
        diff_y = y[1:i+1] - y[:i]
        # Weights b[i-j] for j=1 to i (which is b[i-1], b[i-2], ..., b[0])
        weights = b[:i][::-1] # Reverse slice of b weights

        # Check for NaN before summation
        if np.any(np.isnan(diff_y)) or np.any(np.isnan(weights)):
             warnings.warn(f"NaN detected in diff_y or weights at step i={i}. Result will be NaN.", RuntimeWarning)
             deriv[i] = np.nan
             continue

        try:
            deriv_sum = np.sum(diff_y * weights)
            deriv[i] = prefactor * deriv_sum
        except Exception as e_sum:
            warnings.warn(f"Error during L1 summation at step i={i}: {e_sum}. Setting deriv[{i}] to NaN.", RuntimeWarning)
            deriv[i] = np.nan


    if np.any(np.isnan(deriv)):
        warnings.warn("NaN values detected in the calculated L1 derivative.", RuntimeWarning)
        # Decide if NaN should halt execution
        # raise ValueError("NaN encountered in L1 derivative calculation.")
    return deriv


# --- Direct Problem (DP) Solver (Improved Integration) ---
def solve_dp(t_points, x_points, alpha, L_domain, phi_func, psi_func, f_func, h_func,
             N_terms, series_tol=SERIES_TOL,
             # --- Integration Controls with Defaults ---
             quad_limit=250, quad_epsabs=1e-8, quad_epsrel=1e-8,
             high_err_warn_threshold=5e-4 # Threshold for custom error warning
             ):
    """
    Solves the Direct Problem using Fourier method (Eq. 15).
    Improved version uses quad's 'weight' argument for better singularity handling.
    Args:
        ... (standard args) ...
        quad_limit (int): Limit for quad subdivisions.
        quad_epsabs (float): Absolute tolerance for quad.
        quad_epsrel (float): Relative tolerance for quad.
        high_err_warn_threshold (float): Threshold to warn about high integration error.
    Raises ValueError on critical failure. See main module docstring.
    """
    start_time = time.time(); print("Calculating Fourier coeffs for DP...")
    try:
        phi_n = fourier_coeffs(phi_func, N_terms, L_domain, x_points)
        psi_n = fourier_coeffs(psi_func, N_terms, L_domain, x_points)
        h_n = fourier_coeffs(h_func, N_terms, L_domain, x_points)
    except ValueError as e_fc: # Catch potential NaN errors from fourier_coeffs
         print(f"ERROR: Failed to calculate Fourier coefficients: {e_fc}")
         raise ValueError("DP failed during Fourier coefficient calculation.") from e_fc
    print(f"Fourier coeffs done ({time.time() - start_time:.2f}s).")

    nt = len(t_points); nx = len(x_points); u = np.zeros((nt, nx))
    n_indices = np.arange(1, N_terms + 1); lambdas = eigenvalues(n_indices, L_domain)
    print(f"Solving DP via Fourier series (N={N_terms}, quad_limit={quad_limit}, tol={quad_epsrel:.1e})...");
    Xn_on_grid = np.array([eigenfunctions(n + 1, x_points, L_domain) for n in range(N_terms)])
    source_int_time = 0.0; series_sum_time = 0.0

    # --- Pre-check alpha for weighting ---
    beta_weight = alpha - 1.0
    if not (beta_weight > -1): # Corresponds to alpha > 0
         raise ValueError(f"Invalid alpha={alpha} for 'alg' weight. Need alpha > 0.")
    # wvar = (exponent_at_a, exponent_at_b) for quad's alg weight (x-a)^alpha * (b-x)^beta
    # Our singularity is (t-s)^(alpha-1) near s=t (upper limit)
    wvar = (0.0, beta_weight)

    # Pre-evaluate f(t) on the grid if it's array-compatible, for efficiency
    try:
        f_t_grid = f_func(t_points)
        use_f_grid = True
    except:
        use_f_grid = False
        warnings.warn("f_func does not seem array-compatible, will evaluate point-wise.", RuntimeWarning)


    for i, t in enumerate(t_points):
        loop_start_time = time.time()
        if abs(t) < 1e-15: # Handle t=0 using initial condition phi
            try:
                u[i, :] = phi_func(x_points)
                if np.any(np.isnan(u[i,:])):
                     warnings.warn("NaN detected in initial condition phi(x).", RuntimeWarning)
            except Exception as e_phi:
                 warnings.warn(f"Error evaluating phi_func at t=0: {e_phi}", RuntimeWarning)
                 u[i, :] = np.nan # Mark as NaN if IC fails
            continue # Move to next time step

        # Progress indicator
        if i > 0 and nt > 10 and i % max(1, nt // 10) == 0:
            print(f"  DP progress: t = {t:.3f}/{T_FINAL} (step {i}/{nt-1})")

        int_start = time.time(); integral_vals_n = np.zeros(N_terms)
        # Check if source function f(s) is non-zero anywhere in [0, t]
        source_active = False
        if use_f_grid:
            if np.any(f_t_grid[:i+1] != 0):
                source_active = True
        else:
            # Check a few points if not using grid (less reliable)
            if any(abs(f_func(ts)) > 1e-15 for ts in np.linspace(0, t, min(i+1, 10))):
                source_active = True

        if source_active:
            integration_failed_this_step = False
            for n in range(N_terms): # Potential parallelization target
                lambda_n = lambdas[n]

                # Define the CORE integrand (WITHOUT the singular (t-s)^(alpha-1) factor)
                # Pass necessary variables explicitly to avoid closure issues if parallelized
                def integrand_core(s, t_curr=t, ln=lambda_n, alpha_loc=alpha, f_func_loc=f_func):
                    ts_diff = t_curr - s
                    # Handle potential numerical issues near s=t boundary
                    if ts_diff <= 1e-15: return 0.0 # Integrand involves (t-s)**(alpha-1) which is handled by weight

                    try:
                        # --- Critical Section: ML Calculation ---
                        # Argument for ML can become very large negative: -lambda_n * (t-s)^alpha
                        ml_arg = -ln * (ts_diff**alpha_loc)
                        # Add check for extremely large negative arg where series might fail badly
                        if ml_arg < -700: # Approx exp(-700) is near float limit
                             ml_val = 0.0 # ML function decays rapidly for large negative arg
                        else:
                            ml_val = mittag_leffler(alpha_loc, alpha_loc, ml_arg)

                        # --- Critical Section: f(s) Evaluation ---
                        f_val = f_func_loc(s)

                        # Check for NaN/Inf results from ML or f
                        if not np.isfinite(ml_val) or not np.isfinite(f_val):
                             warnings.warn(f"Non-finite value in integrand_core (ML={ml_val:.1e}, f={f_val:.1e}) at s={s:.2f}, t={t_curr:.2f}, n={n+1}", RuntimeWarning)
                             return 0.0 # Return 0 to avoid corrupting integral, but flag it
                        return ml_val * f_val
                    except OverflowError:
                        warnings.warn(f"Overflow in integrand_core (likely ts_diff**alpha) at s={s:.2f}, t={t_curr:.2f}, n={n+1}. Returning 0.", RuntimeWarning)
                        return 0.0
                    except ValueError as ve_core: # e.g., domain error in power
                        warnings.warn(f"ValueError in integrand_core at s={s:.2f}, t={t_curr:.2f}, n={n+1}: {ve_core}. Returning 0.", RuntimeWarning)
                        return 0.0
                    except Exception as e_intcore: # Catch-all for unexpected issues
                         warnings.warn(f"Unexpected error in integrand_core at s={s:.2f}, t={t_curr:.2f}, n={n+1}: {e_intcore}. Returning NaN.", RuntimeWarning)
                         return np.nan # Use NaN to signal a definite failure

                try:
                    # Use quad with 'alg' weight for (t-s)^(alpha-1) singularity at s=t
                    val, err = spi.quad(integrand_core, 0, t,
                                        weight='alg', wvar=wvar,
                                        limit=quad_limit,
                                        epsabs=quad_epsabs, epsrel=quad_epsrel)

                    # Check integration result
                    if not np.isfinite(val):
                        warnings.warn(f"Integration with 'alg' weight returned non-finite value for n={n+1}, t={t:.3f}. Setting integral to 0.", RuntimeWarning)
                        val = 0.0 # Assign 0 to avoid NaN propagation, but indicates failure
                        integration_failed_this_step = True

                    integral_vals_n[n] = val

                    # Check estimated error against the custom threshold
                    if abs(err) > high_err_warn_threshold:
                        warnings.warn(f"High integration error ({err:.1e}) estimated by quad for n={n+1}, t={t:.3f} (using 'alg' weight)", RuntimeWarning)

                except Exception as e: # Catch errors during the quad call itself
                    warnings.warn(f"Integration call failed for n={n+1}, t={t:.3f}: {e}. Setting integral to NaN.", RuntimeWarning)
                    integral_vals_n[n] = np.nan
                    integration_failed_this_step = True
                    # Optionally print traceback for unexpected quad failures
                    # traceback.print_exc()

            if integration_failed_this_step:
                 warnings.warn(f"One or more integrations failed (returned NaN/Inf or quad failed) at t={t:.3f}.", RuntimeWarning)
                 # Depending on severity, could stop here by raising ValueError

        source_int_time += time.time() - int_start

        # --- Series Summation ---
        sum_start = time.time()
        sum_val_x = np.zeros(nx)
        term_norm_prev = np.inf
        t_pow_alpha = t**alpha # Precompute for efficiency
        terms_skipped = 0

        for n in range(N_terms):
            lambda_n = lambdas[n]
            ml1_val = np.nan; ml2_val = np.nan # Initialize to NaN

            try:
                # Arguments for ML functions in the summation part
                ml_arg_sum = -lambda_n * t_pow_alpha
                # Check arg magnitude before calling ML
                if ml_arg_sum < -700:
                    ml1_val = 0.0
                    ml2_val = 0.0
                else:
                    ml1_val = mittag_leffler(alpha, 1, ml_arg_sum)
                    ml2_val = mittag_leffler(alpha, 2, ml_arg_sum)

            except Exception as e_ml_sum:
                warnings.warn(f"Error calculating ML1/ML2 in summation for n={n+1}, t={t:.3f}: {e_ml_sum}. Skipping term.", RuntimeWarning)
                terms_skipped += 1
                continue # Skip this term

            # Check if ML results are valid
            if not np.isfinite(ml1_val) or not np.isfinite(ml2_val):
                warnings.warn(f"Non-finite ML1/ML2 value in summation for n={n+1}, t={t:.3f}. Skipping term.", RuntimeWarning)
                terms_skipped += 1
                continue

            # Assemble terms: Initial conditions + Source integral term
            term1 = phi_n[n] * ml1_val
            term2 = psi_n[n] * t * ml2_val # Note the 't' factor for psi term
            term3 = h_n[n] * integral_vals_n[n] # Integral result from above

            # Check if integral result was NaN
            if not np.isfinite(integral_vals_n[n]):
                 warnings.warn(f"Using integral result=NaN for term n={n+1}, t={t:.3f}. Skipping term.", RuntimeWarning)
                 terms_skipped += 1
                 continue

            # Combine terms and multiply by eigenfunction
            current_term_combined = term1 + term2 + term3
            current_term_x = current_term_combined * Xn_on_grid[n, :]

            # Check if the combined term is finite before adding
            if not np.all(np.isfinite(current_term_x)):
                warnings.warn(f"Non-finite values detected in summation term n={n+1} at t={t:.3f}. Skipping term.", RuntimeWarning)
                terms_skipped += 1
                continue

            sum_val_x += current_term_x

            # Convergence Check (same logic as before, check norm)
            current_term_norm = np.linalg.norm(current_term_x)
            sum_norm = np.linalg.norm(sum_val_x)
            # Check convergence based on relative contribution
            if n > 5 and current_term_norm < series_tol * (sum_norm + series_tol): # Avoid division by zero
                break # Exit loop early if converged

            term_norm_prev = current_term_norm
        else: # max N_terms reached without break
             # Check if last term was still significant
             sum_norm = np.linalg.norm(sum_val_x)
             if term_norm_prev > series_tol * (sum_norm + series_tol) and np.isfinite(term_norm_prev):
                 warnings.warn(f"DP Fourier Series summation reached N_terms={N_terms} at t={t:.3f} "
                               f"without meeting tolerance {series_tol:.1e}. Last term norm: {term_norm_prev:.1e}", RuntimeWarning)

        series_sum_time += time.time() - sum_start

        # Store the calculated sum for this time step
        u[i, :] = sum_val_x

        # Check for NaNs in the solution at this time step
        if not np.all(np.isfinite(u[i,:])):
             warnings.warn(f"NaN or Inf detected in final solution u(t, x) at t={t:.3f}. Problem might be ill-posed or unstable.", RuntimeWarning)
             # Option: Stop execution if solution becomes invalid
             # raise ValueError(f"Solution became non-finite at t={t:.3f}.")

    total_time = time.time() - start_time
    print(f"DP solved ({total_time:.2f}s). [Integral eval: {source_int_time:.2f}s, Series sum: {series_sum_time:.2f}s]")
    # Final check for NaNs in the whole solution array
    if np.any(np.isnan(u)):
        warnings.warn("NaN values detected in the final DP solution array u(t,x). Results may be unreliable.", RuntimeWarning)
    elif np.any(np.isinf(u)):
        warnings.warn("Infinity values detected in the final DP solution array u(t,x). Results unreliable.", RuntimeWarning)

    return u


# --- Inverse Problem 1 (IP1) Components ---

# _series_sum_with_tol helper (minor robustness added)
def _series_sum_with_tol(terms_generator, N_terms, tol):
    """Helper to sum a generator of terms with tolerance check."""
    total_sum = 0.0
    last_term_mag = np.inf
    nan_count = 0
    for n in range(N_terms):
        try:
            term = next(terms_generator)
        except StopIteration:
            break # Generator exhausted
        except Exception as e_gen:
            warnings.warn(f"Error getting next term from generator at n={n+1}: {e_gen}. Skipping.", RuntimeWarning)
            term = np.nan # Treat as NaN

        if not np.isfinite(term):
            # warnings.warn(f"Non-finite term encountered in series sum at n={n+1}. Skipping.", RuntimeWarning)
            nan_count += 1
            continue # Skip non-finite terms

        total_sum += term
        term_mag = abs(term)
        sum_mag = abs(total_sum)

        # Convergence check
        if n > 5 and term_mag < tol * (sum_mag + tol): # Relative + absolute floor
            break
        last_term_mag = term_mag
    else: # Max terms reached
        sum_mag = abs(total_sum)
        if last_term_mag > tol * (sum_mag + tol) and np.isfinite(last_term_mag):
             warnings.warn(f"Series reached N_terms={N_terms} without meeting tolerance {tol:.1e}. Last mag: {last_term_mag:.1e}", RuntimeWarning)

    if nan_count > 0:
         warnings.warn(f"Skipped {nan_count}/{N_terms} non-finite terms during series summation.", RuntimeWarning)

    # Return NaN if the final sum is not finite, unless it was meant to be zero
    if not np.isfinite(total_sum) and abs(total_sum) > 1e-15 : # Check if it should be zero
         warnings.warn(f"Final series sum is non-finite: {total_sum}. Returning NaN.", RuntimeWarning)
         return np.nan
    return total_sum


# Kernels K, K0 and Term G (rely on _series_sum_with_tol and mittag_leffler)
def kernel_K(t_val, alpha, h_coeffs, lambdas, N_terms, tol=SERIES_TOL):
    """Calculates Kernel K(t) for IP1 Volterra Eq (First Kind)."""
    if abs(t_val) < 1e-15: return 0.0
    if t_val < 0: raise ValueError("kernel_K requires t_val >= 0")
    t_pow_alpha = t_val**alpha
    N_coeffs = len(h_coeffs)
    N_lambdas = len(lambdas)
    max_n_local = min(N_terms, N_coeffs, N_lambdas)

    def terms():
        for n in range(max_n_local):
            try:
                ml_arg = -lambdas[n] * t_pow_alpha
                ml_val = 0.0 if ml_arg < -700 else mittag_leffler(alpha, alpha, ml_arg)
                if not np.isfinite(ml_val): yield np.nan; return # Stop if ML fails
                yield h_coeffs[n]**2 * ml_val
            except Exception as e: yield np.nan; warnings.warn(f"Error in K terms n={n+1}: {e}"); return
    k_sum = _series_sum_with_tol(terms(), max_n_local, tol)
    if np.isnan(k_sum): return np.nan
    return t_val**(alpha - 1.0) * k_sum

def term_G(t_val, alpha, g_func, phi_coeffs, psi_coeffs, h_coeffs, lambdas, N_terms, tol=SERIES_TOL):
    """Calculates Term G(t) = g(t) - SeriesPart for IP1 Volterra Eq."""
    if abs(t_val) < 1e-15: return g_func(0.0) # G(0) = g(0) - Series(t=0) = g(0)-sum(hn*phin) approx g(0) if ICs handled
    t_pow_alpha = t_val**alpha
    N_coeffs = min(len(phi_coeffs), len(psi_coeffs), len(h_coeffs))
    N_lambdas = len(lambdas)
    max_n_local = min(N_terms, N_coeffs, N_lambdas)

    def terms():
        for n in range(max_n_local):
            try:
                ml_arg = -lambdas[n] * t_pow_alpha
                if ml_arg < -700:
                     ml1_val = 0.0; ml2_val = 0.0
                else:
                     ml1_val = mittag_leffler(alpha, 1, ml_arg)
                     ml2_val = mittag_leffler(alpha, 2, ml_arg)
                if not np.isfinite(ml1_val) or not np.isfinite(ml2_val): yield np.nan; return
                yield h_coeffs[n] * (phi_coeffs[n] * ml1_val + psi_coeffs[n] * t_val * ml2_val)
            except Exception as e: yield np.nan; warnings.warn(f"Error in G terms n={n+1}: {e}"); return
    series_part = _series_sum_with_tol(terms(), max_n_local, tol)
    if np.isnan(series_part): return np.nan
    try:
        g_val_at_t = g_func(t_val)
        if not np.isfinite(g_val_at_t):
             warnings.warn(f"g_func({t_val:.2f}) returned non-finite value.", RuntimeWarning)
             return np.nan
        return g_val_at_t - series_part
    except Exception as e_g:
         warnings.warn(f"Error evaluating g_func({t_val:.2f}): {e_g}", RuntimeWarning)
         return np.nan

def kernel_K0(t_val, alpha, h_coeffs, lambdas, N_terms, tol=SERIES_TOL):
    """Calculates Kernel K0(t) for IP1 Volterra Eq (Second Kind)."""
    if abs(t_val) < 1e-15: return 0.0 # Limit as t->0 might be non-zero depending on alpha, but integral contribution is 0
    if t_val < 0: raise ValueError("kernel_K0 requires t_val >= 0")
    t_pow_alpha = t_val**alpha
    N_coeffs = len(h_coeffs)
    N_lambdas = len(lambdas)
    max_n_local = min(N_terms, N_coeffs, N_lambdas)

    def terms():
        for n in range(max_n_local):
            try:
                ml_arg = -lambdas[n] * t_pow_alpha
                ml_val = 0.0 if ml_arg < -700 else mittag_leffler(alpha, alpha, ml_arg)
                if not np.isfinite(ml_val): yield np.nan; return
                yield lambdas[n] * h_coeffs[n]**2 * ml_val # Note lambda_n factor
            except Exception as e: yield np.nan; warnings.warn(f"Error in K0 terms n={n+1}: {e}"); return
    k0_sum = _series_sum_with_tol(terms(), max_n_local, tol)
    if np.isnan(k0_sum): return np.nan
    # Factor is -t^(alpha-1) according to Eq (30) derivation
    return -t_val**(alpha - 1.0) * k0_sum


def solve_volterra_ip1_second_kind(t_points, alpha, L_domain, g_func, phi_coeffs, psi_coeffs, h_coeffs, N_terms, series_tol=SERIES_TOL, check_residual=CHECK_RESIDUAL_IP1):
    """
    Solves IP1 using second-kind Volterra eq (Eq. 29), L1 deriv, Product Trap rule.
    Assumes Eq (29) form: f(t)*Sum(hn^2) + integral( K0(t-s) * f(s) ds ) = G0(t)
    Raises ValueError on critical failure. See main module docstring.
    """
    start_time = time.time(); print("Setting up IP1 second-kind Volterra eq...")
    nt = len(t_points);
    if nt < 2: raise ValueError("Need at least 2 time points for Volterra solver.")
    dt = t_points[1] - t_points[0] # Assumes uniform grid (checked by L1)
    f_est = np.zeros(nt)
    n_indices = np.arange(1, N_terms + 1);
    try:
        lambdas = eigenvalues(n_indices, L_domain)
    except ValueError as e_lam: raise ValueError("Failed to calculate eigenvalues for IP1.") from e_lam

    # --- Calculate G(t) ---
    print("Calculating G(t) required for G0(t)...")
    G_vals = np.array([term_G(t, alpha, g_func, phi_coeffs, psi_coeffs, h_coeffs, lambdas, N_terms, series_tol) for t in t_points])
    if np.any(np.isnan(G_vals)): raise ValueError("NaN encountered in G(t) calculation. Check input functions or ML stability.")

    # --- Calculate G'(t) using gradient ---
    print("Calculating G'(t) using numerical gradient...")
    G_prime = np.gradient(G_vals, dt)
    if np.any(np.isnan(G_prime)): raise ValueError("NaN encountered in G'(t) calculation.")

    # --- Calculate G0(t) = D^(alpha-1) G'(t) using L1 ---
    print("Calculating G0(t) using L1 fractional derivative...")
    frac_order = alpha - 1.0
    if not (0 < frac_order < 1): raise ValueError(f"L1 scheme needs 0 < alpha-1 < 1 for G0 calc, got {frac_order:.3f}")
    try:
        G0_vals = frac_deriv_L1(t_points, G_prime, frac_order)
    except ValueError as e_l1: raise ValueError("Failed to calculate G0(t) using L1 scheme.") from e_l1
    if np.any(np.isnan(G0_vals)):
         warnings.warn("NaN encountered in G0(t) calculation (L1 result).", RuntimeWarning)
         # Option: raise ValueError here? Or let solver handle it.

    # --- Calculate Sum h_n^2 ---
    S_h_sq = np.sum(np.array(h_coeffs[:N_terms])**2) # Ensure using same N_terms consistently
    if S_h_sq < 1e-16:
        warnings.warn(f"Sum h_n^2 = {S_h_sq:.1e} is near zero. IP1 solution might be unstable or non-unique.", RuntimeWarning)
        # Depending on problem, might raise ValueError if h is essentially zero.
        if S_h_sq < 1e-30: raise ValueError(f"Sum h_n^2 = {S_h_sq:.1e} is effectively zero. Cannot solve IP1.")

    # --- Precompute Kernel K0 for Product Trapezoidal ---
    # Uses O(NT^2) memory but speeds up the solver loop.
    print("Precomputing Kernel K0(t) values needed...")
    K0_vals_diff = {} # Store K0(t_i - t_j)
    nan_in_k0 = False
    for i in range(nt):
         for j in range(i + 1):
              time_diff = t_points[i] - t_points[j]
              # Use small epsilon to avoid exact zero if needed, though K0(0) might be handled specially
              time_diff_key = max(time_diff, 0.0) # Ensure non-negative key
              if time_diff_key not in K0_vals_diff:
                   try:
                        k0_temp = kernel_K0(time_diff_key, alpha, h_coeffs, lambdas, N_terms, series_tol)
                        if not np.isfinite(k0_temp):
                             warnings.warn(f"NaN/Inf calculated for K0({time_diff_key:.3f}).", RuntimeWarning)
                             k0_temp = np.nan # Store NaN to signal failure
                             nan_in_k0 = True
                        K0_vals_diff[time_diff_key] = k0_temp
                   except Exception as e_k0:
                        warnings.warn(f"Error calculating K0({time_diff_key:.3f}): {e_k0}. Storing NaN.", RuntimeWarning)
                        K0_vals_diff[time_diff_key] = np.nan
                        nan_in_k0 = True

    if nan_in_k0:
        raise ValueError("NaN/Inf encountered during precomputation of Kernel K0. Cannot solve Volterra equation.")

    print("Solving Volterra Eq. (Product Trapezoidal Rule)...")
    # Product Trapezoidal Rule applied to:
    # f(t)*S_h_sq + integral_0^t K0(t-s)*f(s) ds = G0(t)
    # f_i * S_h_sq + dt * [ w_0*K0(t_i-t_0)*f_0 + w_1*K0(t_i-t_1)*f_1 + ... + w_i*K0(t_i-t_i)*f_i ] = G0_i
    # Weights w_j are 1/2 for j=0, i and 1 otherwise.

    f_est[0] = G0_vals[0] / S_h_sq if S_h_sq > 1e-16 else 0.0 # Estimate f(0) based on G0(0) ~ f(0)*S_h_sq
    # Note: G0(0) should ideally be 0 from L1 definition if G'(0)=0. Check this.
    if abs(G0_vals[0]) > 1e-6:
         warnings.warn(f"G0(0) = {G0_vals[0]:.2e} is non-zero. Check L1 implementation or g(0) condition.", RuntimeWarning)
         # Force f_est[0] = 0 if theory expects it? Depends on derivation details.
         f_est[0] = 0.0
         print("  (Forcing f_est[0] = 0 based on theory)")


    K0_at_0 = K0_vals_diff.get(0.0, 0.0) # K0(t_i - t_i) = K0(0)

    residuals = []; residual_times = [] # For optional residual check

    for i in range(1, nt):
        # Summation part: dt * [ w_0*K0_i0*f_0 + ... + w_{i-1}*K0_i{i-1}*f_{i-1} ]
        sum_term = 0.0
        # j=0 term (weight 1/2)
        K0_i0 = K0_vals_diff.get(t_points[i] - t_points[0], np.nan)
        sum_term += 0.5 * K0_i0 * f_est[0]
        # j=1 to i-1 terms (weight 1)
        for j in range(1, i):
            K0_ij = K0_vals_diff.get(t_points[i] - t_points[j], np.nan)
            sum_term += K0_ij * f_est[j]

        # Check if any K0 values were NaN
        if np.isnan(sum_term):
             warnings.warn(f"NaN encountered in summation term at step i={i}. Problem in K0 values.", RuntimeWarning)
             f_est[i:] = np.nan # Mark rest as NaN
             break

        # Equation for f_i:
        # f_i * S_h_sq + dt * sum_term + dt * w_i * K0(0) * f_i = G0_i
        # f_i * (S_h_sq + dt * 0.5 * K0_at_0) = G0_i - dt * sum_term
        denominator = S_h_sq + dt * 0.5 * K0_at_0

        if abs(denominator) < 1e-16:
             warnings.warn(f"Denominator near zero ({denominator:.1e}) in Volterra solver at step i={i}. Potential instability.", RuntimeWarning)
             # Avoid division by zero, maybe set to NaN or large value?
             f_est[i:] = np.nan
             break

        rhs = G0_vals[i] - dt * sum_term
        if np.isnan(rhs): # Check if G0 was NaN
            warnings.warn(f"NaN RHS in Volterra solver at step i={i} (G0 issue?).", RuntimeWarning)
            f_est[i:] = np.nan
            break

        f_est[i] = rhs / denominator

        if not np.isfinite(f_est[i]):
             warnings.warn(f"Non-finite f_est[{i}] calculated ({f_est[i]:.1e}). Check stability.", RuntimeWarning)
             f_est[i:] = np.nan # Mark rest as NaN
             break

        # Optional Residual Check (Recalculate integral based on current f_est)
        if check_residual and (i % max(1, nt//10) == 0 or i == nt - 1):
             integral_chk = 0.0
             integral_chk += 0.5 * K0_vals_diff.get(t_points[i]-t_points[0], 0.0) * f_est[0] # j=0
             for j in range(1, i): # j=1 to i-1
                 integral_chk += K0_vals_diff.get(t_points[i]-t_points[j], 0.0) * f_est[j]
             integral_chk += 0.5 * K0_at_0 * f_est[i] # j=i
             integral_chk *= dt

             res = G0_vals[i] - (f_est[i]*S_h_sq + integral_chk)
             residuals.append(abs(res)); residual_times.append(t_points[i])

    total_time = time.time() - start_time
    print(f"Volterra eq solved ({total_time:.2f}s).")
    if check_residual and residuals:
        max_res = max(residuals) if residuals else np.nan
        print(f"  Max Volterra residual checked: {max_res:.3e}")
        # Optional residual plot
        plt.figure(figsize=(7,4)); plt.semilogy(residual_times, residuals, 'o-'); plt.title('IP1 Volterra Equation Residuals (Approx)')
        plt.xlabel('Time t'); plt.ylabel('Absolute Residual'); plt.grid(True); plt.tight_layout(); plt.show(block=False)

    if np.any(np.isnan(f_est)):
        warnings.warn("NaN values detected in the final estimated f(t) for IP1.", RuntimeWarning)
    return f_est


# --- Inverse Problem 2 (IP2) Basic Structure ---
def solve_ip2_optimization(
    initial_h_coeffs_guess, t_points, x_points, alpha, L_domain,
    f_func, omega_target_on_grid, N_terms_h, dp_N_terms_optim,
    dp_series_tol_optim=SERIES_TOL*10, reg_lambda=IP2_REG_LAMBDA, maxiter=IP2_MAX_ITER,
    # Add DP integration controls for the internal solver
    dp_quad_limit=150, dp_quad_epsabs=1e-6, dp_quad_epsrel=1e-6
    ):
    """
    Basic structure for solving IP2 using optimization (least-squares).
    Finds Fourier coefficients of h(x) minimizing misfit + regularization.
    Uses Nelder-Mead. Raises ValueError on critical failure.
    """
    print("\n--- Setting up Inverse Problem 2 (Optimization) ---")
    nt = len(t_points); nx = len(x_points)
    if len(omega_target_on_grid) != nx:
         raise ValueError(f"omega_target_on_grid length ({len(omega_target_on_grid)}) != nx ({nx})")
    if len(initial_h_coeffs_guess) < N_terms_h:
         raise ValueError(f"initial_h_coeffs_guess length ({len(initial_h_coeffs_guess)}) < N_terms_h ({N_terms_h})")

    dt = t_points[1] - t_points[0] # Assumes uniform grid
    iteration_costs = []
    iteration_count = [0] # Use list to allow modification within objective_func

    # Store best result found so far
    best_cost = [np.inf]
    best_h_coeffs = [np.copy(initial_h_coeffs_guess[:N_terms_h])]


    def objective_func(current_h_coeffs_padded):
        iteration_count[0] += 1
        iter_start_time = time.time()
        current_h_coeffs = current_h_coeffs_padded[:N_terms_h] # Use only the first N_terms_h

        print(f"  IP2 Iteration {iteration_count[0]}... (Cost eval)")

        # Define the h(x) function based on current coefficients
        def current_h_func(x_arg):
            h_val = np.zeros_like(x_arg, dtype=float) # Ensure float output
            for n in range(N_terms_h):
                 try: h_val += current_h_coeffs[n] * eigenfunctions(n + 1, x_arg, L_domain)
                 except Exception as e_ef: print(f"Error in ef {n+1}: {e_ef}"); return np.nan
            return h_val

        # --- Solve the Direct Problem with current h(x) ---
        # Use zero ICs for the forward simulation within optimization
        phi_zero = lambda x: 0.0 * x
        psi_zero = lambda x: 0.0 * x
        u_current = None
        dp_success = False
        try:
            print(f"    Solving internal DP (N={dp_N_terms_optim}, tol={dp_series_tol_optim:.1e})...")
            # Pass DP integration controls
            u_current = solve_dp(t_points, x_points, alpha, L_domain,
                                 phi_zero, psi_zero,
                                 f_func, current_h_func,
                                 dp_N_terms_optim, dp_series_tol_optim,
                                 quad_limit=dp_quad_limit,
                                 quad_epsabs=dp_quad_epsabs,
                                 quad_epsrel=dp_quad_epsrel
                                 )
            if u_current is not None and np.all(np.isfinite(u_current)):
                dp_success = True
            else:
                 warnings.warn("Internal DP solve resulted in None or non-finite values.", RuntimeWarning)

        except Exception as e:
            warnings.warn(f"    Internal DP solve failed during objective function: {e}", RuntimeWarning)
            # traceback.print_exc() # Optionally print for debugging

        # If DP failed, return a large cost
        if not dp_success:
            print(f"    DP solve failed. Assigning high cost.")
            iteration_costs.append(np.inf); return 1e12 # Return large finite number

        # --- Calculate omega(x) based on the solved u_current ---
        try:
            # Calculate time derivative u_t
            u_t_current = np.gradient(u_current, dt, axis=0)
            if np.any(np.isnan(u_t_current)): raise ValueError("NaN in u_t gradient")

            # Evaluate f(t) and multiply
            f_vals_t = f_func(t_points)[:, np.newaxis] # Ensure shape (nt, 1) for broadcasting
            integrand_vals = f_vals_t * u_t_current
            if np.any(np.isnan(integrand_vals)): raise ValueError("NaN in omega integrand")

            # Integrate over time using Simpson's rule if possible, else trapezoidal
            try:
                 integral_term_x = spi.simps(integrand_vals, t_points, axis=0)
            except ValueError: # If even number of points
                 integral_term_x = np.trapz(integrand_vals, t_points, axis=0)
            if np.any(np.isnan(integral_term_x)): raise ValueError("NaN in omega integral result")

        except Exception as e_omega:
            warnings.warn(f"    Error calculating omega(x) in objective function: {e_omega}", RuntimeWarning)
            iteration_costs.append(np.inf); return 1e12 # High cost if omega calculation fails

        # --- Calculate Cost Function (Misfit + Regularization) ---
        misfit = np.sum((integral_term_x - omega_target_on_grid)**2)
        # Apply regularization only to the active coefficients being optimized
        regularization = float(reg_lambda) * np.sum(current_h_coeffs**2)
        cost = misfit + regularization

        # Check if cost is finite
        if not np.isfinite(cost):
            warnings.warn(f"    Calculated cost is non-finite (Misfit:{misfit:.2e}, Reg:{regularization:.2e}). Assigning high cost.", RuntimeWarning)
            cost = 1e12 # Assign large finite number

        iteration_costs.append(cost)
        iter_time = time.time() - iter_start_time

        # Update best result found
        if cost < best_cost[0]:
             best_cost[0] = cost
             best_h_coeffs[0] = np.copy(current_h_coeffs)

        print(f"    Cost: {cost:.4e} (Misfit: {misfit:.4e}, Reg: {regularization:.4e}) [{iter_time:.1f}s]")
        return cost

    # --- Run the Optimization ---
    print(f"Starting optimization (Nelder-Mead) for {N_terms_h} h-coeffs...")
    print(f"  Regularization lambda = {reg_lambda:.1e}")
    print(f"  Max iterations = {maxiter}")
    opt_start_time = time.time()

    # Use a padded initial guess if optimizer needs full length
    initial_guess_padded = np.pad(initial_h_coeffs_guess[:N_terms_h], (0, max(0, len(initial_h_coeffs_guess) - N_terms_h)))

    optimizer_result = spo.minimize(objective_func, initial_guess_padded, # Pass potentially padded guess
                                    method='Nelder-Mead',
                                    options={'xatol': 1e-7, 'fatol': 1e-7, # Slightly tighter tolerances
                                             'disp': False, # Set to True for more optimizer output
                                             'maxiter': maxiter,
                                             'adaptive': True # Adjust simplex params adaptively
                                             })
    opt_time = time.time() - opt_start_time
    print(f"Optimization finished ({opt_time:.1f}s).")

    # --- Process Results ---
    if not optimizer_result.success:
        warnings.warn(f"IP2 Optimization FAILED to converge: {optimizer_result.message}", RuntimeWarning)
        # Return the best result found during iterations instead of the failed final one
        print(f"  Returning best coefficients found during optimization (Cost: {best_cost[0]:.4e})")
        optimizer_result.x = best_h_coeffs[0] # Overwrite potentially bad final result
        # We might still consider the result usable depending on the cost
        # optimizer_result.success = True # Optionally override success status if cost is low

    else:
        print(f"IP2 Optimization converged after {optimizer_result.nit} iterations.")
        # Ensure the returned result 'x' uses the best cost coefficients
        if optimizer_result.fun > best_cost[0] + 1e-9: # If final cost > best cost found
             warnings.warn("Optimizer final cost > best cost found. Using best cost result.", RuntimeWarning)
             optimizer_result.x = best_h_coeffs[0]
             optimizer_result.fun = best_cost[0]


    # Attach cost history for analysis
    optimizer_result.cost_history = iteration_costs
    # Return only the relevant N_terms_h coefficients
    optimizer_result.final_h_coeffs = optimizer_result.x[:N_terms_h]

    return optimizer_result


# --- Utility for Error Calculation ---
def calculate_errors(estimated, true_vals):
    """Calculates L2 relative and max absolute errors, returns (NaN, NaN) on failure."""
    if estimated is None or true_vals is None: return np.nan, np.nan
    # Ensure inputs are numpy arrays
    estimated = np.asarray(estimated)
    true_vals = np.asarray(true_vals)
    if estimated.shape != true_vals.shape:
        warnings.warn(f"Shape mismatch in error calculation: est {estimated.shape}, true {true_vals.shape}", RuntimeWarning)
        return np.nan, np.nan
    if not np.all(np.isfinite(estimated)) or not np.all(np.isfinite(true_vals)):
        warnings.warn("Non-finite values detected in error calculation input.", RuntimeWarning)
        return np.nan, np.nan

    diff = estimated - true_vals
    norm_diff = np.linalg.norm(diff)
    norm_true = np.linalg.norm(true_vals)

    if norm_true < 1e-15:
        # Handle case where true solution is zero or near-zero
        l2_rel_err = norm_diff # Use absolute norm if true norm is zero
    else:
        l2_rel_err = norm_diff / norm_true

    max_abs_err = np.max(np.abs(diff))

    # Check for NaN results (shouldn't happen if inputs are finite)
    if not np.isfinite(l2_rel_err): l2_rel_err = np.nan
    if not np.isfinite(max_abs_err): max_abs_err = np.nan

    return l2_rel_err, max_abs_err


# --- Main Execution Block (Example Usage) ---
if __name__ == "__main__":
    main_start_time = time.time()
    print("--- Fractional Wave Equation Solver ---")
    print(f"Params: alpha={ALPHA:.2f}, L={L_DOMAIN:.2f}, T={T_FINAL:.2f}")
    print(f"Grid: N_terms={N_TERMS}, NX={NX}, NT={NT}")
    print(f"Tolerances: Series={SERIES_TOL:.1e}, ML={ML_SERIES_TOL:.1e}")

    # Define Example Problem Functions
    phi = lambda x: 0.0 * x # Zero initial displacement
    psi = lambda x: 0.0 * x # Zero initial velocity
    # Example h(x): Combination of first and third eigenfunctions
    h_true_func = lambda x: 1.5 * eigenfunctions(1, x, L_DOMAIN) \
                       - 0.7 * eigenfunctions(3, x, L_DOMAIN)
                       # = 1.5*sqrt(2/pi)sin(x) - 0.7*sqrt(2/pi)sin(3x)
    # Example f(t): Smooth function
    f_true_func = lambda t: 1.0 + np.sin(np.pi * t / T_FINAL)**2

    # Setup grids
    t = np.linspace(0, T_FINAL, NT);
    x = np.linspace(0, L_DOMAIN, NX);
    dt = t[1]-t[0]

    # --- Solve Direct Problem ---
    print("\n--- Solving Direct Problem ---")
    u_dp = None
    dp_success_main = False
    try:
        # Use the improved solve_dp with specific integration controls
        u_dp = solve_dp(t, x, ALPHA, L_DOMAIN, phi, psi, f_true_func, h_true_func,
                        N_terms=N_TERMS, series_tol=SERIES_TOL,
                        quad_limit=300,       # Increased limit
                        quad_epsabs=1e-8,     # Moderately strict tolerances
                        quad_epsrel=1e-8,
                        high_err_warn_threshold=5e-4 # Warning threshold
                       )
        if u_dp is not None and np.all(np.isfinite(u_dp)):
            dp_success_main = True
            print("DP solved successfully.")
            # Plot DP solution
            plt.figure(figsize=(8, 5)); T_mesh, X_mesh = np.meshgrid(t, x, indexing='ij')
            contour=plt.contourf(X_mesh, T_mesh, u_dp, 50, cmap='viridis'); plt.colorbar(contour, label='u(t, x)')
            plt.xlabel('Space x'); plt.ylabel('Time t'); plt.title(f'DP Solution ($\\alpha={ALPHA}$)'); plt.tight_layout(); plt.show(block=False)
        else:
            print("DP solver returned None or non-finite values.")

    except Exception as e:
        print(f"ERROR during Direct Problem solve: {e}")
        traceback.print_exc() # Print stack trace for unexpected errors

    # --- Solve Inverse Problem 1 ---
    if dp_success_main:
        print("\n--- Solving Inverse Problem 1 (Recover f(t)) ---")
        f_estimated_ip1 = None
        ip1_success = False
        try:
            # Calculate g(t) from DP solution
            print("Calculating g(t) data...")
            g_vals=np.zeros(NT); h_vec=h_true_func(x) # h(x) is known
            for i in range(NT): g_vals[i]=np.trapz(h_vec*u_dp[i,:],x)
            g_func_ip1 = lambda tval: np.interp(tval, t, g_vals, left=g_vals[0], right=g_vals[-1])

            # Calculate necessary coefficients
            print("Calculating coefficients for IP1...")
            phi_n_ip1=fourier_coeffs(phi,N_TERMS,L_DOMAIN,x);
            psi_n_ip1=fourier_coeffs(psi,N_TERMS,L_DOMAIN,x)
            h_n_ip1=fourier_coeffs(h_true_func,N_TERMS,L_DOMAIN,x) # h is known

            # Check g(0) condition (optional but good practice)
            g0_data = g_func_ip1(0.0)
            g0_theory = np.sum(h_n_ip1 * phi_n_ip1) # Theoretical value: sum(hn*phin)
            print(f"Checking g(0) condition: Data g(0)={g0_data:.3e}, Theory sum(hn*phin)={g0_theory:.3e}")
            if not np.isclose(g0_data, g0_theory, atol=1e-5): # Tolerance check
                warnings.warn(f"g(0) condition potentially violated (Data: {g0_data:.2e}, Theory: {g0_theory:.2e}). Check ICs/problem setup.", RuntimeWarning)

            # Run IP1 solver
            f_estimated_ip1 = solve_volterra_ip1_second_kind(t, ALPHA, L_DOMAIN, g_func_ip1,
                                                             phi_n_ip1, psi_n_ip1, h_n_ip1,
                                                             N_TERMS, SERIES_TOL, CHECK_RESIDUAL_IP1)

            if f_estimated_ip1 is not None and np.all(np.isfinite(f_estimated_ip1)):
                ip1_success = True
                print("IP1 solved successfully.")
                # Calculate and print errors
                f_true_vals_main = f_true_func(t)
                l2_err,max_err=calculate_errors(f_estimated_ip1, f_true_vals_main)
                print(f"  IP1 f(t) L2 Relative Error : {l2_err:.3e}")
                print(f"  IP1 f(t) Max Absolute Error: {max_err:.3e}")
                # Plot comparison
                plt.figure(figsize=(8, 5));
                plt.plot(t, f_true_vals_main, 'b-', lw=2, label='True f(t)')
                plt.plot(t, f_estimated_ip1, 'r--', lw=1.5, label='Estimated f(t) (IP1)')
                plt.title(f'IP1 Result (L2 Err: {l2_err:.2e})')
                plt.xlabel('Time t'); plt.ylabel('f(t)'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show(block=False)
            else:
                print("IP1 solver returned None or non-finite values.")

        except Exception as e:
            print(f"ERROR during Inverse Problem 1 solve: {e}"); traceback.print_exc()
    else:
        print("\nSkipping Inverse Problem 1 because Direct Problem failed or produced invalid results.")

    # --- Solve Inverse Problem 2 ---
    if dp_success_main:
        print("\n--- Solving Inverse Problem 2 (Recover h(x)) ---")
        ip2_success = False
        try:
            # Calculate omega(x) data from DP solution
            print("Calculating omega(x) data...")
            u_t_dp_main = np.gradient(u_dp, dt, axis=0)
            f_vals_t_main = f_true_func(t)[:, np.newaxis]
            integrand_omega_main = f_vals_t_main * u_t_dp_main
            try: omega_true_vals_main = spi.simps(integrand_omega_main, t, axis=0)
            except ValueError: omega_true_vals_main = np.trapz(integrand_omega_main, t, axis=0)
            if np.any(np.isnan(omega_true_vals_main)): raise ValueError("NaN in omega data calc")

            # Setup and run IP2 Optimization
            N_h_optimize_main = 5 # Number of h coefficients to find
            initial_h_coeffs_main = np.zeros(N_h_optimize_main) # Simple zero initial guess

            ip2_result = solve_ip2_optimization(
                                initial_h_coeffs_main,
                                t, x, ALPHA, L_DOMAIN,
                                f_true_func, # f(t) is KNOWN for IP2
                                omega_true_vals_main, # Target data
                                N_terms_h=N_h_optimize_main,
                                dp_N_terms_optim=IP2_DP_N_TERMS_OPTIM, # Fewer terms for speed
                                dp_series_tol_optim=SERIES_TOL*100, # Relaxed tol
                                reg_lambda=IP2_REG_LAMBDA,
                                maxiter=IP2_MAX_ITER,
                                # Pass DP integration controls for internal solver
                                dp_quad_limit=200, dp_quad_epsabs=1e-7, dp_quad_epsrel=1e-7
                                )

            # Process IP2 results
            # Note: ip2_result.success might be False even if a reasonable result was found (due to optimizer limits)
            # Check the final cost and the returned coefficients.
            if hasattr(ip2_result, 'final_h_coeffs') and ip2_result.final_h_coeffs is not None:
                ip2_success = True # Consider it a success if coeffs were returned
                optimized_h_coeffs_main = ip2_result.final_h_coeffs
                print(f"\nIP2 Optimization finished (Success: {ip2_result.success}). Final Cost: {ip2_result.fun:.4e}")
                print("  Optimized h coeffs:", optimized_h_coeffs_main)

                # Reconstruct h(x) from optimized coefficients
                def h_est_func_ip2(x_pts):
                    h_val = np.zeros_like(x_pts, dtype=float)
                    n_max_rec = min(N_h_optimize_main, len(optimized_h_coeffs_main))
                    for n in range(n_max_rec):
                        h_val += optimized_h_coeffs_main[n] * eigenfunctions(n + 1, x_pts, L_DOMAIN)
                    return h_val

                # Compare with true h(x)
                h_true_grid_main = h_true_func(x)
                h_est_grid_main = h_est_func_ip2(x)
                l2_err_h, max_err_h = calculate_errors(h_est_grid_main, h_true_grid_main)
                print(f"  IP2 h(x) L2 Relative Error : {l2_err_h:.3e}")
                print(f"  IP2 h(x) Max Absolute Error: {max_err_h:.3e}")

                # Plot h(x) comparison
                plt.figure(figsize=(8, 5));
                plt.plot(x, h_true_grid_main, 'b-', lw=2, label='True h(x)')
                plt.plot(x, h_est_grid_main, 'r--', lw=1.5, label=f'Estimated h(x) ({N_h_optimize_main} terms)')
                plt.title(f'IP2 Result: True vs. Estimated h(x) (Err L2={l2_err_h:.2e})')
                plt.xlabel('Space x'); plt.ylabel('h(x)'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show(block=False)

                # Plot cost history if available
                if hasattr(ip2_result, 'cost_history') and ip2_result.cost_history:
                     plt.figure(figsize=(7,4)); plt.semilogy(ip2_result.cost_history, '.-'); plt.title("IP2 Optimization Cost History")
                     plt.xlabel("Iteration"); plt.ylabel("Cost Function"); plt.grid(True); plt.tight_layout(); plt.show(block=False)
            else:
                print(f"IP2 Optimization FAILED to produce usable coefficients.")
                if hasattr(ip2_result, 'message'): print(f"  Optimizer message: {ip2_result.message}")


        except Exception as e:
            print(f"ERROR during Inverse Problem 2 solve: {e}"); traceback.print_exc()
    else:
        print("\nSkipping Inverse Problem 2 because Direct Problem failed or produced invalid results.")

    # --- End ---
    main_end_time = time.time()
    print(f"\nTotal execution time: {main_end_time - main_start_time:.2f} seconds")
    print("\nClose plot windows to exit.")
    plt.show() # Keep plots open until manually closed