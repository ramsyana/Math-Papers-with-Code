import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import scipy.integrate as spi # Needed for omega calculation in test

# --- Import functions from the main solver file ---
# Ensure the main code is saved as 'main_solver.py' in the same directory
try:
    from main_solver import (
        ALPHA as MAIN_ALPHA, L_DOMAIN as MAIN_L_DOMAIN, T_FINAL as MAIN_T_FINAL,
        eigenvalues, eigenfunctions, fourier_coeffs,
        solve_dp, # Will import the IMPROVED version
        term_G, kernel_K0, solve_volterra_ip1_second_kind,
        solve_ip2_optimization, calculate_errors
    )
    print("Successfully imported functions from main_solver.py")
except ImportError as e:
    print(f"ERROR: Could not import from 'main_solver.py'. Make sure the file exists and is accessible.")
    print(f"Details: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    exit()


# --- Test Configuration ---
# Use parameters from main file or override for testing
ALPHA_TEST = MAIN_ALPHA
L_DOMAIN_TEST = MAIN_L_DOMAIN
T_FINAL_TEST = MAIN_T_FINAL

# Use potentially smaller values for faster testing
N_TERMS_TEST = 30         # Max terms for tests (can be smaller than main N_TERMS)
NX_TEST = 41              # Spatial points for tests
NT_TEST = 51              # Time points for tests
SERIES_TOL_TEST = 1e-8    # Series tolerance for tests

# IP2 Specific Test Config
N_H_OPTIMIZE_TEST = 4      # Number of h coefficients to optimize in IP2 test
IP2_REG_LAMBDA_TEST = 1e-10 # Regularization for IP2 test
IP2_MAX_ITER_TEST = 50     # Max iterations for IP2 optimizer in test
IP2_DP_N_TERMS_TEST = 15   # Fewer terms for internal DP during IP2 optim test

# DP Integration controls for tests (can override defaults in solve_dp)
DP_QUAD_LIMIT_TEST = 200
DP_QUAD_EPSABS_TEST = 1e-7
DP_QUAD_EPSREL_TEST = 1e-7
DP_HIGH_ERR_WARN_TEST = 1e-3 # More tolerant warning threshold for tests

print("\n--- Test Configuration ---")
print(f"ALPHA: {ALPHA_TEST}, L_DOMAIN: {L_DOMAIN_TEST:.2f}, T_FINAL: {T_FINAL_TEST}")
print(f"N_TERMS_TEST: {N_TERMS_TEST}, NX_TEST: {NX_TEST}, NT_TEST: {NT_TEST}")
print(f"IP2 N_H_OPTIMIZE: {N_H_OPTIMIZE_TEST}, IP2 Lambda: {IP2_REG_LAMBDA_TEST:.1e}")
print(f"DP Test Int Params: limit={DP_QUAD_LIMIT_TEST}, eps={DP_QUAD_EPSREL_TEST:.1e}")


# --- Define Test Case Functions ---
# Simple, smooth functions satisfying boundary conditions (zero at x=0, L)

# Test Case 1: Simple Sine/Cosine based
phi_test = lambda x: 0.0 * x  # Zero initial displacement
psi_test = lambda x: 0.0 * x  # Zero initial velocity
# True f(t) - smooth, non-zero
f_true_test = lambda t: 1.0 + 0.5 * np.cos(np.pi * t / T_FINAL_TEST)
# True h(x) - combination of eigenfunctions (using helper for consistency)
h_true_test = lambda x: 1.0 * eigenfunctions(1, x, L_DOMAIN_TEST) \
                       - 0.5 * eigenfunctions(2, x, L_DOMAIN_TEST)

print("\n--- Defined Test Functions ---")
print(f"phi(x) = 0")
print(f"psi(x) = 0")
print(f"f_true(t) = 1.0 + 0.5*cos(pi*t/T)")
print(f"h_true(x) = 1.0*X1(x) - 0.5*X2(x)") # Using normalized eigenfunctions


# --- Setup Grids ---
t_test = np.linspace(0, T_FINAL_TEST, NT_TEST)
x_test = np.linspace(0, L_DOMAIN_TEST, NX_TEST)
dt_test = t_test[1] - t_test[0]


# --- Test Suite Execution ---
if __name__ == "__main__":
    overall_start_time = time.time()
    test_errors = {} # Store errors for summary
    test_status = {} # Store pass/fail/skip status

    # === 1. Test Direct Problem (DP) ===
    print("\n--- 1. Testing Direct Problem (solve_dp) ---")
    u_dp_test = None
    dp_success = False
    try:
        # Call the imported solve_dp, passing test-specific integration controls
        u_dp_test = solve_dp(t_test, x_test, ALPHA_TEST, L_DOMAIN_TEST,
                             phi_test, psi_test, f_true_test, h_true_test,
                             N_TERMS_TEST, SERIES_TOL_TEST,
                             # Pass test-specific controls
                             quad_limit=DP_QUAD_LIMIT_TEST,
                             quad_epsabs=DP_QUAD_EPSABS_TEST,
                             quad_epsrel=DP_QUAD_EPSREL_TEST,
                             high_err_warn_threshold=DP_HIGH_ERR_WARN_TEST
                            )

        if u_dp_test is not None and np.all(np.isfinite(u_dp_test)):
            print("Direct Problem solved successfully.")
            # Basic sanity check: Check if BCs are approximately zero
            bc_check_passed = np.allclose(u_dp_test[:, 0], 0, atol=1e-6) and \
                              np.allclose(u_dp_test[:, -1], 0, atol=1e-6)
            # Check if IC is approximately phi
            ic_check_passed = np.allclose(u_dp_test[0, :], phi_test(x_test), atol=1e-6)
            print(f"Basic Checks: BCs zero? {'Yes' if bc_check_passed else 'No'}. IC correct? {'Yes' if ic_check_passed else 'No'}.")
            dp_success = bc_check_passed and ic_check_passed
            test_status['DP'] = 'Pass' if dp_success else 'Fail (Checks)'

            # Optional: Plot DP solution for visual inspection
            plt.figure(figsize=(7, 4))
            T_mesh, X_mesh = np.meshgrid(t_test, x_test, indexing='ij')
            contour = plt.contourf(X_mesh, T_mesh, u_dp_test, 30, cmap='viridis')
            plt.colorbar(contour, label='u(t, x)')
            plt.xlabel('Space x'); plt.ylabel('Time t'); plt.title('DP Test Solution')
            plt.tight_layout(); plt.show(block=False) # Non-blocking plot

        else:
            print("DP solver returned None or non-finite values. Test failed.")
            test_errors['DP'] = 'Solver returned None or NaN/Inf'
            test_status['DP'] = 'Fail (Return)'

    except Exception as e:
        print(f"ERROR during Direct Problem test: {e}")
        test_errors['DP'] = str(e)
        test_status['DP'] = 'Fail (Exception)'
        import traceback
        traceback.print_exc()


    # === 2. Test Inverse Problem 1 (IP1) ===
    print("\n--- 2. Testing Inverse Problem 1 (solve_volterra_ip1_second_kind) ---")
    if dp_success: # Proceed only if DP worked and passed basic checks
        f_estimated_ip1 = None
        ip1_passed = False
        try:
            # 2a. Calculate g(t) data from the DP solution
            print("Calculating g(t) data for IP1...")
            g_vals_ip1_data = np.zeros(NT_TEST)
            h_vec_true_ip1 = h_true_test(x_test) # h(x) is known for IP1
            for i in range(NT_TEST):
                g_vals_ip1_data[i] = np.trapz(h_vec_true_ip1 * u_dp_test[i, :], x_test)
            # Create interpolation function for g(t)
            g_func_ip1_data = lambda t_val: np.interp(t_val, t_test, g_vals_ip1_data, left=g_vals_ip1_data[0], right=g_vals_ip1_data[-1])
            print("g(t) data generated.")

            # 2b. Calculate coefficients needed for IP1 solver
            print("Calculating coefficients for IP1...")
            phi_n_ip1 = fourier_coeffs(phi_test, N_TERMS_TEST, L_DOMAIN_TEST, x_test)
            psi_n_ip1 = fourier_coeffs(psi_test, N_TERMS_TEST, L_DOMAIN_TEST, x_test)
            h_n_ip1 = fourier_coeffs(h_true_test, N_TERMS_TEST, L_DOMAIN_TEST, x_test) # h is known

            # 2c. Run IP1 solver
            print("Running IP1 solver...")
            f_estimated_ip1 = solve_volterra_ip1_second_kind(t_test, ALPHA_TEST, L_DOMAIN_TEST, g_func_ip1_data,
                                                             phi_n_ip1, psi_n_ip1, h_n_ip1, N_TERMS_TEST,
                                                             series_tol=SERIES_TOL_TEST, check_residual=False) # Residual check off for speed

            # 2d. Compare results and calculate errors
            if f_estimated_ip1 is not None and np.all(np.isfinite(f_estimated_ip1)):
                print("IP1 solver finished.")
                f_true_vals_test = f_true_test(t_test)
                l2_err, max_err = calculate_errors(f_estimated_ip1, f_true_vals_test)
                test_errors['IP1_L2Rel'] = l2_err
                test_errors['IP1_MaxAbs'] = max_err
                print(f"  IP1 L2 Relative Error : {l2_err:.4e}")
                print(f"  IP1 Max Absolute Error: {max_err:.4e}")

                # Define pass criteria (e.g., L2 relative error < threshold)
                ip1_passed = l2_err < 1e-2 # Example threshold, adjust as needed
                test_status['IP1'] = 'Pass' if ip1_passed else 'Fail (Accuracy)'

                # Plot comparison
                plt.figure(figsize=(7, 4))
                plt.plot(t_test, f_true_vals_test, 'b-', linewidth=2, label='True f(t)')
                plt.plot(t_test, f_estimated_ip1, 'r--', linewidth=1.5, label='Estimated f(t) (IP1)')
                plt.title(f'IP1 Test: True vs. Estimated f(t) (Err L2={l2_err:.2e})')
                plt.xlabel('Time t'); plt.ylabel('f(t)'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show(block=False)
            else:
                print("IP1 solver returned None or non-finite values. Test failed.")
                test_errors['IP1'] = 'Solver returned None or NaN/Inf'
                test_status['IP1'] = 'Fail (Return)'

        except Exception as e:
            print(f"ERROR during Inverse Problem 1 test: {e}")
            test_errors['IP1'] = str(e)
            test_status['IP1'] = 'Fail (Exception)'
            import traceback
            traceback.print_exc()
    else:
        print("Skipping IP1 test because Direct Problem failed or checks not passed.")
        test_status['IP1'] = 'Skip (DP Fail)'


    # === 3. Test Inverse Problem 2 (IP2) ===
    print("\n--- 3. Testing Inverse Problem 2 (solve_ip2_optimization) ---")
    if dp_success: # Proceed only if DP worked and passed basic checks
        h_estimated_func_ip2 = None
        ip2_passed = False
        try:
            # 3a. Calculate omega(x) data from the DP solution
            print("Calculating omega(x) data for IP2...")
            u_t_dp_test = np.gradient(u_dp_test, dt_test, axis=0)
            f_vals_t_test = f_true_test(t_test)[:, np.newaxis]
            integrand_omega = f_vals_t_test * u_t_dp_test
            try:
                 omega_true_vals_test = spi.simps(integrand_omega, t_test, axis=0)
            except ValueError: # Handle even number of points for simps
                 omega_true_vals_test = np.trapz(integrand_omega, t_test, axis=0)
            print("omega(x) data generated.")

            # 3b. Define initial guess for h coefficients
            initial_h_coeffs_guess_ip2 = np.zeros(N_H_OPTIMIZE_TEST)
            # Optionally add small perturbation or prior knowledge:
            # initial_h_coeffs_guess_ip2[0] = 0.1 # Small guess for first term

            # 3c. Run IP2 solver
            print("Running IP2 solver (Optimization - this may take time)...")
            ip2_result = solve_ip2_optimization(
                            initial_h_coeffs_guess_ip2,
                            t_points=t_test, x_points=x_test, alpha=ALPHA_TEST, L_domain=L_DOMAIN_TEST,
                            f_func=f_true_test, # f(t) is KNOWN for IP2
                            omega_target_on_grid=omega_true_vals_test, # Target data
                            N_terms_h=N_H_OPTIMIZE_TEST,
                            dp_N_terms_optim=IP2_DP_N_TERMS_TEST,
                            dp_series_tol_optim=SERIES_TOL_TEST * 100, # Relaxed tol for speed
                            reg_lambda=IP2_REG_LAMBDA_TEST,
                            maxiter=IP2_MAX_ITER_TEST,
                            # Pass DP controls for internal solver
                            dp_quad_limit=DP_QUAD_LIMIT_TEST,
                            dp_quad_epsabs=DP_QUAD_EPSABS_TEST*10, # Can relax more for optim
                            dp_quad_epsrel=DP_QUAD_EPSREL_TEST*10
                            )

            # 3d. Compare results and calculate errors
            if hasattr(ip2_result, 'final_h_coeffs') and ip2_result.final_h_coeffs is not None:
                print("IP2 optimization finished.")
                optimized_h_coeffs = ip2_result.final_h_coeffs
                print(f"  Success Flag: {ip2_result.success}, Final Cost: {ip2_result.fun:.4e}")

                # Reconstruct h(x) from optimized coefficients
                def h_estimated_func_ip2(x_pts):
                    h_val = np.zeros_like(x_pts, dtype=float)
                    n_max_rec = min(N_H_OPTIMIZE_TEST, len(optimized_h_coeffs))
                    for n in range(n_max_rec):
                         try: h_val += optimized_h_coeffs[n] * eigenfunctions(n + 1, x_pts, L_DOMAIN_TEST)
                         except Exception as e_ef: print(f"Error ef {n+1}:{e_ef}"); return np.nan
                    return h_val

                h_true_vals_test = h_true_test(x_test)
                h_est_vals_test = h_estimated_func_ip2(x_test)

                if np.all(np.isfinite(h_est_vals_test)):
                    l2_err, max_err = calculate_errors(h_est_vals_test, h_true_vals_test)
                    test_errors['IP2_L2Rel'] = l2_err
                    test_errors['IP2_MaxAbs'] = max_err
                    print(f"  IP2 h(x) L2 Relative Error : {l2_err:.4e}")
                    print(f"  IP2 h(x) Max Absolute Error: {max_err:.4e}")
                    # Define pass criteria
                    ip2_passed = l2_err < 5e-2 # Example threshold
                    test_status['IP2'] = 'Pass' if ip2_passed else 'Fail (Accuracy)'

                    # Plot comparison
                    plt.figure(figsize=(7, 4))
                    plt.plot(x_test, h_true_vals_test, 'b-', linewidth=2, label='True h(x)')
                    plt.plot(x_test, h_est_vals_test, 'r--', linewidth=1.5, label=f'Estimated h(x) ({N_H_OPTIMIZE_TEST} terms)')
                    plt.title(f'IP2 Test: True vs. Estimated h(x) (Err L2={l2_err:.2e})')
                    plt.xlabel('Space x'); plt.ylabel('h(x)'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show(block=False)
                else:
                    print("IP2 Estimated h(x) contains non-finite values. Test failed.")
                    test_errors['IP2'] = 'Estimated h(x) contains NaN/Inf'
                    test_status['IP2'] = 'Fail (Return)'

            else:
                print(f"IP2 optimization FAILED to produce usable coefficients.")
                test_errors['IP2'] = f'Optimization failed or no coeffs returned'
                test_status['IP2'] = 'Fail (Optim)'
                if hasattr(ip2_result, 'message'): print(f"  Optimizer message: {ip2_result.message}")


        except Exception as e:
            print(f"ERROR during Inverse Problem 2 test: {e}")
            test_errors['IP2'] = str(e)
            test_status['IP2'] = 'Fail (Exception)'
            import traceback
            traceback.print_exc()

    else:
        print("Skipping IP2 test because Direct Problem failed or checks not passed.")
        test_status['IP2'] = 'Skip (DP Fail)'


    # === Summary ===
    overall_end_time = time.time()
    print("\n--- Test Summary ---")
    print(f"Total testing time: {overall_end_time - overall_start_time:.2f} seconds")

    print("Test Status:")
    for test_name, status in test_status.items():
        print(f"  {test_name}: {status}")

    print("\nError Metrics (if test ran):")
    if not test_errors:
        print("  No significant errors recorded (check plots for accuracy).")
    else:
        for error_name, value in test_errors.items():
            if isinstance(value, float):
                print(f"  {error_name}: {value:.4e}")
            else: # Error message string
                print(f"  {error_name}: {value}")

    # Keep plots open until manually closed
    print("\nClose plot windows to exit.")
    plt.show() # Blocking call to show all plots