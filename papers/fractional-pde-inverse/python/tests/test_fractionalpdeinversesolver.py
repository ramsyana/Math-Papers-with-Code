import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from fractionalpdeinversesolver import ImprovedFractionalPDEInverseSolver

class TestImprovedFractionalPDEInverseSolver(unittest.TestCase):
    """Test cases for the ImprovedFractionalPDEInverseSolver initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        solver = ImprovedFractionalPDEInverseSolver()
        
        # Check default values
        self.assertEqual(solver.N, 1)
        self.assertEqual(solver.M, 100)
        self.assertEqual(solver.K, 100)
        self.assertEqual(solver.T, 1.0)
        self.assertEqual(solver.alpha, 0.5)
        self.assertEqual(solver.Nmax, 10)
        
        # Check grid creation
        self.assertEqual(len(solver.x), 100)
        self.assertAlmostEqual(solver.x[0], 0.0)
        self.assertAlmostEqual(solver.x[-1], np.pi)
        self.assertAlmostEqual(solver.dx, np.pi/99)
        
        # Check temporal grid
        self.assertEqual(len(solver.t), 101)  # K+1 points
        self.assertAlmostEqual(solver.t[0], 0.0)
        self.assertAlmostEqual(solver.t[-1], 1.0)
        self.assertAlmostEqual(solver.dt, 1.0/100)
        
        # Check that other methods were called
        self.assertIsNotNone(solver.eigenfunctions)
        self.assertIsNotNone(solver.eigenvalues)
        self.assertIsNotNone(solver._ml_kernel)
        self.assertEqual(solver._cache, {})

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        solver = ImprovedFractionalPDEInverseSolver(N=2, M=50, K=50, T=2.0, alpha=0.7, Nmax=5)
        
        # Check custom values
        self.assertEqual(solver.N, 2)
        self.assertEqual(solver.M, 50)
        self.assertEqual(solver.K, 50)
        self.assertEqual(solver.T, 2.0)
        self.assertEqual(solver.alpha, 0.7)
        self.assertEqual(solver.Nmax, 5)
        
        # Check grid creation
        self.assertEqual(len(solver.x), 50)
        self.assertAlmostEqual(solver.x[0], 0.0)
        self.assertAlmostEqual(solver.x[-1], np.pi)
        
        # Check temporal grid
        self.assertEqual(len(solver.t), 51)  # K+1 points
        self.assertAlmostEqual(solver.t[0], 0.0)
        self.assertAlmostEqual(solver.t[-1], 2.0)

    def test_invalid_alpha_below_range(self):
        """Test initialization with alpha below valid range."""
        with self.assertRaises(ValueError) as context:
            ImprovedFractionalPDEInverseSolver(alpha=0)
        self.assertEqual(str(context.exception), "alpha must be between 0 and 1")

    def test_invalid_alpha_above_range(self):
        """Test initialization with alpha above valid range."""
        with self.assertRaises(ValueError) as context:
            ImprovedFractionalPDEInverseSolver(alpha=1)
        self.assertEqual(str(context.exception), "alpha must be between 0 and 1")

    def test_invalid_N(self):
        """Test initialization with invalid N."""
        with self.assertRaises(ValueError) as context:
            ImprovedFractionalPDEInverseSolver(N=3)
        self.assertEqual(str(context.exception), "N must be 1 or 2")

    def test_invalid_M(self):
        """Test initialization with invalid M."""
        with self.assertRaises(ValueError) as context:
            ImprovedFractionalPDEInverseSolver(M=0)
        self.assertEqual(str(context.exception), "M, K, and Nmax must be positive integers")

    def test_invalid_K(self):
        """Test initialization with invalid K."""
        with self.assertRaises(ValueError) as context:
            ImprovedFractionalPDEInverseSolver(K=0)
        self.assertEqual(str(context.exception), "M, K, and Nmax must be positive integers")

    def test_invalid_Nmax(self):
        """Test initialization with invalid Nmax."""
        with self.assertRaises(ValueError) as context:
            ImprovedFractionalPDEInverseSolver(Nmax=0)
        self.assertEqual(str(context.exception), "M, K, and Nmax must be positive integers")

    def test_invalid_T(self):
        """Test initialization with invalid T."""
        with self.assertRaises(ValueError) as context:
            ImprovedFractionalPDEInverseSolver(T=0)
        self.assertEqual(str(context.exception), "T must be positive")

    @patch('fractionalpdeinversesolver.ImprovedFractionalPDEInverseSolver._generate_eigenfunctions')
    @patch('fractionalpdeinversesolver.ImprovedFractionalPDEInverseSolver._precompute_ml_kernel')
    @patch('fractionalpdeinversesolver.ImprovedFractionalPDEInverseSolver._precompute_caputo_weights')
    def test_method_calls(self, mock_precompute_caputo, mock_precompute_ml, mock_generate_eigenfunctions):
        """Test that initialization calls the required methods."""
        # Setup mocks
        mock_generate_eigenfunctions.return_value = (np.array([1, 2, 3]), np.array([4, 5, 6]))
        mock_precompute_ml.return_value = np.array([7, 8, 9])
        
        # Initialize solver
        solver = ImprovedFractionalPDEInverseSolver()
        
        # Check that methods were called
        mock_generate_eigenfunctions.assert_called_once()
        mock_precompute_ml.assert_called_once()
        mock_precompute_caputo.assert_called_once()
        
        # Check that the returned values were assigned
        np.testing.assert_array_equal(solver.eigenfunctions, np.array([1, 2, 3]))
        np.testing.assert_array_equal(solver.eigenvalues, np.array([4, 5, 6]))
        np.testing.assert_array_equal(solver._ml_kernel, np.array([7, 8, 9]))

    def test_1D_eigenfunctions(self):
        """Test generation of 1D eigenfunctions."""
        solver = ImprovedFractionalPDEInverseSolver(N=1, Nmax=3)
        
        # Check eigenfunctions and eigenvalues
        self.assertEqual(len(solver.eigenfunctions), 3)
        self.assertEqual(len(solver.eigenvalues), 3)
        
        # Check eigenvalues are correct (should be (n+1)^2)
        np.testing.assert_array_almost_equal(solver.eigenvalues, np.array([1, 4, 9]))
        
        # Check eigenfunctions have correct shape
        self.assertEqual(solver.eigenfunctions[0].shape, (100,))
        
        # Check eigenfunctions are orthogonal
        for i in range(3):
            for j in range(i+1, 3):
                inner_product = np.sum(solver.eigenfunctions[i] * solver.eigenfunctions[j]) * solver.dx
                self.assertAlmostEqual(inner_product, 0.0, places=5)

    def test_2D_eigenfunctions(self):
        """Test generation of 2D eigenfunctions."""
        solver = ImprovedFractionalPDEInverseSolver(N=2, Nmax=2, M=20)
        
        # Check eigenfunctions and eigenvalues
        self.assertEqual(len(solver.eigenvalues), 4)  # 2x2 eigenmodes
        
        # Check eigenvalues are correct (should be (n+1)^2 + (m+1)^2)
        expected_eigenvalues = np.array([2, 5, 5, 8])  # (1,1), (1,2), (2,1), (2,2)
        np.testing.assert_array_almost_equal(solver.eigenvalues, expected_eigenvalues)
        
        # Check eigenfunctions have correct shape
        self.assertEqual(solver.eigenfunctions[0].shape, (20, 20))

    # New tests for core functionality as suggested in the implementation plan
    
    def test_solve_direct_problem(self):
        """Test the direct problem solver with a simple case."""
        solver = ImprovedFractionalPDEInverseSolver(N=1, M=10, K=10, alpha=0.5)
        phi = np.sin(solver.x)  # Initial condition
        h = np.zeros_like(solver.x)  # Source term
        f = np.zeros_like(solver.t)  # Time-dependent source
        
        u = solver.solve_direct_problem(phi, h, f)
        
        # Check solution shape
        self.assertEqual(u.shape, (10, 11))  # M x (K+1)
        
        # Check initial condition is preserved
        np.testing.assert_array_almost_equal(u[:, 0], phi)
        
        # Check solution decays over time (since f=0, h=0)
        self.assertTrue(np.all(np.abs(u[:, -1]) < np.abs(u[:, 0])))

    # Remove duplicate test and keep only one version with a clear name
    def test_solve_inverse_problem_1(self):
        """Test the inverse problem 1 (recovering f(t)) with synthetic data."""
        solver = ImprovedFractionalPDEInverseSolver(N=1, M=10, K=10, alpha=0.5)
        phi = np.sin(solver.x)  # Initial condition
        h = np.zeros_like(solver.x)  # Source term
        f_true = np.sin(solver.t)  # Known time-dependent source
        
        # Generate synthetic data
        u = solver.solve_direct_problem(phi, h, f_true)
        
        # Solve inverse problem to recover f(t)
        f_recovered = solver.solve_inverse_problem_1(u, phi, h)
        
        # Check recovered source is close to true source
        np.testing.assert_array_almost_equal(f_recovered, f_true, decimal=2)

    def test_solve_inverse_problem_2(self):
        """Test the inverse problem 2 (recovering h(x)) with synthetic data."""
        solver = ImprovedFractionalPDEInverseSolver(N=1, M=10, K=10, alpha=0.5)
        phi = np.sin(solver.x)  # Initial condition
        h_true = np.cos(solver.x)  # Known spatial source
        f = np.zeros_like(solver.t)  # Time-dependent source
        
        # Generate synthetic data
        u = solver.solve_direct_problem(phi, h_true, f)
        
        # Solve inverse problem to recover h(x)
        h_recovered = solver.solve_inverse_problem_2(u, phi, f)
        
        # Check recovered source is close to true source
        np.testing.assert_array_almost_equal(h_recovered, h_true, decimal=2)

    def test_solve_direct_problem_analytical(self):
        """Test the direct problem solver against an analytical solution (alpha=1)."""
        solver = ImprovedFractionalPDEInverseSolver(N=1, M=10, K=10, alpha=0.99)  # Using 0.99 instead of 1.0 due to validation
        phi = np.sin(solver.x)  # Initial condition
        h = np.zeros_like(solver.x)  # Spatial source
        f = np.zeros_like(solver.t)  # Temporal source
        
        u_num = solver.solve_direct_problem(phi, h, f)
        
        # Analytical solution for heat equation: u(x,t) = e^(-t) * sin(x)
        u_analytical = np.exp(-solver.t)[:, None] * np.sin(solver.x)[None, :]
        
        # Check numerical solution matches analytical solution
        np.testing.assert_array_almost_equal(u_num, u_analytical, decimal=2)  # Using decimal=2 for near-heat equation

    def test_solve_inverse_problem_1_alpha_variation(self):
        """Test inverse problem 1 accuracy with different alpha values."""
        for alpha in [0.3, 0.7]:
            solver = ImprovedFractionalPDEInverseSolver(N=1, M=10, K=10, alpha=alpha)
            phi = np.sin(solver.x)
            h = np.zeros_like(solver.x)
            f_true = np.sin(solver.t)
            
            u = solver.solve_direct_problem(phi, h, f_true)
            f_recovered = solver.solve_inverse_problem_1(u, phi, h)
            
            # Check recovered source with tighter tolerance
            np.testing.assert_array_almost_equal(f_recovered, f_true, decimal=3)

    def test_solve_direct_problem_2D(self):
        """Test the direct problem solver in 2D."""
        solver = ImprovedFractionalPDEInverseSolver(N=2, M=5, K=5, alpha=0.5)
        phi = np.sin(solver.x)[:, None] * np.sin(solver.x)[None, :]  # 2D initial condition
        h = np.zeros((solver.M, solver.M))  # 2D spatial source
        f = np.zeros_like(solver.t)  # Temporal source
        
        u = solver.solve_direct_problem(phi, h, f)
        
        # Check solution shape
        self.assertEqual(u.shape, (5, 5, 6))  # M x M x (K+1)
        
        # Check initial condition
        np.testing.assert_array_almost_equal(u[:, :, 0], phi)

    def test_solve_inverse_problem_2_2D(self):
        """Test the inverse problem 2 (recovering h(x,y)) in 2D."""
        solver = ImprovedFractionalPDEInverseSolver(N=2, M=5, K=5, alpha=0.5)
        phi = np.sin(solver.x)[:, None] * np.sin(solver.x)[None, :]
        h_true = np.cos(solver.x)[:, None] * np.cos(solver.x)[None, :]
        f = np.zeros_like(solver.t)
        
        u = solver.solve_direct_problem(phi, h_true, f)
        h_recovered = solver.solve_inverse_problem_2(u, phi, f)
        
        # Check recovered source
        np.testing.assert_array_almost_equal(h_recovered, h_true, decimal=2)

    def test_tsvd_regularization(self):
        """Test TSVD regularization with noisy data."""
        solver = ImprovedFractionalPDEInverseSolver(N=1, M=10, K=10, alpha=0.5)
        phi = np.sin(solver.x)
        h = np.zeros_like(solver.x)
        f_true = np.sin(solver.t)
        
        u_clean = solver.solve_direct_problem(phi, h, f_true)
        np.random.seed(42)
        u_noisy = u_clean + np.random.normal(0, 0.01, u_clean.shape)
        
        f_recovered = solver.solve_inverse_problem_1(u_noisy, phi, h, regularization='tsvd', reg_param=5)
        
        # Check solution quality
        np.testing.assert_array_almost_equal(f_recovered, f_true, decimal=1)

    def test_large_grid(self):
        """Test solver with a large grid."""
        solver = ImprovedFractionalPDEInverseSolver(N=1, M=100, K=100, alpha=0.5)
        phi = np.sin(solver.x)
        h = np.zeros_like(solver.x)
        f = np.zeros_like(solver.t)
        
        u = solver.solve_direct_problem(phi, h, f)
        self.assertEqual(u.shape, (100, 101))  # M x (K+1)
        
        # Check initial condition
        np.testing.assert_array_almost_equal(u[:, 0], phi)

    # Remove the first instance of test_estimate_optimal_regularization (lines 447-466)
    # and keep only the more comprehensive version with noise level variation
    
    def test_solve_inverse_problem_both_sources(self):
        """Test inverse problem recovering f(t) with non-zero h(x)."""
        solver = ImprovedFractionalPDEInverseSolver(N=1, M=10, K=10, alpha=0.5)
        phi = np.sin(solver.x)
        h_true = np.cos(solver.x)  # Non-zero spatial source
        f_true = np.sin(solver.t)  # Non-zero temporal source
        
        u = solver.solve_direct_problem(phi, h_true, f_true)
        f_recovered = solver.solve_inverse_problem_1(u, phi, h_true)
        
        np.testing.assert_array_almost_equal(f_recovered, f_true, decimal=2)
    
    def test_performance_large_grid(self):
        """Test performance with a large grid."""
        import time
        solver = ImprovedFractionalPDEInverseSolver(N=1, M=200, K=200, alpha=0.5)
        phi = np.sin(solver.x)
        h = np.zeros_like(solver.x)
        f = np.zeros_like(solver.t)
        
        start_time = time.time()
        u = solver.solve_direct_problem(phi, h, f)
        elapsed_time = time.time() - start_time
        
        print(f"Performance test (M=200, K=200): {elapsed_time:.4f} seconds")
        self.assertEqual(u.shape, (200, 201))
        
        # Check for numerical stability
        self.assertFalse(np.any(np.isnan(u)))
        self.assertFalse(np.any(np.isinf(u)))
    
    # Modify the existing test_extreme_alpha to include numerical stability checks
    def test_extreme_alpha(self):
        """Test solver with alpha close to boundaries."""
        # Test with alpha close to 0
        solver_small = ImprovedFractionalPDEInverseSolver(N=1, M=5, K=5, alpha=0.01)
        
        # Test with alpha close to 1
        solver_large = ImprovedFractionalPDEInverseSolver(N=1, M=5, K=5, alpha=0.99)
        
        # Check both solvers initialize correctly
        self.assertEqual(solver_small.alpha, 0.01)
        self.assertEqual(solver_large.alpha, 0.99)
        
        # Test basic functionality for both
        phi = np.sin(solver_small.x)
        h = np.zeros_like(solver_small.x)
        f = np.zeros_like(solver_small.t)
        
        u_small = solver_small.solve_direct_problem(phi, h, f)
        u_large = solver_large.solve_direct_problem(phi, h, f)
        
        # Solutions should be different due to different alpha values
        self.assertTrue(np.any(np.abs(u_small - u_large) > 1e-10))
        
        # Check for numerical stability in both extreme cases
        self.assertFalse(np.any(np.isnan(u_small)))
        self.assertFalse(np.any(np.isinf(u_small)))
        self.assertFalse(np.any(np.isnan(u_large)))
        self.assertFalse(np.any(np.isinf(u_large)))
    
    # Modify the existing test_large_nmax to include numerical stability checks
    def test_large_nmax(self):
        """Test solver with large Nmax to ensure scalability."""
        solver = ImprovedFractionalPDEInverseSolver(N=1, M=10, K=10, Nmax=20)
        
        # Check eigenfunction count
        self.assertEqual(len(solver.eigenvalues), 20)
        
        # Test basic functionality
        phi = np.sin(solver.x)
        h = np.zeros_like(solver.x)
        f = np.zeros_like(solver.t)
        
        u = solver.solve_direct_problem(phi, h, f)
        self.assertEqual(u.shape, (10, 11))  # M x (K+1)
        
        # Check for numerical stability
        self.assertFalse(np.any(np.isnan(u)))
        self.assertFalse(np.any(np.isinf(u)))

    # Add new suggested tests
    def test_l1_regularization(self):
        """Test L1 regularization with noisy data."""
        solver = ImprovedFractionalPDEInverseSolver(N=1, M=10, K=10, alpha=0.5)
        phi = np.sin(solver.x)
        h = np.zeros_like(solver.x)
        f_true = np.sin(solver.t)
        
        u_clean = solver.solve_direct_problem(phi, h, f_true)
        np.random.seed(42)
        u_noisy = u_clean + np.random.normal(0, 0.01, u_clean.shape)
        
        f_recovered = solver.solve_inverse_problem_1(u_noisy, phi, h, regularization='l1', reg_param=0.01)
        
        # Check solution quality
        np.testing.assert_array_almost_equal(f_recovered, f_true, decimal=1)
    
    def test_estimate_optimal_regularization(self):
        """Test optimal regularization parameter estimation."""
        solver = ImprovedFractionalPDEInverseSolver(N=1, M=10, K=10, alpha=0.5)
        phi = np.sin(solver.x)
        h = np.zeros_like(solver.x)
        f_true = np.sin(solver.t)
        
        u_clean = solver.solve_direct_problem(phi, h, f_true)
        np.random.seed(42)
        u_noisy = u_clean + np.random.normal(0, 0.01, u_clean.shape)
        
        reg_param = solver.estimate_optimal_regularization(u_noisy, phi, h, method='l_curve')
        self.assertGreater(reg_param, 0)
        
        f_recovered = solver.solve_inverse_problem_1(u_noisy, phi, h, regularization='tikhonov', reg_param=reg_param)
        np.testing.assert_array_almost_equal(f_recovered, f_true, decimal=1)
    
    def test_boundary_conditions(self):
        """Test that the solution satisfies Dirichlet boundary conditions."""
        solver = ImprovedFractionalPDEInverseSolver(N=1, M=10, K=10, alpha=0.5)
        phi = np.sin(solver.x)  # sin(x) is 0 at x=0 and x=π
        h = np.zeros_like(solver.x)
        f = np.zeros_like(solver.t)
        
        u = solver.solve_direct_problem(phi, h, f)
        
        # Check boundary values are close to zero
        self.assertTrue(np.all(np.abs(u[0, :]) < 1e-10))
        self.assertTrue(np.all(np.abs(u[-1, :]) < 1e-10))
        
        # Additional test: Even with non-zero sources, boundaries should remain zero
        h_nonzero = np.cos(solver.x)  # This still satisfies h(0)=h(π)=0
        f_nonzero = np.sin(solver.t)
        
        u_nonzero = solver.solve_direct_problem(phi, h_nonzero, f_nonzero)
        self.assertTrue(np.all(np.abs(u_nonzero[0, :]) < 1e-10))
        self.assertTrue(np.all(np.abs(u_nonzero[-1, :]) < 1e-10))

    def test_estimate_optimal_regularization(self):
        """Test optimal regularization parameter estimation."""
        solver = ImprovedFractionalPDEInverseSolver(N=1, M=10, K=10, alpha=0.5)
        phi = np.sin(solver.x)
        h = np.zeros_like(solver.x)
        f_true = np.sin(solver.t)
        
        u_clean = solver.solve_direct_problem(phi, h, f_true)
        
        # Test with different noise levels
        for noise_level in [0.01, 0.05]:
            np.random.seed(42)
            u_noisy = u_clean + np.random.normal(0, noise_level, u_clean.shape)
            
            reg_param = solver.estimate_optimal_regularization(u_noisy, phi, h, method='l_curve')
            self.assertGreater(reg_param, 0)
            
            f_recovered = solver.solve_inverse_problem_1(u_noisy, phi, h, regularization='tikhonov', reg_param=reg_param)
            
            # Higher noise should allow less precision in recovery
            decimal_places = 1 if noise_level <= 0.01 else 0
            np.testing.assert_array_almost_equal(f_recovered, f_true, decimal=decimal_places)

    def test_large_grid(self):
        """Test solver with a large grid."""
        solver = ImprovedFractionalPDEInverseSolver(N=1, M=100, K=100, alpha=0.5)
        phi = np.sin(solver.x)
        h = np.zeros_like(solver.x)
        f = np.zeros_like(solver.t)
        
        # Basic timing benchmark
        import time
        start_time = time.time()
        u = solver.solve_direct_problem(phi, h, f)
        elapsed_time = time.time() - start_time
        
        # Check solution shape and initial condition
        self.assertEqual(u.shape, (100, 101))  # M x (K+1)
        np.testing.assert_array_almost_equal(u[:, 0], phi)
        
        # Check for numerical stability (no NaNs or infinities)
        self.assertFalse(np.any(np.isnan(u)))
        self.assertFalse(np.any(np.isinf(u)))
        
        # Log performance info (optional)
        print(f"\nLarge grid solution time: {elapsed_time:.4f} seconds")
    
    def test_non_orthogonal_source_recovery(self):
        """Test inverse problem with h(x) not in eigenfunction subspace."""
        solver = ImprovedFractionalPDEInverseSolver(N=1, M=20, K=50)
        phi = np.zeros_like(solver.x)
        h_true = solver.x * (np.pi - solver.x)  # Not a pure eigenfunction
        f = np.zeros_like(solver.t)
        
        u = solver.solve_direct_problem(phi, h_true, f)
        h_recovered = solver.solve_inverse_problem_2(u, phi, f)
        
        error = np.linalg.norm(h_recovered - h_true) / np.linalg.norm(h_true)
        self.assertLess(error, 0.1)  # Allow 10% error due to spectral approximation

    def test_convergence_with_grid_refinement(self):
        """Test direct problem solution converges as M, K increase."""
        errors = []
        for M, K in [(10, 10), (20, 20), (40, 40)]:
            solver = ImprovedFractionalPDEInverseSolver(N=1, M=M, K=K, alpha=0.99)  # Near heat equation
            phi = np.sin(solver.x)
            h = np.zeros_like(solver.x)
            f = np.zeros_like(solver.t)
            
            u = solver.solve_direct_problem(phi, h, f)
            u_analytical = np.exp(-solver.t)[:, None] * np.sin(solver.x)[None, :]
            errors.append(np.max(np.abs(u - u_analytical.T)))  # Transpose to match dimensions
        
        # Check error decreases by at least 30% each refinement
        self.assertLess(errors[1], 0.7 * errors[0])
        self.assertLess(errors[2], 0.7 * errors[1])

    def test_inverse_problem_1_with_noise(self):
        """Test inverse problem 1 with noise using separate solver instances."""
        # Create two separate solver instances
        solver_gen = ImprovedFractionalPDEInverseSolver(N=1, M=10, K=10)
        solver_inv = ImprovedFractionalPDEInverseSolver(N=1, M=10, K=10)  # Separate instance
        
        # Generate data
        phi = np.sin(solver_gen.x)
        h = np.zeros_like(solver_gen.x)
        f_true = np.sin(solver_gen.t)
        
        # Generate clean and noisy data
        u_clean = solver_gen.solve_direct_problem(phi, h, f_true)
        np.random.seed(42)  # For reproducibility
        u_noisy = u_clean + 0.01 * np.random.randn(*u_clean.shape)
        
        # Recover with solver_inv using middle time point data
        f_recovered = solver_inv.solve_inverse_problem_1(u_noisy, phi, h)
        
        # Calculate relative error
        error = np.linalg.norm(f_recovered - f_true) / np.linalg.norm(f_true)
        self.assertLess(error, 0.1)  # Allow 10% error due to noise

    # Add parameterized tests for alpha variation
    def test_alpha_variation_direct_problem(self):
        """Test direct problem with different alpha values."""
        for alpha in [0.3, 0.7, 0.99]:
            with self.subTest(alpha=alpha):
                solver = ImprovedFractionalPDEInverseSolver(N=1, M=10, K=10, alpha=alpha)
                phi = np.sin(solver.x)
                h = np.zeros_like(solver.x)
                f = np.zeros_like(solver.t)
                
                u = solver.solve_direct_problem(phi, h, f)
                
                # Basic checks that should pass for any alpha
                self.assertEqual(u.shape, (10, 11))
                np.testing.assert_array_almost_equal(u[:, 0], phi)
                self.assertFalse(np.any(np.isnan(u)))
                self.assertFalse(np.any(np.isinf(u)))
                
                # For alpha near 1, compare with analytical solution
                if alpha > 0.9:
                    u_analytical = np.exp(-solver.t)[:, None] * np.sin(solver.x)[None, :]
                    np.testing.assert_array_almost_equal(u, u_analytical.T, decimal=1)
    
if __name__ == '__main__':
    unittest.main()