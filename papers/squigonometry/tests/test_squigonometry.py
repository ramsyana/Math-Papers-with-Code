import unittest
import math

# Import the functions from the original module
from ..squigonometry import (
    precompute_factorials, 
    compute_maclaurin_coefficients, 
    evaluate_squigonometric
)

class TestSquigonometricFunctions(unittest.TestCase):
    def setUp(self):
        # Precompute factorials for use across tests
        self.max_k = 100
        self.factorials = precompute_factorials(self.max_k)

    def test_precompute_factorials(self):
        """Test factorial precomputation."""
        # Check first few known factorial values
        expected_factorials = [1, 1, 2, 6, 24, 120]
        for k, expected in enumerate(expected_factorials):
            self.assertEqual(self.factorials[k], expected, 
                             f"Factorial for {k} is incorrect")
        
        # Check length of factorials list
        self.assertEqual(len(self.factorials), self.max_k + 1, 
                         "Incorrect number of precomputed factorials")

    def test_compute_maclaurin_coefficients(self):
        """Test MacLaurin coefficients computation."""
        # Test cases for different parameters
        test_cases = [
            # p, m, n, J, expected length
            (4, 1, 0, 10, 11),  # cosquine
            (4, 0, 1, 10, 11),  # squine
            (3, 2, 0, 5, 6),    # Another test case
        ]

        for p, m, n, J, expected_length in test_cases:
            coeffs = compute_maclaurin_coefficients(m, n, p, J, self.factorials)
            
            # Check coefficient list length
            self.assertEqual(len(coeffs), expected_length, 
                             f"Incorrect number of coefficients for p={p}, m={m}, n={n}")
            
            # Check first coefficient is positive
            self.assertGreater(abs(coeffs[0]), 0, 
                               "First coefficient should be non-zero")

    def test_evaluate_squigonometric(self):
        """Test squigonometric function evaluation."""
        # Test cases with known or expected properties
        test_cases = [
            # p, n, t, (expected condition)
            (4, 0, 0, lambda x: abs(x - 1.0) < 1e-10),  # cosquine at t=0
            (4, 1, 0, lambda x: abs(x) < 1e-10),        # squine at t=0
        ]

        for p, n, t, condition in test_cases:
            # Compute coefficients
            coeffs = compute_maclaurin_coefficients(1 if n == 0 else 0, n, p, 10, self.factorials)
            
            # Evaluate function
            result = evaluate_squigonometric(coeffs, p, n, t)
            
            # Check condition
            self.assertTrue(condition(result), 
                            f"Unexpected result for p={p}, n={n}, t={t}")

    def test_pythagorean_identity(self):
        """Test Pythagorean-like identity for squigonometric functions."""
        # Parameters for testing
        test_cases = [
            (4, 0.5),   # Test at t=0.5 with p=4
            (3, 0.3),   # Test with different p
            (5, 0.7),
        ]

        for p, t in test_cases:
            # Compute cosquine and squine coefficients
            cq_coeffs = compute_maclaurin_coefficients(1, 0, p, 10, self.factorials)
            sq_coeffs = compute_maclaurin_coefficients(0, 1, p, 10, self.factorials)

            # Evaluate functions
            cq_t = evaluate_squigonometric(cq_coeffs, p, 0, t)
            sq_t = evaluate_squigonometric(sq_coeffs, p, 1, t)

            # Compute identity value
            identity_value = abs(sq_t) ** p + abs(cq_t) ** p

            # Check if identity is close to 1
            self.assertAlmostEqual(identity_value, 1.0, 
                                   places=2, 
                                   msg=f"Pythagorean identity failed for p={p}, t={t}")

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with minimal parameters
        p_values = [2, 3, 4, 5]
        for p in p_values:
            # Minimal J and m, n values
            coeffs = compute_maclaurin_coefficients(0, 0, p, 1, self.factorials)
            self.assertTrue(len(coeffs) > 0, f"No coefficients generated for p={p}")

    def test_numerical_stability(self):
        """Check numerical stability for different parameters."""
        test_cases = [
            (4, 0, 0.1),   # Small t
            (4, 1, 0.01),  # Very small t
            (4, 0, 1.0),   # Large t within convergence
        ]

        for p, n, t in test_cases:
            # Compute coefficients
            coeffs = compute_maclaurin_coefficients(1 if n == 0 else 0, n, p, 10, self.factorials)
            
            # Evaluate function
            result = evaluate_squigonometric(coeffs, p, n, t)
            
            # Check that result is a valid number
            self.assertTrue(math.isfinite(result), 
                            f"Non-finite result for p={p}, n={n}, t={t}")

if __name__ == '__main__':
    unittest.main()