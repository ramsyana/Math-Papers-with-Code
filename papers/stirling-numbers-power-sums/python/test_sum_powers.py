import unittest
import math
import time
import random
from sum_powers import sum_powers, sum_powers_alt, stirling2

class TestSumPowers(unittest.TestCase):
    
    def test_small_cases(self):
        """Test small input cases with known answers."""
        # n = 0
        self.assertEqual(sum_powers(0, 0), 0)
        self.assertEqual(sum_powers(0, 1), 0)
        self.assertEqual(sum_powers(0, 5), 0)
        
        # n = 1
        self.assertEqual(sum_powers(1, 0), 1)
        self.assertEqual(sum_powers(1, 1), 1)
        self.assertEqual(sum_powers(1, 2), 1)
        self.assertEqual(sum_powers(1, 10), 1)
        
        # k = 0
        self.assertEqual(sum_powers(5, 0), 5)
        self.assertEqual(sum_powers(100, 0), 100)
    
    def test_known_formulas(self):
        """Test against known mathematical formulas for specific values of k."""
        # Test several values of n
        for n in [5, 10, 20, 50, 100]:
            # k = 1: arithmetic series: n(n+1)/2
            self.assertEqual(sum_powers(n, 1), n * (n + 1) // 2)
            
            # k = 2: sum of squares: n(n+1)(2n+1)/6
            self.assertEqual(sum_powers(n, 2), n * (n + 1) * (2 * n + 1) // 6)
            
            # k = 3: sum of cubes: [n(n+1)/2]^2
            self.assertEqual(sum_powers(n, 3), (n * (n + 1) // 2) ** 2)
            
            # k = 4: sum of 4th powers: n(n+1)(2n+1)(3n^2+3n-1)/30
            self.assertEqual(sum_powers(n, 4), 
                            n * (n + 1) * (2 * n + 1) * (3 * n ** 2 + 3 * n - 1) // 30)
    
    def test_against_direct_calculation(self):
        """Test formula method against direct calculation."""
        # Try various combinations of n and k
        test_cases = [
            (10, 5), (15, 3), (20, 7), (30, 4), (50, 6)
        ]
        
        for n, k in test_cases:
            formula_result = sum_powers(n, k)
            direct_result = sum_powers_alt(n, k)
            self.assertEqual(formula_result, direct_result,
                            f"Failed for n={n}, k={k}: formula={formula_result}, direct={direct_result}")
    
    def test_stirling_numbers(self):
        """Test Stirling numbers of the second kind calculation."""
        # Known Stirling number values
        known_values = {
            (1, 1): 1,
            (2, 1): 1, (2, 2): 1,
            (3, 1): 1, (3, 2): 3, (3, 3): 1,
            (4, 1): 1, (4, 2): 7, (4, 3): 6, (4, 4): 1,
            (5, 1): 1, (5, 2): 15, (5, 3): 25, (5, 4): 10, (5, 5): 1
        }
        
        for (k, j), expected in known_values.items():
            self.assertEqual(stirling2(k, j), expected, f"Failed for S({k}, {j})")
        
        # Test boundary conditions
        self.assertEqual(stirling2(0, 0), 1)
        self.assertEqual(stirling2(5, 0), 0)
        self.assertEqual(stirling2(5, 6), 0)
        self.assertEqual(stirling2(5, -1), 0)
    
    def test_input_validation(self):
        """Test input validation for the sum_powers function."""
        # Test negative n
        with self.assertRaises(ValueError):
            sum_powers(-1, 5)
        
        # Test negative k
        with self.assertRaises(ValueError):
            sum_powers(10, -1)
        
        # Test non-integer inputs - currently these raise ValueError in your implementation
        with self.assertRaises(ValueError):
            sum_powers(10.5, 2)
        
        with self.assertRaises(ValueError):
            sum_powers(10, 2.5)
    
    def test_large_inputs(self):
        """Test with larger inputs that might cause performance issues."""
        # These should still be calculated quickly with the formula
        large_cases = [
            (200, 10),
            (500, 5),
            (1000, 3)
        ]
        
        for n, k in large_cases:
            formula_result = sum_powers(n, k)
            # We don't verify with direct calculation as it would be too slow
            # Just ensure we get a result without error
            self.assertIsInstance(formula_result, int)
    
    def test_lru_cache_effectiveness(self):
        """Test the effectiveness of the LRU cache for Stirling numbers."""
        # First call should be slow, subsequent calls should be faster
        start_time = time.time()
        stirling2(20, 10)  # A moderately complex calculation
        first_call_time = time.time() - start_time
        
        # Second call should use cached result
        start_time = time.time()
        stirling2(20, 10)
        second_call_time = time.time() - start_time
        
        # The second call should be significantly faster
        self.assertLess(second_call_time, first_call_time / 2)
    
    def test_performance_comparison(self):
        """Compare performance between formula and direct calculation."""
        # Choose values where both methods should be reasonably fast
        n, k = 100, 5
        
        # Time the formula method
        start_time = time.time()
        formula_result = sum_powers(n, k)
        formula_time = time.time() - start_time
        
        # Time the direct summation
        start_time = time.time()
        direct_result = sum_powers_alt(n, k)
        direct_time = time.time() - start_time
        
        # Verify results match
        self.assertEqual(formula_result, direct_result)
        
        # Print performance comparison
        print(f"\nPerformance comparison for n={n}, k={k}:")
        print(f"  Formula method: {formula_time:.6f}s")
        print(f"  Direct summation: {direct_time:.6f}s")
        
        # For these values, the formula method should be faster
        # but we don't assert this as it depends on the machine
        if formula_time < direct_time:
            print(f"  Formula method is {direct_time/formula_time:.2f}x faster")
        else:
            print(f"  Direct summation is {formula_time/direct_time:.2f}x faster")
    
    def test_random_cases(self):
        """Test with random inputs (within reasonable bounds)."""
        for _ in range(5):
            n = random.randint(5, 50)
            k = random.randint(1, 10)
            
            formula_result = sum_powers(n, k)
            direct_result = sum_powers_alt(n, k)
            
            self.assertEqual(formula_result, direct_result,
                            f"Failed for random case n={n}, k={k}")

def main():
    unittest.main()

if __name__ == "__main__":
    main()