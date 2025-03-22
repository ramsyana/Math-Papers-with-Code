import math
import time
from functools import lru_cache

@lru_cache(maxsize=1000)
def stirling2(k, j):
    """Calculate Stirling number of the second kind, S(k, j).
    
    Implements the combinatorial formula for partitioning a set of k labeled objects
    into j non-empty subsets.
    
    Args:
        k (int): Number of elements (non-negative integer)
        j (int): Number of non-empty subsets (non-negative integer)
        
    Returns:
        int: S(k, j), the number of ways to partition k objects into j non-empty subsets
    """
    if j == 0:
        return 1 if k == 0 else 0
    if j > k or j < 0:
        return 0
    if j == 1 or j == k:
        return 1
    
    # Use the explicit formula:
    # S(k, j) = (1/j!) * sum_{i=0}^j (-1)^(j-i) * binomial(j, i) * i^k
    result = 0
    for i in range(j + 1):
        # Sign term: alternates based on parity of (j - i)
        term_sign = (-1) ** (j - i)
        # Binomial coefficient: number of ways to choose i from j
        binom_coeff = math.comb(j, i)
        # Power term: contribution of i raised to k
        power_term = i ** k
        # Accumulate: sign * binomial * power
        result += term_sign * binom_coeff * power_term
    
    return result // math.factorial(j)

def coefficient(k, j):
    """Calculate the coefficient term for the sum of powers formula.
    
    Computes (-1)^(k-j) * j! * S(k, j)
    
    Args:
        k (int): Power
        j (int): Index in the summation
        
    Returns:
        int: The coefficient value
    """
    return ((-1) ** (k - j)) * math.factorial(j) * stirling2(k, j)

def binomial_term(n, j):
    """Calculate the binomial term for the sum of powers formula.
    
    Computes binomial(n+j, j+1)
    
    Args:
        n (int): Upper limit of the sum
        j (int): Index in the summation
        
    Returns:
        int: The binomial term value
    """
    return math.comb(n + j, j + 1)

def sum_powers(n, k):
    """Calculate the sum of powers S_k(n) = 1^k + 2^k + ... + n^k
    
    Uses the formula:
    S_k(n) = sum_{j=0}^k (-1)^(k-j) * j! * S(k, j) * binomial(n+j, j+1)
    
    Args:
        n (int): Upper limit of the sum (non-negative integer)
        k (int): Power (non-negative integer)
        
    Returns:
        int: The sum of the first n positive integers raised to the power k
    """
    # Input validation
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")
    
    if n <= 0:
        return 0
    if k == 0:
        return n
    
    result = 0
    for j in range(k + 1):
        # Use helper functions for clearer structure
        coef = coefficient(k, j)
        binom = binomial_term(n, j)
        
        # Add the term to the result
        result += coef * binom
    
    return result

def sum_powers_alt(n, k):
    """Alternative direct implementation for calculating sum of powers.
    
    Computes 1^k + 2^k + ... + n^k directly for comparison.
    
    Args:
        n (int): Upper limit of the sum
        k (int): Power
        
    Returns:
        int: The sum of the first n positive integers raised to the power k
    """
    return sum(i**k for i in range(1, n + 1))

def verify_known_formulas(n):
    """Verify our implementation against known formulas for specific k values
    
    Args:
        n (int): Upper limit to test
    """
    print(f"For n = {n}:")
    
    # S_0(n) = n (sum of 0th powers = n)
    s0_computed = sum_powers(n, 0)
    s0_formula = n
    print(f"Sum of 0th powers: computed = {s0_computed}, formula = {s0_formula}, match = {s0_computed == s0_formula}")
    
    # S_1(n) = n(n+1)/2
    s1_computed = sum_powers(n, 1)
    s1_formula = n * (n + 1) // 2
    print(f"Sum of 1st powers: computed = {s1_computed}, formula = {s1_formula}, match = {s1_computed == s1_formula}")
    
    # S_2(n) = n(n+1)(2n+1)/6
    s2_computed = sum_powers(n, 2)
    s2_formula = n * (n + 1) * (2 * n + 1) // 6
    print(f"Sum of 2nd powers: computed = {s2_computed}, formula = {s2_formula}, match = {s2_computed == s2_formula}")
    
    # S_3(n) = [n(n+1)/2]^2
    s3_computed = sum_powers(n, 3)
    s3_formula = (n * (n + 1) // 2) ** 2
    print(f"Sum of 3rd powers: computed = {s3_computed}, formula = {s3_formula}, match = {s3_computed == s3_formula}")
    
    # S_4(n) = n(n+1)(2n+1)(3n^2+3n-1)/30
    s4_computed = sum_powers(n, 4)
    s4_formula = n * (n + 1) * (2 * n + 1) * (3 * n ** 2 + 3 * n - 1) // 30
    print(f"Sum of 4th powers: computed = {s4_computed}, formula = {s4_formula}, match = {s4_computed == s4_formula}")
    
    # Add test for n = 0 case
    assert sum_powers(0, 1) == 0, "Failed n=0, k=1"
    assert sum_powers(0, 5) == 0, "Failed n=0, k=5"
    
    # Add test for k = 0 case
    assert sum_powers(5, 0) == 5, "Failed n=5, k=0"
    assert sum_powers(100, 0) == 100, "Failed n=100, k=0"
    
    print("All test cases passed!")

def performance_benchmark(n, k):
    """Compare performance between the formula method and direct summation.
    
    Args:
        n (int): Upper limit of the sum
        k (int): Power
    """
    print(f"\nPerformance benchmark for n={n}, k={k}:")
    
    # Time the formula method
    start = time.time()
    formula_result = sum_powers(n, k)
    formula_time = time.time() - start
    
    # Time the direct summation
    start = time.time()
    direct_result = sum_powers_alt(n, k)
    direct_time = time.time() - start
    
    print(f"Formula method: {formula_result} in {formula_time:.6f}s")
    print(f"Direct summation: {direct_result} in {direct_time:.6f}s")
    print(f"Results match: {formula_result == direct_result}")
    
    if formula_time < direct_time:
        print(f"Formula is {direct_time/formula_time:.2f}x faster")
    else:
        print(f"Direct summation is {formula_time/direct_time:.2f}x faster")

def display_stirling_table(max_k):
    """Display a table of Stirling numbers of the second kind.
    
    Args:
        max_k (int): Maximum k value to display
    """
    print("\nStirling numbers of the second kind S(k, j):")
    print("k\\j", end="")
    for j in range(1, max_k + 1):
        print(f"{j:6}", end="")
    print()
    
    for k in range(1, max_k + 1):
        print(f"{k:3}", end="")
        for j in range(1, k + 1):
            print(f"{stirling2(k, j):6}", end="")
        print()

def main():
    """Main function to demonstrate the code."""
    print("Calculating sums of powers (S_k(n) = 1^k + 2^k + ... + n^k)")
    
    # Example 1: Calculate S_2(10) = 1^2 + 2^2 + ... + 10^2
    n, k = 10, 2
    result = sum_powers(n, k)
    print(f"\nS_{k}({n}) = {result}")
    # Verify with direct calculation
    print(f"Direct calculation: {sum_powers_alt(n, k)}")
    
    # Example 2: Calculate S_3(5) = 1^3 + 2^3 + ... + 5^3
    n, k = 5, 3
    result = sum_powers(n, k)
    print(f"\nS_{k}({n}) = {result}  # Expected: 225")
    # Verify with direct calculation
    print(f"Direct calculation: {sum_powers_alt(n, k)}")
    
    # Verify against known formulas
    print("\nVerifying against known formulas:")
    verify_known_formulas(10)
    
    # Display Stirling numbers table
    display_stirling_table(6)
    
    # Perform performance benchmark
    performance_benchmark(1000, 5)
    
    # Try a large case to demonstrate scalability
    n, k = 100, 5
    print(f"\nLarge case: S_{k}({n}) = {sum_powers(n, k)}")

if __name__ == "__main__":
    main()