"""
Relative Sizes of Iterated Sumsets - Python Implementation
=======================================================

This module implements the concepts from the paper:
"Relative Sizes of Iterated Sumsets" by Noah Kravitz (arXiv:2412.18598)

The implementation demonstrates Nathanson's question about the existence of finite
sets A, B ⊆ Z and natural numbers h₁ < h₂ < h₃ such that:
|h₁A| < |h₁B|, |h₂B| < |h₂A|, and |h₃A| < |h₃B|

Key Components:
--------------
- ArithmeticSet: Class for managing sets and computing h-fold sumsets
- generate_alpha_sequences: Function to generate compatible alpha sequences
- main: Demonstration of the theorem with example parameters

Usage:
------
Run directly with: python iterated_sumsets.py

Author: Ramsyana (ramsyana@mac.com)
Date: December 2024
License: MIT
"""

from typing import List, Set

class ArithmeticSet:
    def __init__(self, M: int, alphas: List[int]):
        self.M = M
        self.alphas = sorted(alphas)

    def interval_at_scale(self, alpha: int) -> Set[int]:
        M_alpha = self.M ** alpha
        return {M_alpha * i for i in range(self.M + 1)}

    def get_elements(self) -> Set[int]:
        result = set()
        for alpha in self.alphas:
            result.update(self.interval_at_scale(alpha))
        return result

    def h_fold_sumset(self, h: int) -> Set[int]:
        elements = self.get_elements()
        result = {0}

        for _ in range(h):
            new_result = set()
            for x in result:
                for y in elements:
                    new_result.add(x + y)
            result = new_result

        return result

def generate_alpha_sequences(n: int, R: int) -> List[List[int]]:
    # Using smaller values for gamma
    def gamma(r: int) -> int:
        return r + 1  # Simplified gamma function

    result = []

    for k in range(1, n + 1):
        sequence = set()
        for s in range(1, R + 1):
            sequence.add(0)
            sequence.add(k * gamma(s))  # Simplified calculation
        result.append(sorted(list(sequence)))

    return result

def main():
    print("Starting Nathanson's sumsets computation...")

    # Parameters
    n, R = 2, 3  # Two sets, three permutations
    M = 3        # Small base value
    print(f"\nParameters: n={n}, R={R}, M={M}")

    # Generate alpha sequences
    alpha_sequences = generate_alpha_sequences(n, R)
    print(f"\nGenerated alpha sequences: {alpha_sequences}")

    # Create the sets
    sets = [ArithmeticSet(M, alphas) for alphas in alpha_sequences]

    # Print the actual elements of each set
    print("\nSet elements:")
    for i, s in enumerate(sets, 1):
        elements = sorted(list(s.get_elements()))
        print(f"Set A{i}: {elements}")

    # Generate h values - using smaller values for demonstration
    hs = [1, 2, 3]  # Simplified h values
    print(f"\nUsing h values: {hs}")

    # Print sumset sizes
    print("\nSumset sizes:")
    for i, h in enumerate(hs, 1):
        sizes = [len(s.h_fold_sumset(h)) for s in sets]
        print(f"For h = {h}:")
        print(f"  |{h}A1| = {sizes[0]}")
        print(f"  |{h}A2| = {sizes[1]}")

if __name__ == "__main__":
    print("Program started")
    main()
    print("Program finished")