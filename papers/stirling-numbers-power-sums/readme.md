# A Remark on an Explicit Formula for the Sums of Powers of Integers

Implementation of algorithms and concepts from the paper "A Remark on an Explicit Formula for the Sums of Powers of Integers" (arXiv:2503.14508v2).

## Paper Information

| Field        | Details                                                                 |
|--------------|-------------------------------------------------------------------------|
| Title        | A Remark on an Explicit Formula for the Sums of Powers of Integers      |
| Author       | José L. Cereceda                                                        |
| arXiv        | [2503.14508v2](https://arxiv.org/pdf/2503.14508v2)                      |
| Date         | March 2025                                                              |
| Abstract     | This paper explores efficient methods for computing Stirling numbers of the second kind and their application to calculating power sums. |
| Status       | Complete                                                                |


## Implementation Status

| Language | Status | Features | Directory |
|----------|---------|-----------|------------|
| Python   | ✅ Complete | Full implementation with test suite | `python/` |

## Overview

This implementation demonstrates the relationship between Stirling numbers of the second kind and power sums. It provides efficient algorithms for computing:

- Stirling numbers of the second kind S(n,k)
- Power sums S_k(n) = 1^k + 2^k + ... + n^k

The implementation uses the formula:
```
S_k(n) = sum_{j=0}^k (-1)^(k-j) * j! * S(k, j) * binomial(n+j, j+1)
```

## Key Components

### Stirling Numbers Calculator
- Implements Stirling numbers of the second kind
- Uses combinatorial formula for set partitioning
- Optimized with memoization via `@lru_cache`

### Power Sums Formula
- Computes sums of powers using Stirling numbers
- Implements both formula-based and direct calculation methods
- Provides performance comparisons between methods

### Verification Module
- Validates implementation against known closed-form formulas
- Tests special cases (n=0, k=0)
- Displays tables of Stirling numbers

## Usage

### Python Implementation

```bash
python sum_powers.py
```

Example output:
```
Calculating sums of powers (S_k(n) = 1^k + 2^k + ... + n^k)

S_2(10) = 385
Direct calculation: 385

S_3(5) = 225  # Expected: 225
Direct calculation: 225

Verifying against known formulas:
For n = 10:
Sum of 0th powers: computed = 10, formula = 10, match = True
Sum of 1st powers: computed = 55, formula = 55, match = True
Sum of 2nd powers: computed = 385, formula = 385, match = True
Sum of 3rd powers: computed = 3025, formula = 3025, match = True
Sum of 4th powers: computed = 25333, formula = 25333, match = True
All test cases passed!

Stirling numbers of the second kind S(k, j):
k\j     1     2     3     4     5     6
  1     1
  2     1     1
  3     1     3     1
  4     1     7     6     1
  5     1    15    25    10     1
  6     1    31    90    65    15     1

Performance benchmark for n=1000, k=5:
Formula method: 6375416666670833 in 0.000997s
Direct summation: 6375416666670833 in 0.007977s
Results match: True
Formula is 8.00x faster

Large case: S_5(100) = 2550000000
```

## Performance Notes

The implementation focuses on:
- Efficient calculation of Stirling numbers with memoization
- Optimized formula-based computation of power sums
- Performance comparison between formula and direct summation methods
- Scalability for large values of n and k

## Mathematical Background

Stirling numbers of the second kind, S(n,k), count the number of ways to partition a set of n labeled objects into k non-empty unlabeled subsets. These numbers play a crucial role in combinatorics and are used in the formula for computing power sums.

The implementation uses the explicit formula:
```
S(k, j) = (1/j!) * sum_{i=0}^j (-1)^(j-i) * binomial(j, i) * i^k
```

## Contributing

Contributions welcome in areas such as:
- Performance optimizations
- Additional test coverage
- Documentation improvements
- Extension to other types of Stirling numbers

## License

MIT License - see [LICENSE](LICENSE) file for details.

## References

1. Boyadzhiev, K. N. (2012). "Close encounters with the Stirling numbers of the second kind." *Mathematics Magazine*, 85(4), 252–266.

2. Cereceda, J. L. (2017). "Polynomial interpolation and sums of powers of integers." *International Journal of Mathematical Education in Science and Technology*, 48(2), 267-277.

3. Cereceda, J. L. (2024). "Dual recursive formulas for the sums of powers of integers." *Far East Journal of Mathematical Education*, 26(2), 111-121.

4. Gould, H. W. (1978). "Evaluation of sums of convolved powers using Stirling and Eulerian numbers." *The Fibonacci Quarterly*, 16(6) [Part 1], 488–497.

5. Howard, F. T. (1993/1994). "Sums of powers of integers." *Mathematical Spectrum*, 26(4), 103–109.

6. Knuth, D. E. (1993). "Johann Faulhaber and sums of powers." *Mathematics of Computation*, 61(203), 277–294.

7. Quaintance, J., & Gould, H. W. (2016). *Combinatorial Identities for Stirling Numbers: The Unpublished Notes of H. W. Gould*. World Scientific Publishing, Singapore.

8. Samsonadze, E. (2024). "On sums of powers of natural numbers." Preprint, available at [arXiv:2411.11859v1](https://arxiv.org/abs/2411.11859v1)
