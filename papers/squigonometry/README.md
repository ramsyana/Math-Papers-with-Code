# Squigonometry

This project provides a Python implementation of squigonometric functions, specifically the squine and cosquine, based on their MacLaurin series expansions. Squigonometric functions generalize trigonometric functions for p-circles, defined by the equation `|x|^p + |y|^p = 1`. For `p=2`, these functions reduce to the standard sine and cosine.

## Introduction

Squigonometric functions are used to parametrize p-circles, which are shapes defined by the equation `|x|^p + |y|^p = 1`. When `p=2`, this shape is a circle, and the squigonometric functions become the familiar sine and cosine functions. For other values, such as `p=4`, the shape is known as a squircle.

This implementation computes the squine and cosquine functions using their MacLaurin series expansions. The series coefficients are calculated with a dynamic programming approach based on the recursion relation from the research paper.

## Mathematical Background

According to Van Lith's research [1], squigonometry studies "imperfect circles" - geometric planar shapes that generalize circles. These shapes, called p-circles or squircles, are defined by the equation `|x|^p + |y|^p = 1` for `1 ≤ p < ∞`.

The squigonometric functions (squine and cosquine) parametrize these p-circles. While there are multiple ways to define these parametrizations (by area, arc length, or angle), this implementation uses the areal parametrization. The arcsquine function, which is fundamental to the theory, is defined as: arcsq(x) = ∫₀ˣ (1 - uᵖ)^(1/p - 1) du

The squine (sq t) and cosquine (cq t) functions are the inverses of these parametrizations, and their MacLaurin series expansions form the basis of our numerical implementation.

## Installation

To use this code, you need Python installed on your system. The implementation relies solely on standard Python libraries, so no additional dependencies are required.

1. Clone the repository or download the code files.
2. Ensure the following files are in the tests directory:
   - `squigonometry.py`
   - `tests/test_squigonometry.py`

## Usage

The core functions are defined in `squigonometry.py`. Below is an example of how to compute the 4-cosquine and 4-squine at `t=0.5`:

```python
import squigonometry

# Precompute factorials
max_k = 100
factorials = squigonometry.precompute_factorials(max_k)

# Set parameters
p = 4
J = 10

# Compute cosquine (cq) coefficients: m=1, n=0
cq_coeffs = squigonometry.compute_maclaurin_coefficients(m=1, n=0, p=p, J=J, factorials=factorials)

# Compute squine (sq) coefficients: m=0, n=1
sq_coeffs = squigonometry.compute_maclaurin_coefficients(m=0, n=1, p=p, J=J, factorials=factorials)

# Evaluate at t=0.5
t = 0.5
cq_t = squigonometry.evaluate_squigonometric(cq_coeffs, p, 0, t)
sq_t = squigonometry.evaluate_squigonometric(sq_coeffs, p, 1, t)

print(f"cq_{p}({t}) ≈ {cq_t}")
print(f"sq_{p}({t}) ≈ {sq_t}")

# Verify Pythagorean identity
identity_value = abs(sq_t) ** p + abs(cq_t) ** p
print(f"|sq|^{p} + |cq|^{p} ≈ {identity_value}")
```

## References

[1] Van Lith, Bart S. "Derivative Polynomials and Infinite Series for Squigonometric Functions." arXiv:2503.19624v1 [math.CA], https://arxiv.org/abs/2503.19624. The paper provides the theoretical foundation for the MacLaurin series expansions and recursive coefficient calculations implemented in this code.
