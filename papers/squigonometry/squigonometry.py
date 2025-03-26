import math

def precompute_factorials(max_k):
    """Precompute factorials up to max_k for efficiency."""
    factorials = [1]
    for k in range(1, max_k + 1):
        factorials.append(factorials[-1] * k)
    return factorials

def compute_maclaurin_coefficients(m, n, p, J, factorials):
    """
    Compute MacLaurin series coefficients for squigonometric functions using DP.
    
    Args:
        m (int): Parameter m from the recursion relation.
        n (int): Parameter n from the recursion relation (exponent offset).
        p (int): Squigonometric parameter (e.g., p=4 for 4-squine/cosquine).
        J (int): Number of terms in the series.
        factorials (list): Precomputed factorials up to max_k.
    
    Returns:
        list: Coefficients of the MacLaurin series.
    """
    max_k = min(n + p * J, len(factorials) - 1)
    coefficients = []
    prev_dp = [1] + [0] * J  # dp[0][j] for j=0 to J

    # Check for k=0
    if n == 0:
        coefficients.append(1 / factorials[0])  # For j=0, k=0

    for k in range(1, max_k + 1):
        current_dp = [0] * (J + 1)
        for j in range(min(k, J) + 1):
            term1 = (n - (k - 1) + p * j) * prev_dp[j]
            term2 = 0
            if j > 0:
                term2 = (m + (k - 1) * (p - 1) - p * (j - 1)) * prev_dp[j - 1]
            current_dp[j] = term1 + term2
        if k >= n and (k - n) % p == 0:
            j = (k - n) // p
            if 0 <= j <= J:
                coefficients.append( (-1)**j * current_dp[j] / factorials[k] )
        prev_dp = current_dp

    return coefficients

def evaluate_squigonometric(coefficients, p, n, t):
    """
    Evaluate the squigonometric function at point t using the MacLaurin series.
    
    Args:
        coefficients (list): MacLaurin series coefficients.
        p (int): Squigonometric parameter.
        n (int): Exponent offset.
        t (float): Evaluation point.
    
    Returns:
        float: Value of the squigonometric function at t.
    """
    result = 0.0
    for j, c in enumerate(coefficients):
        exponent = n + p * j
        term = c * (t ** exponent)
        result += term
    return result

# Precompute factorials for efficiency
max_k = 100  # Adjust based on expected J and p
factorials = precompute_factorials(max_k)

# Example usage for p=4
p = 4
J = 10  # Number of terms for demonstration

# Compute cosquine (cq) coefficients: m=1, n=0
cq_coeffs = compute_maclaurin_coefficients(m=1, n=0, p=p, J=J, factorials=factorials)

# Compute squine (sq) coefficients: m=0, n=1
sq_coeffs = compute_maclaurin_coefficients(m=0, n=1, p=p, J=J, factorials=factorials)

# Evaluate at a point within the convergence radius (e.g., t=0.5)
t = 0.5
cq_t = evaluate_squigonometric(cq_coeffs, p, 0, t)
sq_t = evaluate_squigonometric(sq_coeffs, p, 1, t)

print(f"cq_{p}({t}) ≈ {cq_t}")
print(f"sq_{p}({t}) ≈ {sq_t}")

# Verify Pythagorean identity: |sq|^p + |cq|^p ≈ 1
identity_value = abs(sq_t) ** p + abs(cq_t) ** p
print(f"|sq|^{p} + |cq|^{p} ≈ {identity_value}")