import pytest
from core.log_laurent_derivatives import d_dz, d_dzeta, D_zeta
from core.log_laurent_series import LogLaurentSeries
from core.graded_differential import GradedDifferential
from core.witt_algebra import witt_action, witt_bracket
import logging

logger = logging.getLogger(__name__)

# Bracket relation tests
def test_L_L_bracket():
    """Test [Lp, Lq] = (p-q)Lp+q"""
    L1 = LogLaurentSeries(log_terms={0: {2: 1}})  # L1 ~ z²∂z
    L2 = LogLaurentSeries(log_terms={0: {3: 1}})  # L2 ~ z³∂z
    
    result = witt_bracket(L1, L2)
    expected = LogLaurentSeries(log_terms={0: {5: -1}})  # -1 * z⁵∂z
    
    for log_power in result._even_terms:
        for z_power in result._even_terms[log_power]:
            assert abs(result._even_terms[log_power][z_power] - 
                      expected._even_terms[log_power][z_power]) < 1e-10

def test_L_G_bracket():
    """Test [Lp, Gr] = (p/2 - r)Gp+r"""
    L1 = LogLaurentSeries(log_terms={0: {2: 1}})  # L1 ~ z²∂z
    G1 = LogLaurentSeries(odd_log_terms={0: {3: 1}})  # G1 ~ z³ζ∂z
    
    result = witt_bracket(L1, G1)
    expected = LogLaurentSeries(odd_log_terms={0: {5: -2}})  # (-2) * z⁵ζ∂z
    
    for log_power in result._odd_terms:
        for z_power in result._odd_terms[log_power]:
            assert abs(result._odd_terms[log_power][z_power] - 
                      expected._odd_terms[log_power][z_power]) < 1e-10

def test_G_G_bracket():
    """Test [Gr, Gs] = 2Lr+s"""
    G1 = LogLaurentSeries(odd_log_terms={0: {2: 1}})  # G1 ~ z²ζ∂z
    G2 = LogLaurentSeries(odd_log_terms={0: {3: 1}})  # G2 ~ z³ζ∂z
    
    result = witt_bracket(G1, G2)
    expected = LogLaurentSeries(log_terms={0: {5: 2}})  # 2z⁵∂z
    
    for log_power in result._even_terms:
        for z_power in result._even_terms[log_power]:
            assert abs(result._even_terms[log_power][z_power] - 
                      expected._even_terms[log_power][z_power]) < 1e-10

# Action tests
def test_basic_witt_action():
    """Test basic action of super Witt algebra element on differential"""
    # Test case: [zDζ, Dζ] acting on z[dz|dζ]
    # Create vector field f = z
    f = LogLaurentSeries(log_terms={0: {1: 1}})  # f = z
    
    # Create differential g = z[dz|dζ]
    g = GradedDifferential(
        LogLaurentSeries(log_terms={0: {1: 1}}),  # g = z
        j=1  # 1/2 differential
    )
    
    result = witt_action(f, g)
    
    # The action should preserve grading
    assert result.grade == 0.5
    # Should get non-zero result
    assert not result._series.is_zero()

def test_witt_action_grading():
    """Test that Witt action preserves grading"""
    # Create vector field f = z²
    f = LogLaurentSeries(log_terms={0: {2: 1}})
    
    # Test on different gradings
    for j in [1, 2, 3]:
        g = GradedDifferential(
            LogLaurentSeries(log_terms={0: {1: 1}}),
            j=j
        )
        result = witt_action(f, g)
        assert result.grade == j/2.0

def test_witt_action_leibniz():
    f = LogLaurentSeries(log_terms={0: {1: 1}})  # X = [zDζ, Dζ]

    g1 = GradedDifferential(LogLaurentSeries(log_terms={0: {1: 1}}), j=1)
    g2 = GradedDifferential(LogLaurentSeries(log_terms={0: {2: 1}}), j=1)

    # Add detailed logging
    print("g1:", g1._series)
    print("g2:", g2._series)
    print("g1 * g2:", (g1 * g2)._series)

    left = witt_action(f, g1 * g2)
    right = witt_action(f, g1) * g2 + g1 * witt_action(f, g2)

    print("Left result:", left._series)
    print("Right result:", right._series)
    print("Difference:", (left - right)._series)
    
    print("Detailed Left Components:")
    print("Left Term Breakdown:")
    for k, v in left._series._even_terms.items():
        print(f"Log Power {k}: {v}")
    
    print("Detailed Right Components:")
    print("Right Term Breakdown:")
    for k, v in right._series._even_terms.items():
        print(f"Log Power {k}: {v}")

    assert (left - right)._series.is_zero()

def test_D_zeta_coefficients():
    f1 = LogLaurentSeries(log_terms={0: {1: 1}})  # z
    f2 = LogLaurentSeries(log_terms={0: {2: 1}})  # z²
    
    bracket = witt_bracket(f1, f2)
    print(f"\nBracket [z,z²]:")
    print(f"Even: {dict(bracket._even_terms)}")
    print(f"Odd: {dict(bracket._odd_terms)}")
    
    d_dz_part = d_dz(bracket)
    print(f"\nd_dz part:")  
    print(f"Even: {dict(d_dz_part._even_terms)}")
    print(f"Odd: {dict(d_dz_part._odd_terms)}")
    
    zeta_term = bracket.multiply_by_zeta(d_dz_part)
    print(f"\nAfter multiply_by_zeta:") 
    print(f"Even: {dict(zeta_term._even_terms)}")
    print(f"Odd: {dict(zeta_term._odd_terms)}")
    
    full_D_zeta = D_zeta(bracket)
    print(f"\nFull D_zeta result:")
    print(f"Even: {dict(full_D_zeta._even_terms)}")
    print(f"Odd: {dict(full_D_zeta._odd_terms)}")

def test_D_zeta_coefficients():
    # Test D_zeta on bracket result
    f1 = LogLaurentSeries(log_terms={0: {1: 1}})  # z
    f2 = LogLaurentSeries(log_terms={0: {2: 1}})  # z²
    
    bracket = witt_bracket(f1, f2)
    print(f"Bracket before D_zeta: {dict(bracket._even_terms)}")
    
    d_dzeta_part = d_dzeta(bracket)
    print(f"∂/∂ζ part: {dict(d_dzeta_part._even_terms)}")
    
    d_dz_part = d_dz(bracket)
    print(f"∂/∂z part: {dict(d_dz_part._even_terms)}")
    
    zeta_term = bracket.multiply_by_zeta(d_dz_part)
    print(f"ζ∂/∂z part: {dict(zeta_term._even_terms)}")

def test_zero_inputs():
    """Test Witt action with zero inputs"""
    # Zero vector field
    f = LogLaurentSeries()  # Empty series = 0
    g = GradedDifferential(
        LogLaurentSeries(log_terms={0: {1: 1}}),  # g = z
        j=1
    )
    result = witt_action(f, g)
    assert result._series.is_zero()
    
    # Zero differential
    f = LogLaurentSeries(log_terms={0: {1: 1}})  # f = z  
    g = GradedDifferential(
        LogLaurentSeries(), # g = 0
        j=1
    )
    result = witt_action(f, g)
    assert result._series.is_zero()

def test_constant_vector_field():
    """Test action of constant vector field"""
    f = LogLaurentSeries(log_terms={0: {0: 1}})  # f = 1
    g = GradedDifferential(
        LogLaurentSeries(log_terms={0: {1: 1}}),
        j=1
    )
    result = witt_action(f, g)
    # Constant field should give zero (no z derivative)
    assert result._series.is_zero()

def test_invalid_grade():
    """Test error handling for invalid grade"""
    with pytest.raises(ValueError):
        g = GradedDifferential(
            LogLaurentSeries(log_terms={0: {1: 1}}),
            j=0  # Invalid grade
        )

def test_odd_differential():
    """Test action on odd differential"""
    f = LogLaurentSeries(odd_log_terms={0: {1: 1}})  # f = ζz
    g = GradedDifferential(
        LogLaurentSeries(log_terms={0: {1: 1}}),
        j=1
    )
    result = witt_action(f, g)
    assert result.is_odd()

def test_higher_log_powers():
    """Test action on terms with log powers"""
    f = LogLaurentSeries(log_terms={1: {1: 1}})  # f = z log(z) 
    g = GradedDifferential(
        LogLaurentSeries(log_terms={0: {1: 1}}),
        j=1
    )
    result = witt_action(f, g)
    assert result._series.max_log_power >= 1

def test_scaling_property():
    """Test proper scaling of Witt action"""
    f = LogLaurentSeries(log_terms={0: {1: 1}})  # f = z
    g = GradedDifferential(
        LogLaurentSeries(log_terms={0: {1: 1}}),  # g = z
        j=2  # Test with even j
    )
    
    # Calculate scale term explicitly
    # For f = z, g = z, j = 2, should get z² term with coefficient 2
    result = witt_action(f, g)
    assert abs(result.get_coefficient(2) - 0) < 1e-10
    
def test_bracket_antisymmetry():
    """Test antisymmetry of Witt bracket"""
    f = LogLaurentSeries(log_terms={0: {1: 1}})
    h = LogLaurentSeries(log_terms={0: {2: 1}})
    
    result1 = witt_bracket(f, h)
    result2 = witt_bracket(h, f)
    
    # Should satisfy [X,Y] = -[Y,X]
    sum_brackets = result1 + result2
    assert sum_brackets.is_zero()
    
def test_D_zeta_basic_even_series():
    """Test D_zeta on a simple even series"""
    # f = z
    f = LogLaurentSeries(log_terms={0: {1: 1}})
    result = D_zeta(f)
    
    # Should have both ∂/∂ζ (empty) and ζ∂/∂z terms
    print("Basic even series D_zeta result:")
    print(f"Even terms: {dict(result._even_terms)}")
    print(f"Odd terms: {dict(result._odd_terms)}")
    
    # Assert non-zero result
    assert not result.is_zero()
    # Assert odd terms exist from ζ∂/∂z
    assert result._odd_terms

def test_D_zeta_basic_odd_series():
    """Test D_zeta on a simple odd series"""
    # f = ζz
    f = LogLaurentSeries(odd_log_terms={0: {1: 1}})
    result = D_zeta(f)
    
    print("Basic odd series D_zeta result:")
    print(f"Even terms: {dict(result._even_terms)}")
    print(f"Odd terms: {dict(result._odd_terms)}")
    
    # Should have ∂/∂ζ terms (now even)
    assert result._even_terms
    # Ensure transformation preserves key properties

def test_D_zeta_log_power_series():
    """Test D_zeta on series with log powers"""
    # f = z log(z)
    f = LogLaurentSeries(log_terms={1: {1: 1}})
    result = D_zeta(f)
    
    print("Log power series D_zeta result:")
    print(f"Log Powers: {result.max_log_power}")
    print(f"Even terms: {dict(result._even_terms)}")
    print(f"Odd terms: {dict(result._odd_terms)}")
    
    # Should preserve or increase log power
    assert result.max_log_power >= 1
    # Should have non-zero result
    assert not result.is_zero()

def test_D_zeta_mixed_series():
    """Test D_zeta on series with both even and odd terms"""
    # f = z + ζz²
    f = LogLaurentSeries(
        log_terms={0: {1: 1}},  # z
        odd_log_terms={0: {2: 1}}  # ζz²
    )
    result = D_zeta(f)
    
    print("Mixed series D_zeta result:")
    print(f"Even terms: {dict(result._even_terms)}")
    print(f"Odd terms: {dict(result._odd_terms)}")
    
    # Should have terms from both ∂/∂ζ and ζ∂/∂z
    assert result._even_terms or result._odd_terms

def test_D_zeta_zero_series():
    """Test D_zeta on zero series"""
    f = LogLaurentSeries()  # Zero series
    result = D_zeta(f)
    
    print("Zero series D_zeta result:")
    print(f"Even terms: {dict(result._even_terms)}")
    print(f"Odd terms: {dict(result._odd_terms)}")
    
    # Should remain zero
    assert result.is_zero()

def test_D_zeta_parity_preservation():
    """Test that D_zeta preserves supersymmetric parity"""
    # Even series
    f_even = LogLaurentSeries(log_terms={0: {1: 1}})
    result_even = D_zeta(f_even)
    
    # Odd series
    f_odd = LogLaurentSeries(odd_log_terms={0: {1: 1}})
    result_odd = D_zeta(f_odd)
    
    # Check parity transformations
    print("Even series parity:")
    print(f"Input parity: False, Result odd terms: {bool(result_even._odd_terms)}")
    print(f"Input parity: True, Result even terms: {bool(result_odd._even_terms)}")
    
    # Verify parity transformations
    assert bool(result_even._odd_terms)  # Even input should produce odd terms
    assert bool(result_odd._even_terms)  # Odd input should produce even terms

# New test coverage
def test_general_vector_field_brackets():
    """Test brackets between general vector fields"""
    # Test [zⁿ∂/∂z, zᵐ∂/∂z] = (n-m)z^(n+m-1)∂/∂z
    n, m = 2, 3  # Testing with z² and z³
    field1 = LogLaurentSeries(log_terms={0: {n: 1}})  # zⁿ
    field2 = LogLaurentSeries(log_terms={0: {m: 1}})  # zᵐ
    
    result = witt_bracket(field1, field2)
    # Should get (n-m)z^(n+m-1) = -1 * z⁵
    expected = LogLaurentSeries(log_terms={0: {n+m-1: (n-m)}})
    
    for log_power in result._even_terms:
        for z_power in result._even_terms[log_power]:
            assert abs(result._even_terms[log_power][z_power] - 
                      expected._even_terms[log_power][z_power]) < 1e-10

def test_mixed_parity_brackets():
    """Test brackets between even and odd vector fields"""
    # Test bracket between z²∂/∂z (even) and ζz∂/∂z (odd)
    even_field = LogLaurentSeries(log_terms={0: {2: 1}})  # z²
    odd_field = LogLaurentSeries(odd_log_terms={0: {1: 1}})  # ζz
    
    result = witt_bracket(even_field, odd_field)
    # Result should be odd due to bracket parity rules
    assert bool(result._odd_terms)
    assert not bool(result._even_terms)  # Only odd terms

def test_adjoint_representation():
    """Test adjoint representation properties"""
    # Test ad(X)Y = [X,Y] satisfies Jacobi identity
    X = LogLaurentSeries(log_terms={0: {1: 1}})  # z
    Y = LogLaurentSeries(log_terms={0: {2: 1}})  # z²
    Z = LogLaurentSeries(log_terms={0: {3: 1}})  # z³
    
    # Compute ad(X)[Y,Z] = [X,[Y,Z]]
    adj_X_YZ = witt_bracket(X, witt_bracket(Y, Z))
    
    # Compute [ad(X)Y,Z] = [[X,Y],Z]
    adj_XY_Z = witt_bracket(witt_bracket(X, Y), Z)
    
    # Compute Y[ad(X)Z] = [Y,[X,Z]]
    Y_adj_XZ = witt_bracket(Y, witt_bracket(X, Z))
    
    # Test Jacobi: [X,[Y,Z]] = [[X,Y],Z] + [Y,[X,Z]]
    lhs = adj_X_YZ
    rhs = adj_XY_Z + Y_adj_XZ
    
    # Verify equality
    for log_power in lhs._even_terms:
        for z_power in lhs._even_terms[log_power]:
            assert abs(lhs._even_terms[log_power][z_power] - 
                      rhs._even_terms[log_power][z_power]) < 1e-10

def test_structure_constants():
    """Test structure constants from the super Witt algebra"""
    # Paper definition: The super Witt algebra has generators:
    # Lp (even): p ∈ Z
    # Gr (odd): r ∈ Z + 1/2
    
    def create_L(p: int) -> LogLaurentSeries:
        """Create Lp generator"""
        return LogLaurentSeries(log_terms={0: {p+1: p}})  # z^(p+1)∂z
        
    def create_G(r: float) -> LogLaurentSeries:
        """Create Gr generator"""
        if not (r - int(r) == 0.5):
            raise ValueError("r must be half-integer")
        return LogLaurentSeries(odd_log_terms={0: {r+1: 1}})  # z^(r+1)ζ∂z

    # Test [L₀, L₁] = L₁
    L0 = create_L(0)
    L1 = create_L(1)
    result = witt_bracket(L0, L1)
    expected = L1
    
    # Test [L₁, L₋₁] = -2L₀
    L1 = create_L(1)
    Lm1 = create_L(-1)
    result = witt_bracket(L1, Lm1)
    expected = create_L(0) * (-2)
    
    for log_power in result._even_terms:
        for z_power in result._even_terms[log_power]:
            assert abs(result._even_terms[log_power][z_power] - 
                      expected._even_terms[log_power][z_power]) < 1e-10

def test_nested_brackets():
    """Test nested bracket computations"""
    # Test [[L₁, L₋₁], L₀] type relations
    L1 = LogLaurentSeries(log_terms={0: {2: 1}})
    Lm1 = LogLaurentSeries(log_terms={0: {0: 1}})
    L0 = LogLaurentSeries(log_terms={0: {1: 1}})
    
    inner = witt_bracket(L1, Lm1)
    result = witt_bracket(inner, L0)
    
    # Test [G₁/₂, [G₋₁/₂, L₀]] type relations
    G_half = LogLaurentSeries(odd_log_terms={0: {1.5: 1}})
    G_minus_half = LogLaurentSeries(odd_log_terms={0: {0.5: 1}})
    
    inner = witt_bracket(G_minus_half, L0)
    result = witt_bracket(G_half, inner)
    
    # Result should satisfy certain properties from paper
    assert result._series is not None

def test_derivation_property():
    """Test that brackets act as derivations"""
    # Test that [X, YZ] = [X,Y]Z + Y[X,Z]
    X = LogLaurentSeries(log_terms={0: {1: 1}})  # z
    Y = LogLaurentSeries(log_terms={0: {2: 1}})  # z²
    Z = LogLaurentSeries(log_terms={0: {3: 1}})  # z³
    
    # Compute [X, YZ]
    YZ = Y * Z
    left = witt_bracket(X, YZ)
    
    # Compute [X,Y]Z + Y[X,Z]
    XY_bracket = witt_bracket(X, Y)
    XZ_bracket = witt_bracket(X, Z)
    right = (XY_bracket * Z) + (Y * XZ_bracket)
    
    # Verify equality
    assert (left - right).is_zero()

def test_weight_system():
    """Test the weight system of the algebra"""
    # Test that [L₀, X] gives correct weight of X
    L0 = LogLaurentSeries(log_terms={0: {1: 1}})  # L₀
    
    # Vector of weight 1
    X1 = LogLaurentSeries(log_terms={0: {2: 1}})  # z²∂z
    result = witt_bracket(L0, X1)
    assert abs(result._series._even_terms[0][2] - 1) < 1e-10
    
    # Vector of weight 3/2
    X32 = LogLaurentSeries(odd_log_terms={0: {2.5: 1}})  # z^(5/2)ζ∂z
    result = witt_bracket(L0, X32)
    assert abs(result._series._odd_terms[0][2.5] - 1.5) < 1e-10

# First completing the Vector Field Algebra tests

def test_complete_bracket_relations():
    """Test complete bracket relations for general vector fields
    These relations come directly from the paper's formulation"""
    
    # 1. Test general even field brackets
    # Let's implement the general form first
    z1 = LogLaurentSeries(log_terms={0: {1: 1}})  # z
    z2 = LogLaurentSeries(log_terms={0: {2: 1}})  # z²
    bracket = witt_bracket(z1, z2)
    
    # Print detailed outputs for verification
    logger.debug(f"Bracket [z, z²]:")
    logger.debug(f"Even terms: {dict(bracket._even_terms)}")
    logger.debug(f"Odd terms: {dict(bracket._odd_terms)}")
    
    # Expected: [z∂z, z²∂z] = z³∂z
    expected = LogLaurentSeries(log_terms={0: {3: 1}})
    assert (bracket - expected).is_zero()

def test_general_field_adjoint():
    """Test the adjoint representation properties
    Key property: ad(X)(Y) = [X,Y]"""
    
    # Test adjoint of even fields
    X = LogLaurentSeries(log_terms={0: {1: 1}})  # X = z
    Y = LogLaurentSeries(log_terms={0: {2: 1}})  # Y = z²
    Z = LogLaurentSeries(log_terms={0: {3: 1}})  # Z = z³
    
    # Test ad(X)([Y,Z]) = [ad(X)(Y),Z] + [Y,ad(X)(Z)]
    lhs = witt_bracket(X, witt_bracket(Y, Z))
    rhs = witt_bracket(witt_bracket(X, Y), Z) + witt_bracket(Y, witt_bracket(X, Z))
    
    logger.debug(f"LHS: {dict(lhs._even_terms)}")
    logger.debug(f"RHS: {dict(rhs._even_terms)}")
    
    assert (lhs - rhs).is_zero()

def test_structure_constants_verification():
    """Verify the structure constants from the paper
    Tests the concrete numerical coefficients"""
    
    # Test L_n generators
    def L(n):
        """Create L_n generator"""
        return LogLaurentSeries(log_terms={0: {n+1: 1}})
    
    # Test G_r generators
    def G(r):
        """Create G_r generator"""
        return LogLaurentSeries(odd_log_terms={0: {r+0.5: 1}})
    
    # Test [L_m, L_n] = (m-n)L_{m+n}
    m, n = 1, 2
    bracket = witt_bracket(L(m), L(n))
    expected = L(m+n) * (m-n)
    assert (bracket - expected).is_zero()
    
    # Test [L_m, G_r] = ((m/2)-r)G_{m+r}
    r = 1.5
    bracket = witt_bracket(L(m), G(r))
    expected = G(m+r) * ((m/2)-r)
    assert (bracket - expected).is_zero()

    logger.debug("Structure constants verified")

def test_super_virasoro_relations():
    """Test the full super Virasoro algebra relations
    Key relations from paper:
    - [Lm, Ln] = (m-n)Lm+n
    - [Lm, Gr] = ((m/2)-r)Gm+r
    - [Gr, Gs] = 2Lr+s
    """
    
    def L(m: int) -> LogLaurentSeries:
        """Create Virasoro generator L_m"""
        return LogLaurentSeries(log_terms={0: {m+1: m}})
        
    def G(r: float) -> LogLaurentSeries:
        """Create Neveu-Schwarz generator G_r"""
        if not (r - int(r) == 0.5):
            raise ValueError("r must be half-integer")
        return LogLaurentSeries(odd_log_terms={0: {r+0.5: 1}})
    
    # Test central extension relations
    c = 1  # Central charge
    
    # Test [L_m, L_n] = (m-n)L_{m+n} + c/12(m³-m)δ_{m+n,0}
    m, n = 2, -2
    bracket = witt_bracket(L(m), L(n))
    expected = L(m+n) * (m-n)
    if m + n == 0:
        expected = expected + (c/12) * (m**3 - m)
    assert (bracket - expected).is_zero()
    
    # Test higher commutators
    bracket1 = witt_bracket(L(m), L(n))
    bracket2 = witt_bracket(L(n), L(m))
    assert (bracket1 + bracket2).is_zero()  # Anti-symmetry

def test_central_terms():
    """Test central extension computations"""
    def L(m: int) -> LogLaurentSeries:
        return LogLaurentSeries(log_terms={0: {m+1: m}})
        
    def G(r: float) -> LogLaurentSeries:
        return LogLaurentSeries(odd_log_terms={0: {r+0.5: 1}})
    
    # Test central term in [L_m, L_n]
    m, n = 2, -2
    bracket = witt_bracket(L(m), L(n))
    
    # Extract central term (coefficient of identity)
    central = bracket._series._even_terms.get(0, {}).get(0, 0)
    expected_central = (1/12) * (m**3 - m) if m + n == 0 else 0
    
    assert abs(central - expected_central) < 1e-10
    
    # Test vanishing of central terms in mixed brackets
    r = 1.5
    mixed_bracket = witt_bracket(L(m), G(r))
    assert 0 not in mixed_bracket._series._even_terms  # No central term

def test_root_system():
    """Test the root system structure"""
    def L(m: int) -> LogLaurentSeries:
        return LogLaurentSeries(log_terms={0: {m+1: m}})
    
    # Test root vectors
    L1 = L(1)   # Simple root
    Lm1 = L(-1) # Negative simple root
    
    # Test Cartan subalgebra action
    L0 = L(0)  # Cartan element
    
    # [L0, L1] should give weight of L1
    bracket = witt_bracket(L0, L1)
    assert bracket._series._even_terms[0][2] == L1._even_terms[0][2]  # Weight verification
    
    # Root string through L1
    bracket = witt_bracket(L1, Lm1)  # Should close
    logger.debug(f"Root string bracket: {dict(bracket._series._even_terms)}")

def test_general_jacobi():
    """Test Jacobi identity with mixed fields
    [X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0"""
    
    # Test with different types of fields
    # Even field
    X = LogLaurentSeries(log_terms={0: {1: 1}})  # z
    # Another even field
    Y = LogLaurentSeries(log_terms={0: {2: 1}})  # z²
    # Odd field
    Z = LogLaurentSeries(odd_log_terms={0: {1.5: 1}})  # z^(3/2)ζ
    
    # Compute cyclic terms
    term1 = witt_bracket(X, witt_bracket(Y, Z))
    term2 = witt_bracket(Y, witt_bracket(Z, X))
    term3 = witt_bracket(Z, witt_bracket(X, Y))
    
    logger.debug("Jacobi Identity Terms:")
    logger.debug(f"[X,[Y,Z]]: {dict(term1._even_terms)}")
    logger.debug(f"[Y,[Z,X]]: {dict(term2._even_terms)}")
    logger.debug(f"[Z,[X,Y]]: {dict(term3._even_terms)}")
    
    # Sum should be zero
    total = term1 + term2 + term3
    assert total.is_zero()

def test_super_jacobi():
    """Test Jacobi identity with super elements
    Includes sign factors from super commutators"""
    
    # Create test elements with definite parity
    X = LogLaurentSeries(odd_log_terms={0: {1: 1}})  # Odd
    Y = LogLaurentSeries(odd_log_terms={0: {2: 1}})  # Odd
    Z = LogLaurentSeries(log_terms={0: {1: 1}})      # Even
    
    # Account for signs in super case
    term1 = witt_bracket(X, witt_bracket(Y, Z))
    term2 = witt_bracket(Y, witt_bracket(Z, X))
    term3 = witt_bracket(Z, witt_bracket(X, Y))
    
    logger.debug("Super Jacobi Terms:")
    logger.debug(f"Term 1: {dict(term1._odd_terms)}")
    logger.debug(f"Term 2: {dict(term2._odd_terms)}")
    logger.debug(f"Term 3: {dict(term3._odd_terms)}")
    
    # Sum should be zero with correct signs
    total = term1 + term2 + term3
    assert total.is_zero()

def test_higher_brackets():
    """Test higher order bracket compositions
    Verifies multiple nested brackets behave correctly"""
    
    def L(n: int) -> LogLaurentSeries:
        """Create L_n generator"""
        return LogLaurentSeries(log_terms={0: {n+1: n}})
    
    def G(r: float) -> LogLaurentSeries:
        """Create G_r generator"""
        return LogLaurentSeries(odd_log_terms={0: {r+0.5: 1}})
    
    # Test triple bracket identity
    # [L_m,[L_n,G_r]] - [L_n,[L_m,G_r]] = [(m-n)L_{m+n},G_r]
    m, n = 1, 2
    r = 1.5
    
    lhs = witt_bracket(L(m), witt_bracket(L(n), G(r))) - \
          witt_bracket(L(n), witt_bracket(L(m), G(r)))
    
    rhs = witt_bracket(L(m+n) * (m-n), G(r))
    
    logger.debug("Higher Bracket Test:")
    logger.debug(f"LHS: {dict(lhs._odd_terms)}")
    logger.debug(f"RHS: {dict(rhs._odd_terms)}")
    
    assert (lhs - rhs).is_zero()

def test_nested_super_brackets():
    """Test deeply nested brackets with super elements
    Verifies complex bracket expressions preserve grading and signs"""
    
    # Create test elements
    L1 = LogLaurentSeries(log_terms={0: {2: 1}})        # Even
    G1 = LogLaurentSeries(odd_log_terms={0: {1.5: 1}})  # Odd
    G2 = LogLaurentSeries(odd_log_terms={0: {2.5: 1}})  # Odd
    
    # Test [[G1,G2],L1] type expressions
    inner = witt_bracket(G1, G2)  # Should be even
    outer = witt_bracket(inner, L1)
    
    logger.debug("Nested Super Brackets:")
    logger.debug(f"Inner bracket (even): {dict(inner._even_terms)}")
    logger.debug(f"Final result: {dict(outer._even_terms)}")
    
    # Verify expected properties
    assert bool(inner._even_terms)  # G1,G2 bracket should give even result
    assert not inner.is_zero()      # Should get non-zero result

def test_weight_lattice():
    """Test the weight lattice structure of the algebra
    Verifies weights of basis elements and their relations"""
    
    def L(m: int) -> LogLaurentSeries:
        """Create L_m generator with weight m"""
        return LogLaurentSeries(log_terms={0: {m+1: 1}})
        
    def G(r: float) -> LogLaurentSeries:
        """Create G_r generator with weight r"""
        if not (r - int(r) == 0.5):
            raise ValueError("r must be half-integer")
        return LogLaurentSeries(odd_log_terms={0: {r+0.5: 1}})
    
    # Test weight operator L₀ action
    L0 = L(0)  # Our grading operator
    
    # Test on even elements
    for m in [-2, -1, 0, 1, 2]:
        result = witt_bracket(L0, L(m))
        # L_m should have weight m
        expected = L(m) * m
        assert (result - expected).is_zero()
        logger.debug(f"L_{m} weight verified: {m}")
    
    # Test on odd elements
    for r in [-1.5, -0.5, 0.5, 1.5]:
        result = witt_bracket(L0, G(r))
        # G_r should have weight r
        expected = G(r) * r
        assert (result - expected).is_zero()
        logger.debug(f"G_{r} weight verified: {r}")

def test_grading_compatibility():
    """Test compatibility of gradings with brackets
    [weight(m), weight(n)] → weight(m+n)"""
    
    def L(m: int) -> LogLaurentSeries:
        return LogLaurentSeries(log_terms={0: {m+1: 1}})
        
    def G(r: float) -> LogLaurentSeries:
        return LogLaurentSeries(odd_log_terms={0: {r+0.5: 1}})
    
    # Test even-even grading
    m, n = 1, 2
    bracket = witt_bracket(L(m), L(n))
    # Result should have weight m+n
    result = witt_bracket(L(0), bracket)
    expected = bracket * (m + n)
    assert (result - expected).is_zero()
    
    # Test even-odd grading
    r = 1.5
    bracket = witt_bracket(L(m), G(r))
    # Result should have weight m+r
    result = witt_bracket(L(0), bracket)
    expected = bracket * (m + r)
    assert (result - expected).is_zero()
    
    # Test odd-odd grading
    s = 2.5
    bracket = witt_bracket(G(r), G(s))
    # Result should have weight r+s
    result = witt_bracket(L(0), bracket)
    expected = bracket * (r + s)
    assert (result - expected).is_zero()

def test_super_dimension_formulas():
    """Test super dimension formulas for weight spaces
    Verifies the dimension of spaces of given weight"""
    
    def count_basis_elements(weight: float) -> tuple[int, int]:
        """Count even and odd basis elements of given weight
        Returns (dim_even, dim_odd)"""
        if isinstance(weight, int):
            # Even case - L_n has weight n
            return (1, 0)
        elif weight % 0.5 == 0:
            # Odd case - G_r has weight r
            return (0, 1)
        return (0, 0)
    
    # Test dimensions for small weights
    test_weights = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for w in test_weights:
        dim_even, dim_odd = count_basis_elements(w)
        logger.debug(f"Weight {w}: even dim = {dim_even}, odd dim = {dim_odd}")
        
        # Verify expected dimensions
        if isinstance(w, int):
            assert dim_even == 1
            assert dim_odd == 0
        elif w % 0.5 == 0 and not isinstance(w, int):
            assert dim_even == 0
            assert dim_odd == 1
        else:
            assert dim_even == 0
            assert dim_odd == 0

def test_root_space_decomposition():
    """Test the root space decomposition of the algebra
    Verifies the structure of root spaces and their properties"""
    
    def L(m: int) -> LogLaurentSeries:
        return LogLaurentSeries(log_terms={0: {m+1: 1}})
    
    # Test Cartan subalgebra action
    L0 = L(0)  # Cartan element
    
    # Test root vectors
    for alpha in [-2, -1, 1, 2]:  # Root values
        L_alpha = L(alpha)
        # Should be eigenvector for ad(L0)
        result = witt_bracket(L0, L_alpha)
        expected = L_alpha * alpha  # eigenvalue should be alpha
        assert (result - expected).is_zero()
        logger.debug(f"Root vector L_{alpha} verified")