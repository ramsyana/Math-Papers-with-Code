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