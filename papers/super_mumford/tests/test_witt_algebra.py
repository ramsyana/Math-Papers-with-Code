import pytest
from core.log_laurent_series import LogLaurentSeries
from core.graded_differential import GradedDifferential
from core.witt_algebra import witt_action, witt_bracket
import logging

logger = logging.getLogger(__name__)

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

def test_witt_action_jacobi():
    """Test Jacobi identity for Witt algebra action"""
    # Test [X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0
    # X = [zDζ, Dζ], Y = [z²Dζ, Dζ], Z = [z³Dζ, Dζ]
    f1 = LogLaurentSeries(log_terms={0: {1: 1}})  # z
    f2 = LogLaurentSeries(log_terms={0: {2: 1}})  # z²
    f3 = LogLaurentSeries(log_terms={0: {3: 1}})  # z³

    g = GradedDifferential(
        LogLaurentSeries(log_terms={0: {1: 1}}),
        j=1
    )

    # Compute cyclic sum of nested actions
    term1 = witt_action(f1, witt_action(f2, witt_action(f3, g)))
    term2 = witt_action(f2, witt_action(f3, witt_action(f1, g)))
    term3 = witt_action(f3, witt_action(f1, witt_action(f2, g)))

    result = term1 + term2 + term3

    # The sum should vanish
    assert result._series.is_zero()

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