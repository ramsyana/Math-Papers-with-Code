import pytest
from super_mumford.core.log_laurent_series import LogLaurentSeries
from super_mumford.core.graded_differential import GradedDifferential
from super_mumford.core.lie_derivative import lie_derivative, lie_bracket
import logging

logger = logging.getLogger(__name__)

def test_lie_derivative_basics():
    """Test basic properties of Lie derivative"""
    # Test on simple function g = z with j = 2
    f = LogLaurentSeries(log_terms={0: {1: 1}})  # f = z
    g = GradedDifferential(
        LogLaurentSeries(log_terms={0: {1: 1}}),  # g = z
        j=2
    )
    
    result = lie_derivative(f, g)
    assert result.grade == 1  # Should preserve grade
    # For f = z, g = z[dz|dζ]⊗2, result should have z² term from ∂f/∂z·g
    assert result.get_coefficient(2) == 2

def test_lie_derivative_odd():
    """Test Lie derivative on odd differentials"""
    # Test with j = 1 (odd differential)
    f = LogLaurentSeries(log_terms={0: {1: 1}})  # f = z
    g = GradedDifferential(
        LogLaurentSeries(log_terms={0: {1: 1}}),  # g = z
        j=1
    )
    
    result = lie_derivative(f, g)
    assert result.is_odd()  # Should preserve odd parity
    assert result.grade == 0.5  # j/2 = 1/2

def test_lie_bracket():
    """Test Lie bracket computations"""
    # Test [D, D] = 0 where D = [Dζ, Dζ]
    f = LogLaurentSeries(log_terms={0: {0: 0}})  # f = 0
    h = LogLaurentSeries(log_terms={0: {2: 1}})  # h = z²
    
    result = lie_bracket(f, h)
    assert str(result) == '0: {2: 1}'  # Should get zero
    
    # Test bracket of z with itself
    f = LogLaurentSeries(log_terms={0: {1: 1}})  # f = z
    h = LogLaurentSeries(log_terms={0: {1: 1}})  # h = z
    
    result = lie_bracket(f, h)
    # Should get non-zero result from non-commuting vector fields
    assert str(result) != "0"

def test_lie_derivative_linearity():
    """Test linearity properties of Lie derivative"""
    # Test on sum: ρ(X)(g1 + g2) = ρ(X)g1 + ρ(X)g2
    f = LogLaurentSeries(log_terms={0: {1: 1}})  # f = z
    
    g1 = GradedDifferential(
        LogLaurentSeries(log_terms={0: {1: 1}}),  # z
        j=2
    )
    g2 = GradedDifferential(
        LogLaurentSeries(log_terms={0: {2: 1}}),  # z²
        j=2
    )
    
    # Compute ρ(X)(g1 + g2)
    sum_result = lie_derivative(f, g1 + g2)
    
    # Compute ρ(X)g1 + ρ(X)g2
    separate_result = lie_derivative(f, g1) + lie_derivative(f, g2)
    
    # Check coefficients match
    for power in range(-5, 5):
        assert abs(sum_result.get_coefficient(power) - 
                  separate_result.get_coefficient(power)) < 1e-10

def test_lie_derivative_jacobi():
    """Test Jacobi identity for Lie derivatives"""
    # Test [X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0
    f = LogLaurentSeries(log_terms={0: {1: 1}})  # X = z
    g = LogLaurentSeries(log_terms={0: {2: 1}})  # Y = z²
    h = LogLaurentSeries(log_terms={0: {3: 1}})  # Z = z³
    
    # Compute cyclic sum of brackets
    bracket1 = lie_bracket(f, lie_bracket(g, h))
    bracket2 = lie_bracket(g, lie_bracket(h, f))
    bracket3 = lie_bracket(h, lie_bracket(f, g))
    
    result = bracket1 + bracket2 + bracket3
    
    # Check that cyclic sum vanishes
    for k in result._even_terms.keys():
        for p in result._even_terms[k].keys():
            assert abs(result._even_terms[k][p]) < 1e-9
    for k in result._odd_terms.keys():
        for p in result._odd_terms[k].keys():
            assert abs(result._odd_terms[k][p]) < 1e-9

def test_scale_term_computation():
    """Test explicitly that scale term produces z² terms"""
    f = LogLaurentSeries(log_terms={0: {1: 1}})  # f = z
    g = GradedDifferential(
        LogLaurentSeries(log_terms={0: {1: 1}}),  # g = z
        j=2
    )
    
    result = lie_derivative(f, g)
    print(f"Scale term coefficients:")
    for power in range(3):  # Check z^0, z^1, z^2 terms
        print(f"z^{power}: {result._series._even_terms[0].get(power, 0)}")
    
    assert result._series._even_terms[0].get(2, 0) == 2

def test_minimal_scale_term():
    """Test the minimal case where f=z, g=z with detailed logging"""
    logger.debug("\n=== Testing Minimal Scale Term Case ===")
    
    f = LogLaurentSeries(log_terms={0: {1: 1}})  # f = z
    logger.debug(f"f = {f}")
    logger.debug(f"f coefficients: {f._even_terms}")
    
    g = GradedDifferential(
        LogLaurentSeries(log_terms={0: {1: 1}}),  # g = z
        j=2
    )
    logger.debug(f"g = {g._series}")
    logger.debug(f"g coefficients: {g._series._even_terms}")
    logger.debug(f"g.grade = {g.grade}")
    
    # Test step by step
    # 1. Scale term calculation
    df_dz = d_dz(f)
    logger.debug(f"\n1. df/dz = {df_dz}")
    logger.debug(f"df/dz coefficients: {df_dz._even_terms}")
    
    scale_term = df_dz * g._series * g.grade
    logger.debug(f"2. Scale term = {scale_term}")
    logger.debug(f"Scale term coefficients: {scale_term._even_terms}")
    
    result = lie_derivative(f, g)
    logger.debug(f"\nFinal result = {result._series}")
    logger.debug(f"Final coefficients: {result._series._even_terms}")
    
    # Print all terms in detail
    for power in range(3):
        coeff = result._series._even_terms[0].get(power, 0)
        logger.debug(f"z^{power} coefficient: {coeff}")
    
    assert result._series._even_terms[0].get(2, 0) == 2

def test_minimal_bracket():
    """Test minimal bracket case with detailed logging"""
    logger.debug("\n=== Testing Minimal Bracket Case ===")
    
    f = LogLaurentSeries(log_terms={0: {0: 0}})  # f = 0
    h = LogLaurentSeries(log_terms={0: {2: 1}})  # h = z²
    
    logger.debug(f"f = {f}")
    logger.debug(f"f coefficients: {f._even_terms}")
    logger.debug(f"h = {h}")
    logger.debug(f"h coefficients: {h._even_terms}")
    
    result = lie_bracket(f, h)
    
    logger.debug(f"\nResult = {result}")
    logger.debug(f"Result coefficients: {result._even_terms}")
    logger.debug(f"Result string representation: {str(result)}")
    
    assert str(result) == '0: {2: 1}'