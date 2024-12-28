import pytest
from super_mumford.core.log_laurent_series import LogLaurentSeries
from super_mumford.core.graded_differential import GradedDifferential

def test_graded_differential_creation():
    """Test creation of j/2-differentials with proper grading"""
    # Test even j/2 differential (j=2)
    diff = GradedDifferential(
        LogLaurentSeries(log_terms={0: {1: 1}}),  # z
        j=2
    )
    assert diff.grade == 1  # j/2 = 1
    assert diff.is_even()
    
    # Test odd j/2 differential (j=1)
    diff = GradedDifferential(
        LogLaurentSeries(log_terms={0: {1: 1}}),  # z
        j=1
    )
    assert diff.grade == 0.5  # j/2 = 1/2
    assert diff.is_odd()

def test_graded_differential_multiplication():
    """Test multiplication of j/2-differentials respecting grading"""
    # (z[dz|dζ]) * (w[dz|dζ]^2) = zw[dz|dζ]^3
    diff1 = GradedDifferential(
        LogLaurentSeries(log_terms={0: {1: 1}}),  # z
        j=1
    )
    diff2 = GradedDifferential(
        LogLaurentSeries(log_terms={0: {1: 1}}),  # z
        j=2
    )
    result = diff1 * diff2
    assert result.grade == 1.5  # (1+2)/2
    assert result.is_odd()  # Since 1+2=3 is odd
    
def test_graded_differential_parity():
    """Test parity rules for j/2-differentials"""
    # Create differentials with even and odd series
    even_series = LogLaurentSeries(log_terms={0: {1: 1}})  # z
    odd_series = LogLaurentSeries(odd_log_terms={0: {0: 1}})  # ζ
    
    # Test j even case
    diff_even_j = GradedDifferential(even_series, j=2)
    assert diff_even_j.total_parity() == 0  # even + even = even
    
    diff_odd_j = GradedDifferential(odd_series, j=2)
    assert diff_odd_j.total_parity() == 1  # odd + even = odd
    
    # Test j odd case
    diff_even_j = GradedDifferential(even_series, j=1)
    assert diff_even_j.total_parity() == 1  # even + odd = odd
    
    diff_odd_j = GradedDifferential(odd_series, j=1)
    assert diff_odd_j.total_parity() == 0  # odd + odd = even

def test_tensor_power():
    """Test handling of tensor powers [dz|dζ]^⊗j"""
    # Start with z[dz|dζ]
    diff = GradedDifferential(
        LogLaurentSeries(log_terms={0: {1: 1}}),
        j=1
    )
    
    # Tensor with itself
    result = diff.tensor_power(2)
    assert result.grade == 1.0  # j/2 = 2/2
    assert result.get_coefficient(1) == 1  # Coefficient of z should be preserved
    
    # Test higher tensor power
    result = diff.tensor_power(3)
    assert result.grade == 1.5  # j/2 = 3/2
    assert result.get_coefficient(1) == 1  # Coefficient of z should be preserved

def test_invalid_operations():
    """Test that invalid operations raise appropriate errors"""
    diff1 = GradedDifferential(
        LogLaurentSeries(log_terms={0: {1: 1}}),
        j=1
    )
    
    # Test invalid tensor power
    with pytest.raises(ValueError):
        diff1.tensor_power(0)
    
    # Test invalid j value
    with pytest.raises(ValueError):
        GradedDifferential(
            LogLaurentSeries(log_terms={0: {1: 1}}),
            j=0
        )