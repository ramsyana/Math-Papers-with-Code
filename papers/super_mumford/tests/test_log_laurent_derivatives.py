import pytest
from super_mumford.core.log_laurent_series import LogLaurentSeries
from super_mumford.core.log_laurent_derivatives import d_dz, d_dzeta, D_zeta

def test_d_dz():
    # Test d/dz(z) = 1
    series = LogLaurentSeries(log_terms={0: {1: 1}})  # z
    result = d_dz(series)
    assert "1" in str(result)
    
    # Test d/dz(log(z)) = 1/z
    series = LogLaurentSeries(log_terms={1: {0: 1}})  # log(z)
    result = d_dz(series)
    assert "1z^-1" in str(result)
    
    # Test d/dz(z*log(z)) = log(z) + 1
    series = LogLaurentSeries(log_terms={1: {1: 1}})  # z*log(z)
    result = d_dz(series)
    assert "log(z)" in str(result)
    assert "1" in str(result)

def test_d_dzeta():
    # Test d/dζ(ζ) = 1
    series = LogLaurentSeries(odd_log_terms={0: {0: 1}})  # ζ
    result = d_dzeta(series)
    assert "1" in str(result)
    
    # Test d/dζ(1) = 0
    series = LogLaurentSeries(log_terms={0: {0: 1}})  # 1
    result = d_dzeta(series)
    assert "0" in str(result) or str(result) == "0"

def test_D_zeta():
    # Test D_ζ(ζ) = 1
    series = LogLaurentSeries(odd_log_terms={0: {0: 1}})  # ζ
    result = D_zeta(series)
    assert "1" in str(result)
    
    # Test D_ζ(z) = ζ
    series = LogLaurentSeries(log_terms={0: {1: 1}})  # z
    result = D_zeta(series)
    assert "ζ" in str(result)