import pytest
from super_mumford.core.log_laurent_series import LogLaurentSeries
from super_mumford.core.log_laurent_integration import integrate_z

def test_integrate_z():
    # Test ∫ 1 dz = z
    series = LogLaurentSeries(log_terms={0: {0: 1}})  # 1
    result = integrate_z(series)
    assert "z" in str(result)
    
    # Test ∫ dz/z = log(z)
    series = LogLaurentSeries(log_terms={0: {-1: 1}})  # 1/z
    result = integrate_z(series)
    assert "log(z)" in str(result)
    
    # Test ∫ z dz = z^2/2
    series = LogLaurentSeries(log_terms={0: {1: 1}})  # z
    result = integrate_z(series)
    assert "0.5z^2" in str(result)

def test_integrate_z_logs():
    # Test ∫ log(z) dz = z*log(z) - z 
    series = LogLaurentSeries(log_terms={1: {0: 1}})  # log(z)
    result = integrate_z(series)
    assert "z*log(z)" in str(result)
    assert "-1z" in str(result)
    
    # Test ∫ z*log(z) dz = (z^2/2)*log(z) - z^2/4
    series = LogLaurentSeries(log_terms={1: {1: 1}})  # z*log(z)
    result = integrate_z(series)
    assert "0.5z^2*log(z)" in str(result)
    assert "-0.25z^2" in str(result)

def test_integrate_z_odd():
    # Test ∫ ζ dz = ζz
    series = LogLaurentSeries(odd_log_terms={0: {0: 1}})
    result = integrate_z(series) 
    assert "ζ*z" in str(result)

def test_integrate_z_odd_power():
    # Test ∫ ζ/z dz = ζ*log(z)
    series = LogLaurentSeries(odd_log_terms={0: {-1: 1}})
    result = integrate_z(series)
    assert "ζ*log(z)" in str(result)