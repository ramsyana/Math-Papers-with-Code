import pytest
from super_mumford.core.laurent_series import LaurentSeries
import numpy as np

def test_laurent_series_creation():
    # Test empty series
    empty = LaurentSeries()
    assert str(empty) == "0"
    
    # Test even series
    even = LaurentSeries(coefficients={0: 1, 1: 2, -1: 3})
    assert "3z^-1" in str(even)
    assert "1" in str(even)
    assert "2z^1" in str(even)
    
    # Test odd series
    odd = LaurentSeries(odd_coefficients={0: 1, 1: 2})
    assert "ζ" in str(odd)
    assert "2ζz^1" in str(odd)

def test_laurent_series_addition():
    # Create two series
    s1 = LaurentSeries(coefficients={0: 1, 1: 2}, odd_coefficients={0: 3})
    s2 = LaurentSeries(coefficients={1: 1, 2: 2}, odd_coefficients={0: -3})
    
    # Add them
    result = s1 + s2
    
    # Check results
    assert result._even_coeffs[0] == 1  # 1 + 0
    assert result._even_coeffs[1] == 3  # 2 + 1
    assert result._even_coeffs[2] == 2  # 0 + 2
    assert result._odd_coeffs[0] == 0   # 3 + (-3)

def test_series_order():
    series = LaurentSeries(
        coefficients={-2: 1, 0: 1, 3: 1},
        odd_coefficients={-1: 1, 2: 1}
    )
    assert series.order == 3
    assert series.min_order == -2

def test_small_coefficients():
    # Test that very small coefficients are treated as zero
    series = LaurentSeries(
        coefficients={1: 1e-16},
        odd_coefficients={1: 1e-16}
    )
    assert str(series) == "0"

def test_laurent_series_multiplication():
    # Test even * even
    s1 = LaurentSeries(coefficients={0: 1, 1: 2})
    s2 = LaurentSeries(coefficients={0: 3, 1: 4})
    result = s1 * s2
    assert result._even_coeffs[0] == 3  # 1 * 3
    assert result._even_coeffs[1] == 10  # 1 * 4 + 2 * 3
    assert result._even_coeffs[2] == 8  # 2 * 4
    assert not result._odd_coeffs  # Should have no odd terms

    # Test odd * odd
    s3 = LaurentSeries(odd_coefficients={0: 1})
    s4 = LaurentSeries(odd_coefficients={0: 2})
    result = s3 * s4
    assert result._even_coeffs[0] == -2  # -1 * 2 (minus from Koszul rule)
    assert not result._odd_coeffs  # Should have no odd terms

    # Test mixed even/odd
    s5 = LaurentSeries(coefficients={0: 2}, odd_coefficients={1: 3})
    s6 = LaurentSeries(coefficients={0: 4}, odd_coefficients={0: 5})
    result = s5 * s6
    assert result._even_coeffs[0] == 8  # 2 * 4
    assert result._even_coeffs[1] == -15  # -(3 * 5)
    assert result._odd_coeffs[0] == 10  # 2 * 5
    assert result._odd_coeffs[1] == 12  # 3 * 4

def test_scalar_multiplication():
    # Test scalar multiplication
    s = LaurentSeries(coefficients={0: 1, 1: 2}, odd_coefficients={0: 3, 1: 4})
    result = 2 * s
    assert result._even_coeffs[0] == 2
    assert result._even_coeffs[1] == 4
    assert result._odd_coeffs[0] == 6
    assert result._odd_coeffs[1] == 8

def test_negation():
    # Test negation
    s = LaurentSeries(coefficients={0: 1, 1: 2}, odd_coefficients={0: 3, 1: 4})
    result = -s
    assert result._even_coeffs[0] == -1
    assert result._even_coeffs[1] == -2
    assert result._odd_coeffs[0] == -3
    assert result._odd_coeffs[1] == -4

def test_z_derivative():
    # Test z derivative of even terms
    s = LaurentSeries(coefficients={0: 1, 1: 2, 2: 3})
    result = s.d_dz()
    assert result._even_coeffs[0] == 2  # d/dz(2z) = 2
    assert result._even_coeffs[1] == 6  # d/dz(3z^2) = 6z
    # assert 0 not in result._even_coeffs  # constant term should vanish
    
    # Test z derivative of odd terms
    s = LaurentSeries(odd_coefficients={1: 2, 2: 3})
    result = s.d_dz()
    assert result._odd_coeffs[0] == 2  # d/dz(2ζz) = 2ζ
    assert result._odd_coeffs[1] == 6  # d/dz(3ζz^2) = 6ζz

def test_zeta_derivative():
    # Test ζ derivative
    s = LaurentSeries(
        coefficients={0: 1, 1: 2},  # Even terms should vanish
        odd_coefficients={0: 3, 1: 4}  # Odd coefficients become even
    )
    result = s.d_dzeta()
    assert result._even_coeffs[0] == 3
    assert result._even_coeffs[1] == 4
    assert not result._odd_coeffs  # No odd terms in result
    
def test_super_derivative():
    # Test D_ζ = ∂/∂ζ + ζ∂/∂z on a simple example
    s = LaurentSeries(
        coefficients={1: 2},  # 2z
        odd_coefficients={0: 1}  # ζ
    )
    result = s.D_zeta()
    
    # Should give 1 + 2ζ
    assert result._even_coeffs[0] == 1  # From ∂/∂ζ of ζ
    assert result._odd_coeffs[0] == 2   # From ζ∂/∂z of 2z
    
    # Test on a more complex example
    s = LaurentSeries(
        coefficients={2: 1},  # z^2
        odd_coefficients={1: 1}  # ζz
    )
    result = s.D_zeta()
    
    # Should give z + 2ζz
    assert result._even_coeffs[1] == 1  # From ∂/∂ζ of ζz
    assert result._odd_coeffs[1] == 2   # From ζ∂/∂z of z^2

def test_z_integration():
    # Test integration of even terms
    s = LaurentSeries(coefficients={0: 2, 1: 3})  # 2 + 3z
    result = s.integrate_dz()
    assert result._even_coeffs[1] == 2  # ∫ 2 dz = 2z
    assert result._even_coeffs[2] == 1.5  # ∫ 3z dz = (3/2)z²
    
    # Test integration of odd terms
    s = LaurentSeries(odd_coefficients={0: 2, 1: 4})  # 2ζ + 4ζz
    result = s.integrate_dz()
    assert result._odd_coeffs[1] == 2  # ∫ 2ζ dz = 2ζz
    assert result._odd_coeffs[2] == 2  # ∫ 4ζz dz = 2ζz²

def test_z_integration_with_negative_powers():
    # Test integration of negative powers (except -1)
    s = LaurentSeries(coefficients={-2: 2})  # 2z^(-2)
    result = s.integrate_dz()
    assert result._even_coeffs[-1] == -2  # ∫ 2z^(-2) dz = -2z^(-1)
    
    # Test that integrating z^(-1) raises an error
    s = LaurentSeries(coefficients={-1: 1})
    with pytest.raises(ValueError):
        s.integrate_dz()

def test_residue():
    # Test residue of series with z^(-1) term
    s = LaurentSeries(coefficients={-1: 2, 0: 1, 1: 3})  # 2z^(-1) + 1 + 3z
    assert s.residue() == 2
    
    # Test residue of series without z^(-1) term
    s = LaurentSeries(coefficients={0: 1, 1: 3})  # 1 + 3z
    assert s.residue() == 0
    
    # Test that odd terms don't contribute to residue
    s = LaurentSeries(
        coefficients={-1: 2},  # 2z^(-1)
        odd_coefficients={-1: 3}  # 3ζz^(-1)
    )
    assert s.residue() == 2

def test_contour_integration():
    # Test contour integration of series with residue
    s = LaurentSeries(coefficients={-1: 1})  # z^(-1)
    assert abs(s.contour_integrate() - 2j * np.pi) < 1e-10
    
    # Test contour integration of series without residue
    s = LaurentSeries(coefficients={0: 1, 1: 2})  # 1 + 2z
    assert abs(s.contour_integrate()) < 1e-10
    
    # Test that odd terms don't contribute to contour integral
    s = LaurentSeries(
        coefficients={-1: 1},  # z^(-1)
        odd_coefficients={-1: 2}  # 2ζz^(-1)
    )
    assert abs(s.contour_integrate() - 2j * np.pi) < 1e-10
    
    # Test scaling of contour integral
    s = LaurentSeries(coefficients={-1: 2})  # 2z^(-1)
    assert abs(s.contour_integrate() - 4j * np.pi) < 1e-10