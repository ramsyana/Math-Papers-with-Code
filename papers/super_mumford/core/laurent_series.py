from typing import Dict, Union, Optional
import numpy as np
from collections import defaultdict

class LaurentSeries:
    """
    A class representing formal Laurent series in z and ζ of the form:
    Σ (an + αnζ)z^n where an are even and αn are odd coefficients.
    """
    def __init__(self, 
                 coefficients: Dict[int, Union[float, complex]] = None,
                 odd_coefficients: Dict[int, Union[float, complex]] = None,
                 truncation_order: int = 10):
        """
        Initialize a Laurent series.
        
        Args:
            coefficients: Dictionary mapping powers to even coefficients
            odd_coefficients: Dictionary mapping powers to odd coefficients (ζ terms)
            truncation_order: Maximum order for series truncation
        """
        self._even_coeffs = defaultdict(float)
        self._odd_coeffs = defaultdict(float)
        self._truncation = truncation_order
        
        if coefficients:
            for power, coeff in coefficients.items():
                if abs(coeff) > 1e-15:  # Numerical threshold
                    self._even_coeffs[power] = coeff
                    
        if odd_coefficients:
            for power, coeff in odd_coefficients.items():
                if abs(coeff) > 1e-15:  # Numerical threshold
                    self._odd_coeffs[power] = coeff

    @property
    def order(self) -> int:
        """Maximum non-zero coefficient power."""
        even_max = max(self._even_coeffs.keys()) if self._even_coeffs else float('-inf')
        odd_max = max(self._odd_coeffs.keys()) if self._odd_coeffs else float('-inf')
        return int(max(even_max, odd_max))

    @property
    def min_order(self) -> int:
        """Minimum non-zero coefficient power."""
        even_min = min(self._even_coeffs.keys()) if self._even_coeffs else float('inf')
        odd_min = min(self._odd_coeffs.keys()) if self._odd_coeffs else float('inf')
        return int(min(even_min, odd_min))

    def __add__(self, other: 'LaurentSeries') -> 'LaurentSeries':
        """Add two Laurent series."""
        result_even = defaultdict(float)
        result_odd = defaultdict(float)
        
        # Combine even coefficients
        for power in set(self._even_coeffs.keys()) | set(other._even_coeffs.keys()):
            coeff = self._even_coeffs[power] + other._even_coeffs[power]
            if abs(coeff) > 1e-15:
                result_even[power] = coeff
                
        # Combine odd coefficients
        for power in set(self._odd_coeffs.keys()) | set(other._odd_coeffs.keys()):
            coeff = self._odd_coeffs[power] + other._odd_coeffs[power]
            if abs(coeff) > 1e-15:
                result_odd[power] = coeff
        
        return LaurentSeries(
            coefficients=dict(result_even),
            odd_coefficients=dict(result_odd),
            truncation_order=min(self._truncation, other._truncation)
        )

    def __mul__(self, other: 'LaurentSeries') -> 'LaurentSeries':
        """
        Multiply two Laurent series following the Koszul rule for signs.
        The product follows the pattern:
        (a + αζ)(b + βζ) = ab + (αb + aβ)ζ
        """
        result_even = defaultdict(float)
        result_odd = defaultdict(float)
        
        # Handle even * even terms (ab)
        for p1, c1 in self._even_coeffs.items():
            for p2, c2 in other._even_coeffs.items():
                power = p1 + p2
                if abs(power) <= self._truncation:
                    result_even[power] += c1 * c2

        # Handle odd * odd terms (-αβ)
        # Note: Product of two odd elements is even with a minus sign
        for p1, c1 in self._odd_coeffs.items():
            for p2, c2 in other._odd_coeffs.items():
                power = p1 + p2
                if abs(power) <= self._truncation:
                    result_even[power] -= c1 * c2  # Minus sign from Koszul rule
        
        # Handle mixed even * odd terms
        # First term's even * Second term's odd (aβ)
        for p1, c1 in self._even_coeffs.items():
            for p2, c2 in other._odd_coeffs.items():
                power = p1 + p2
                if abs(power) <= self._truncation:
                    result_odd[power] += c1 * c2

        # First term's odd * Second term's even (αb)
        for p1, c1 in self._odd_coeffs.items():
            for p2, c2 in other._even_coeffs.items():
                power = p1 + p2
                if abs(power) <= self._truncation:
                    result_odd[power] += c1 * c2

        # Remove any tiny coefficients that may have accumulated from floating point ops
        result_even = {k: v for k, v in result_even.items() if abs(v) > 1e-15}
        result_odd = {k: v for k, v in result_odd.items() if abs(v) > 1e-15}

        return LaurentSeries(
            coefficients=result_even,
            odd_coefficients=result_odd,
            truncation_order=min(self._truncation, other._truncation)
        )

    def __rmul__(self, scalar: Union[int, float, complex]) -> 'LaurentSeries':
        """
        Multiply the series by a scalar from the left.
        """
        result_even = {k: scalar * v for k, v in self._even_coeffs.items()}
        result_odd = {k: scalar * v for k, v in self._odd_coeffs.items()}
        
        return LaurentSeries(
            coefficients=result_even,
            odd_coefficients=result_odd,
            truncation_order=self._truncation
        )

    def __neg__(self) -> 'LaurentSeries':
        """
        Return the negation of the series.
        """
        return self.__rmul__(-1)

    def d_dz(self) -> 'LaurentSeries':
        """
        Compute the partial derivative with respect to z.
        For a series Σ (an + αnζ)z^n, returns Σ (n*an + n*αnζ)z^(n-1)
        """
        result_even = defaultdict(float)
        result_odd = defaultdict(float)
        
        # Differentiate even terms
        for power, coeff in self._even_coeffs.items():
            if power != 0:  # Skip constant terms as they vanish
                new_power = power - 1
                if abs(new_power) <= self._truncation:
                    result_even[new_power] = power * coeff
                    
        # Differentiate odd terms
        for power, coeff in self._odd_coeffs.items():
            if power != 0:  # Skip constant terms as they vanish
                new_power = power - 1
                if abs(new_power) <= self._truncation:
                    result_odd[new_power] = power * coeff
        
        return LaurentSeries(
            coefficients=dict(result_even),
            odd_coefficients=dict(result_odd),
            truncation_order=self._truncation
        )

    def d_dzeta(self) -> 'LaurentSeries':
        """
        Compute the partial derivative with respect to ζ.
        For a series Σ (an + αnζ)z^n, returns Σ αnz^n
        Note: Result is even as we lose the ζ term.
        """
        result_even = {power: coeff for power, coeff in self._odd_coeffs.items() 
                      if abs(power) <= self._truncation}
        
        return LaurentSeries(
            coefficients=result_even,
            truncation_order=self._truncation
        )

    def D_zeta(self) -> 'LaurentSeries':
        """
        Compute the super derivative D_ζ = ∂/∂ζ + ζ∂/∂z.
        This is key for the superconformal structure.
        """
        # First term: ∂/∂ζ
        result = self.d_dzeta()
        
        # Second term: ζ∂/∂z
        dz_term = self.d_dz()
        zeta_dz = LaurentSeries(
            odd_coefficients={p: c for p, c in dz_term._even_coeffs.items()},
            coefficients={p: c for p, c in dz_term._odd_coeffs.items()},
            truncation_order=self._truncation
        )
        
        return result + zeta_dz

    def __str__(self) -> str:
        """String representation of the series."""
        terms = []
        
        # Add even terms
        for power, coeff in sorted(self._even_coeffs.items()):
            if abs(coeff) > 1e-15:
                if power == 0:
                    terms.append(f"{coeff:g}")
                else:
                    terms.append(f"{coeff:g}z^{power}")
                    
        # Add odd terms
        for power, coeff in sorted(self._odd_coeffs.items()):
            if abs(coeff) > 1e-15:
                if power == 0:
                    terms.append(f"{coeff:g}ζ")
                else:
                    terms.append(f"{coeff:g}ζz^{power}")
                    
        return " + ".join(terms) if terms else "0"