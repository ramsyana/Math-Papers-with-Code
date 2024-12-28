from typing import Union, Optional
from .log_laurent_series import LogLaurentSeries

class GradedDifferential:
    """
    A class representing j/2-differentials of the form:
    f(z|ζ)[dz|dζ]⊗j where:
    - f(z|ζ) is a log Laurent series
    - [dz|dζ] is odd
    - j is the tensor power
    """
    def __init__(self, series: LogLaurentSeries, j: int):
        """
        Initialize a j/2-differential.
        
        Args:
            series: The coefficient Laurent series
            j: The tensor power of [dz|dζ]
        
        Raises:
            ValueError: If j is 0 or invalid
        """
        if j == 0:
            raise ValueError("j must be non-zero")
            
        self._series = series
        self._j = j
        self._grade = j/2
        
    @property
    def grade(self) -> float:
        """The j/2 grading of the differential."""
        return self._grade
        
    def is_even(self) -> bool:
        """Check if j is even, making [dz|dζ]⊗j even."""
        return self._j % 2 == 0
        
    def is_odd(self) -> bool:
        """Check if j is odd, making [dz|dζ]⊗j odd."""
        return self._j % 2 == 1
        
    def total_parity(self) -> int:
        """
        Compute total parity of f(z|ζ)[dz|dζ]⊗j.
        Returns:
            0 for even, 1 for odd parity
        """
        # If we have odd terms in series, it contributes 1 to parity
        series_parity = 1 if self._series._odd_terms else 0
        # j odd means [dz|dζ]⊗j is odd, contributing 1 to parity
        form_parity = self._j % 2
        # Total parity is sum mod 2
        return (series_parity + form_parity) % 2
    
    def __mul__(self, other: 'GradedDifferential') -> 'GradedDifferential':
        """
        Multiply two graded differentials using:
        (f[dz|dζ]⊗j)(g[dz|dζ]⊗k) = (f·g)[dz|dζ]⊗(j+k)
        """
        # Multiply the series parts
        new_series = self._series * other._series
        # Add the tensor powers
        new_j = self._j + other._j
        
        return GradedDifferential(new_series, new_j)
        
    def tensor_power(self, n: int) -> 'GradedDifferential':
        """
        Compute the nth tensor power of the differential.
        
        Args:
            n: The tensor power (must be positive)
            
        Returns:
            The tensor power as a new GradedDifferential
            
        Raises:
            ValueError: If n is not positive
        """
        if n <= 0:
            raise ValueError("Tensor power must be positive")
            
        # Keep the same series, only multiply j
        new_j = self._j * n
        
        return GradedDifferential(self._series, new_j)
        
    def get_coefficient(self, power: int) -> complex:
        """Get coefficient of z^power in the series."""
        # Sum coefficients from even and odd terms
        coeff = 0
        for terms in self._series._even_terms.values():
            if power in terms:
                coeff += terms[power]
        for terms in self._series._odd_terms.values():
            if power in terms:
                coeff += terms[power]
        return coeff
        
    def __add__(self, other: 'GradedDifferential') -> 'GradedDifferential':
        """Add two graded differentials."""
        if self._j != other._j:
            raise ValueError("Cannot add differentials of different grades")
        return GradedDifferential(self._series + other._series, self._j)