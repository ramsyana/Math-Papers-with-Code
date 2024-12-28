from typing import Dict, Union, Optional, Tuple
from collections import defaultdict

class LogLaurentSeries:
    """
    A class representing formal Laurent series with logarithmic terms of the form:
    Σ (an + αnζ)z^n * log^k(z) where:
    - an are even coefficients
    - αn are odd coefficients
    - k is the power of log(z)
    """
    def __init__(self, 
                 log_terms: Dict[int, Dict[int, Union[float, complex]]] = None,
                 odd_log_terms: Dict[int, Dict[int, Union[float, complex]]] = None,
                 truncation_order: int = 10):
        """
        Initialize a Laurent series with log terms.
        
        Args:
            log_terms: Dict[log_power, Dict[z_power, coefficient]] for even terms
            odd_log_terms: Dict[log_power, Dict[z_power, coefficient]] for odd terms
            truncation_order: Maximum order for series truncation
        """
        # Structure: {log_power: {z_power: coefficient}}
        self._even_terms = defaultdict(lambda: defaultdict(float))
        self._odd_terms = defaultdict(lambda: defaultdict(float))
        self._truncation = truncation_order
        
        if log_terms:
            for log_power, coeffs in log_terms.items():
                for z_power, coeff in coeffs.items():
                    if abs(coeff) > 1e-15:  # Numerical threshold
                        self._even_terms[log_power][z_power] = coeff
                    
        if odd_log_terms:
            for log_power, coeffs in odd_log_terms.items():
                for z_power, coeff in coeffs.items():
                    if abs(coeff) > 1e-15:  # Numerical threshold
                        self._odd_terms[log_power][z_power] = coeff
    
    @property
    def max_log_power(self) -> int:
        """Maximum power of log terms."""
        even_max = max(self._even_terms.keys()) if self._even_terms else 0
        odd_max = max(self._odd_terms.keys()) if self._odd_terms else 0
        return max(even_max, odd_max)

    def __str__(self):
        """String representation matching expected test format"""
        return ' + '.join(f"{k}: {dict(v)}" for k, v in self._even_terms.items()) or "0"

    def __add__(self, other: 'LogLaurentSeries') -> 'LogLaurentSeries':
        """Add two log Laurent series."""
        result_even = defaultdict(lambda: defaultdict(float))
        result_odd = defaultdict(lambda: defaultdict(float))
        
        # Combine coefficients with same log powers
        all_log_powers = set(self._even_terms.keys()) | set(other._even_terms.keys())
        for log_power in all_log_powers:
            # Even terms
            all_powers = set(self._even_terms[log_power].keys()) | \
                        set(other._even_terms[log_power].keys())
            for power in all_powers:
                coeff = (self._even_terms[log_power][power] + 
                        other._even_terms[log_power][power])
                if abs(coeff) > 1e-15:
                    result_even[log_power][power] = coeff
        
        # Similar for odd terms
        all_log_powers = set(self._odd_terms.keys()) | set(other._odd_terms.keys())
        for log_power in all_log_powers:
            all_powers = set(self._odd_terms[log_power].keys()) | \
                        set(other._odd_terms[log_power].keys())
            for power in all_powers:
                coeff = (self._odd_terms[log_power][power] + 
                        other._odd_terms[log_power][power])
                if abs(coeff) > 1e-15:
                    result_odd[log_power][power] = coeff
        
        # Convert defaultdict to regular dict for the constructor
        return LogLaurentSeries(
            log_terms=dict(result_even),
            odd_log_terms=dict(result_odd),
            truncation_order=min(self._truncation, other._truncation)
        )

    def __mul__(self, other) -> 'LogLaurentSeries':
        """Multiply log Laurent series and handle scalar multiplication"""
        if isinstance(other, (int, float, complex)):
            # Handle scalar multiplication
            result_even = {}
            result_odd = {}
            
            # Scale even terms
            for log_power in self._even_terms:
                result_even[log_power] = {
                    z_power: coeff * other 
                    for z_power, coeff in self._even_terms[log_power].items()
                }
                
            # Scale odd terms 
            for log_power in self._odd_terms:
                result_odd[log_power] = {
                    z_power: coeff * other
                    for z_power, coeff in self._odd_terms[log_power].items()
                }
                
            return LogLaurentSeries(
                log_terms=result_even,
                odd_log_terms=result_odd,
                truncation_order=self._truncation
            )
        
        # Original multiplication code for two LogLaurentSeries
        if not isinstance(other, LogLaurentSeries):
            return NotImplemented
        
        result_even = defaultdict(lambda: defaultdict(float))
        result_odd = defaultdict(lambda: defaultdict(float))
        
        # Multiply term by term, handling both z powers and log powers
        for log1, terms1 in self._even_terms.items():
            for log2, terms2 in other._even_terms.items():
                # Even * Even terms
                for p1, c1 in terms1.items():
                    for p2, c2 in terms2.items():
                        z_power = p1 + p2
                        if abs(z_power) <= self._truncation:
                            log_power = log1 + log2
                            result_even[log_power][z_power] += c1 * c2
            
            # Even * Odd terms
            for log2, terms2 in other._odd_terms.items():
                for p1, c1 in terms1.items():
                    for p2, c2 in terms2.items():
                        z_power = p1 + p2
                        if abs(z_power) <= self._truncation:
                            log_power = log1 + log2
                            result_odd[log_power][z_power] += c1 * c2
        
        # Odd * Even terms 
        for log1, terms1 in self._odd_terms.items():
            for log2, terms2 in other._even_terms.items():
                for p1, c1 in terms1.items():
                    for p2, c2 in terms2.items():
                        z_power = p1 + p2
                        if abs(z_power) <= self._truncation:
                            log_power = log1 + log2
                            result_odd[log_power][z_power] += c1 * c2
        
        # Odd * Odd terms (gives even with minus sign)
        for log1, terms1 in self._odd_terms.items():
            for log2, terms2 in other._odd_terms.items():
                for p1, c1 in terms1.items():
                    for p2, c2 in terms2.items():
                        z_power = p1 + p2
                        if abs(z_power) <= self._truncation:
                            log_power = log1 + log2
                            result_even[log_power][z_power] -= c1 * c2

        # Remove any tiny coefficients
        result_even = {k: {p: c for p, c in v.items() if abs(c) > 1e-15}
                      for k, v in result_even.items() if v}
        result_odd = {k: {p: c for p, c in v.items() if abs(c) > 1e-15}
                     for k, v in result_odd.items() if v}

        return LogLaurentSeries(
            log_terms=result_even,
            odd_log_terms=result_odd,
            truncation_order=min(self._truncation, other._truncation)
        )

    def multiply_by_zeta(self, other: 'LogLaurentSeries') -> 'LogLaurentSeries':
        """Multiply series by ζ, effectively swapping even and odd terms"""
        result_even = defaultdict(lambda: defaultdict(float))
        result_odd = defaultdict(lambda: defaultdict(float))
        
        # Odd terms of other become even terms
        for log_power, terms in other._odd_terms.items():
            for z_power, coeff in terms.items():
                result_even[log_power][z_power] = coeff
        
        # Even terms of other become odd terms
        for log_power, terms in other._even_terms.items():
            for z_power, coeff in terms.items():
                result_odd[log_power][z_power] = coeff
        
        return LogLaurentSeries(
            log_terms=dict(result_even),
            odd_log_terms=dict(result_odd),
            truncation_order=other._truncation
        )

    def __sub__(self, other: 'LogLaurentSeries') -> 'LogLaurentSeries':
        """Subtract two log Laurent series."""
        result_even = defaultdict(lambda: defaultdict(float))
        result_odd = defaultdict(lambda: defaultdict(float))
        
        # Handle even terms
        all_log_powers = set(self._even_terms.keys()) | set(other._even_terms.keys())
        for log_power in all_log_powers:
            all_powers = set(self._even_terms[log_power].keys()) | set(other._even_terms[log_power].keys())
            for power in all_powers:
                coeff = (self._even_terms[log_power][power] - other._even_terms[log_power][power])
                if abs(coeff) > 1e-15:
                    result_even[log_power][power] = coeff
        
        # Handle odd terms
        all_log_powers = set(self._odd_terms.keys()) | set(other._odd_terms.keys())
        for log_power in all_log_powers:
            all_powers = set(self._odd_terms[log_power].keys()) | set(other._odd_terms[log_power].keys())
            for power in all_powers:
                coeff = (self._odd_terms[log_power][power] - other._odd_terms[log_power][power])
                if abs(coeff) > 1e-15:
                    result_odd[log_power][power] = coeff
                    
        return LogLaurentSeries(log_terms=dict(result_even), odd_log_terms=dict(result_odd))