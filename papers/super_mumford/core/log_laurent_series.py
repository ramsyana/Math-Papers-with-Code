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

    def __str__(self) -> str:
        """String representation of the log Laurent series."""
        terms = []
        
        # Add even terms
        for log_power, z_terms in sorted(self._even_terms.items()):
            for z_power, coeff in sorted(z_terms.items()):
                if abs(coeff) > 1e-15:
                    term = f"{coeff:g}"
                    if z_power != 0:
                        if z_power == 1:
                            term += "z"
                        else:
                            term += f"z^{z_power}"
                    if log_power > 0:
                        if z_power != 0:
                            term += "*"
                        if log_power == 1:
                            term += "log(z)"
                        else:
                            term += f"log^{log_power}(z)"
                    terms.append(term)
        
        # Add odd terms
        for log_power, z_terms in sorted(self._odd_terms.items()):
            for z_power, coeff in sorted(z_terms.items()):
                if abs(coeff) > 1e-15:
                    term = f"{coeff:g}ζ"
                    if z_power != 0:
                        if z_power == 1:
                            term += "*z"
                        else: 
                            term += f"*z^{z_power}"
                    if log_power > 0:
                        term += "*"
                        if log_power == 1:
                            term += "log(z)"
                        else:
                            term += f"log^{log_power}(z)"
                    terms.append(term)
                    
        return " + ".join(terms) if terms else "0"

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

    def __mul__(self, other: 'LogLaurentSeries') -> 'LogLaurentSeries':
        """
        Multiply two log Laurent series following the Koszul rule.
        For log terms, we use: log^a(z) * log^b(z) = log^(a+b)(z)
        """
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