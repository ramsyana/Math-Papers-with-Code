from typing import Optional
from .log_laurent_series import LogLaurentSeries

def integrate_z(series: LogLaurentSeries) -> LogLaurentSeries:
    """Indefinite z-integration of log Laurent series"""
    result_even = {}
    result_odd = {}
    
    # Handle even terms
    for log_power, z_terms in series._even_terms.items():
        for z_power, coeff in z_terms.items():
            # log(z) term needs special handling
            if log_power > 0:
                # Term with log^k(z)
                if log_power not in result_even:
                    result_even[log_power] = {}
                result_even[log_power][z_power + 1] = coeff / (z_power + 1)
                
                # Fixed integration by parts term
                if log_power - 1 not in result_even:
                    result_even[log_power - 1] = {}
                if z_power + 1 not in result_even[log_power - 1]:
                    result_even[log_power - 1][z_power + 1] = 0
                result_even[log_power - 1][z_power + 1] -= coeff * log_power / ((z_power + 1) * (z_power + 1))
            else:
                if z_power == -1:  # Special case: ∫ dz/z = log(z)
                    if 1 not in result_even:
                        result_even[1] = {}
                    result_even[1][0] = result_even[1].get(0, 0) + coeff
                else:  # Standard case
                    if 0 not in result_even:
                        result_even[0] = {}
                    result_even[0][z_power + 1] = coeff / (z_power + 1)

    # Similar for odd terms
    for log_power, z_terms in series._odd_terms.items():
        result_odd[log_power] = {}
        for z_power, coeff in z_terms.items():
            if z_power == -1:  # Special case: ∫ dζ/z = ζ*log(z)
                if 1 not in result_odd:
                    result_odd[1] = {}
                result_odd[1][0] = result_odd[1].get(0, 0) + coeff
            else:
                result_odd[log_power][z_power + 1] = coeff / (z_power + 1)

    return LogLaurentSeries(log_terms=result_even, odd_log_terms=result_odd)
