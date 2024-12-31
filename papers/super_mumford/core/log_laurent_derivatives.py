from typing import Optional
from .log_laurent_series import LogLaurentSeries

def d_dz(series: LogLaurentSeries) -> LogLaurentSeries:
    """z-derivative of log Laurent series"""
    result_even = {}
    result_odd = {}

    # Handle even terms
    for log_power, z_terms in series._even_terms.items():
        result_even[log_power] = {}
        for z_power, coeff in z_terms.items():
            # Chain rule: d/dz(z^n * log^k(z))
            # = n*z^(n-1)*log^k(z) + k*z^(n-1)*log^(k-1)(z)
            if z_power != 0:
                result_even[log_power][z_power - 1] = z_power * coeff

            if log_power > 0:  # Term with log(z)
                if log_power - 1 not in result_even:
                    result_even[log_power - 1] = {}
                if z_power - 1 not in result_even[log_power - 1]:
                    result_even[log_power - 1][z_power - 1] = 0
                result_even[log_power - 1][z_power - 1] += log_power * coeff

    # Handle odd terms similarly
    for log_power, z_terms in series._odd_terms.items():
        result_odd[log_power] = {}
        for z_power, coeff in z_terms.items():
            if z_power != 0:
                result_odd[log_power][z_power - 1] = z_power * coeff

            if log_power > 0:
                if log_power - 1 not in result_odd:
                    result_odd[log_power - 1] = {}
                if z_power - 1 not in result_odd[log_power - 1]:
                    result_odd[log_power - 1][z_power - 1] = 0
                result_odd[log_power - 1][z_power - 1] += log_power * coeff

    return LogLaurentSeries(log_terms=result_even, odd_log_terms=result_odd)

def d_dzeta(series: LogLaurentSeries) -> LogLaurentSeries:
    """ζ-derivative of log Laurent series"""
    # Transform even terms to odd and odd terms to even (with sign change)
    result_even = {}  # d/dζ of odd terms become even
    for k, v in series._odd_terms.items():
        result_even[k] = v
    result_odd = {}  # d/dζ of even terms = 0

    return LogLaurentSeries(log_terms=result_even, odd_log_terms=result_odd)

def D_zeta(series: LogLaurentSeries) -> LogLaurentSeries:
    """Super derivative D_ζ = ∂/∂ζ + ζ∂/∂z"""
    d_dzeta_result = d_dzeta(series)
    d_dz_result = d_dz(series)
    
    # Multiply d_dz_result by ζ using the multiply_by_zeta method
    zeta_d_dz_result = series.multiply_by_zeta(d_dz_result)
    
    return d_dzeta_result + zeta_d_dz_result