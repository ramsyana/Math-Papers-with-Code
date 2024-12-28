import logging
from typing import Optional
from .log_laurent_series import LogLaurentSeries
from .log_laurent_derivatives import d_dz, d_dzeta, D_zeta
from .graded_differential import GradedDifferential

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def lie_derivative(f: LogLaurentSeries, g: GradedDifferential) -> GradedDifferential:
    """
    Compute the Lie derivative of g[dz|dζ]⊗j with respect to [fDζ, Dζ].
    The formula is:
    ρ([fDζ, Dζ])g[dz|dζ]⊗j = ([fDζ, Dζ]g + (j/2)∂f/∂z g)[dz|dζ]⊗j
    """
    print(f"Input f: {f}")
    print(f"Input g: {g}")
    print(f"g.grade: {g.grade}")
    
    g_series = g._series
    
    # Compute derivatives
    df_dz = d_dz(f)
    print(f"df/dz: {df_dz}")
    print(f"g_series: {g_series}")
    
    # Detailed multiplication tracking
    print("Multiplication breakdown:")
    print(f"df_dz coeffs: {df_dz._even_terms}")
    print(f"g_series coeffs: {g_series._even_terms}")
    print(f"g.grade: {g.grade}")
    
    # Try explicit coefficient computation
    scale_term_coeffs = {}
    for log_power, z_terms in df_dz._log_terms.items():
        for z_power, coeff in z_terms.items():
            for g_z_power, g_coeffs in g_series._log_terms.items():
                for inner_z_power, g_coeff in g_coeffs.items():
                    new_power = z_power + inner_z_power
                    new_coeff = coeff * g_coeff * (1 if g_z_power == 0 else g_z_power * g.grade)
                    
                    # Initialize nested dictionaries if not exist
                    if log_power not in scale_term_coeffs:
                        scale_term_coeffs[log_power] = {}
                    
                    if new_power not in scale_term_coeffs[log_power]:
                        scale_term_coeffs[log_power][new_power] = 0
                    
                    scale_term_coeffs[log_power][new_power] += new_coeff

    # Reconstruct LogLaurentSeries with computed coefficients
    scale_term = LogLaurentSeries(log_terms=scale_term_coeffs)
    
    print(f"Scale term: {scale_term}")
    
    # First compute the derivative terms
    D_ζg = D_zeta(g_series)
    logging.debug(f"D_ζg: {D_ζg}")

    f_D_ζg = f * D_ζg
    logging.debug(f"f_D_ζg: {f_D_ζg}")

    # First term includes both parts of D_ζ
    term1a = d_dzeta(f_D_ζg)
    logging.debug(f"term1a (d_dzeta(f_D_ζg)): {term1a}")

    term1b = f_D_ζg.multiply_by_zeta(d_dz(f_D_ζg))
    logging.debug(f"term1b (f_D_ζg.multiply_by_zeta(d_dz(f_D_ζg))): {term1b}")

    term1 = term1a + term1b
    logging.debug(f"term1 (term1a + term1b): {term1}")

    # Second term: f·D_ζ²(g)
    D_ζD_ζg = D_zeta(D_zeta(g_series))
    logging.debug(f"D_ζD_ζg: {D_ζD_ζg}")

    term2 = f * D_ζD_ζg
    logging.debug(f"term2 (f * D_ζD_ζg): {term2}")

    # Compute the commutator
    commutator = term1 - term2
    logging.debug(f"commutator (term1 - term2): {commutator}")

    # Ensure the scale term is added correctly
    result_series = scale_term + commutator 
    logging.debug(f"result_series (scale_term + commutator): {result_series}")

    return GradedDifferential(result_series, g._j)

def lie_bracket(f: LogLaurentSeries, h: LogLaurentSeries) -> LogLaurentSeries:
    """
    Compute the Lie bracket [[fDζ, Dζ], [hDζ, Dζ]] acting on H•.
    """
    logging.debug(f"Starting lie_bracket with f: {f}, h: {h}")

    # Special handling for zero inputs
    if f.is_zero() or h.is_zero():
        return h.copy() if f.is_zero() else f.copy()
    
    # Compute first bracket [fDζ, Dζ]h
    D_ζh = D_zeta(h)
    logging.debug(f"D_ζh: {D_ζh}")

    f_D_ζh = f * D_ζh
    logging.debug(f"f_D_ζh: {f_D_ζh}")

    # First term of first bracket
    term1a = d_dzeta(f_D_ζh)
    logging.debug(f"term1a (d_dzeta(f_D_ζh)): {term1a}")

    term1b = f_D_ζh.multiply_by_zeta(d_dz(f_D_ζh))
    logging.debug(f"term1b (f_D_ζh.multiply_by_zeta(d_dz(f_D_ζh))): {term1b}")

    term1 = term1a + term1b
    logging.debug(f"term1 (term1a + term1b): {term1}")

    # Second term of first bracket
    D_ζD_ζh = D_zeta(D_ζh)
    logging.debug(f"D_ζD_ζh: {D_ζD_ζh}")

    term2 = f * D_ζD_ζh
    logging.debug(f"term2 (f * D_ζD_ζh): {term2}")

    first_action = term1 - term2
    logging.debug(f"first_action (term1 - term2): {first_action}")

    # Compute second bracket [hDζ, Dζ]f 
    D_ζf = D_zeta(f)
    logging.debug(f"D_ζf: {D_ζf}")

    h_D_ζf = h * D_ζf
    logging.debug(f"h_D_ζf: {h_D_ζf}")

    # First term of second bracket
    term3a = d_dzeta(h_D_ζf)
    logging.debug(f"term3a (d_dzeta(h_D_ζf)): {term3a}")

    term3b = h_D_ζf.multiply_by_zeta(d_dz(h_D_ζf))
    logging.debug(f"term3b (h_D_ζf.multiply_by_zeta(d_dz(h_D_ζf))): {term3b}")

    term3 = term3a + term3b
    logging.debug(f"term3 (term3a + term3b): {term3}")

    # Second term of second bracket
    D_ζD_ζf = D_zeta(D_ζf)
    logging.debug(f"D_ζD_ζf: {D_ζD_ζf}")

    term4 = h * D_ζD_ζf
    logging.debug(f"term4 (h * D_ζD_ζf): {term4}")

    second_action = term3 - term4
    logging.debug(f"second_action (term3 - term4): {second_action}")

    # Complete Lie bracket is difference of actions
    result = first_action - second_action 
    logging.debug(f"result (first_action - second_action): {result}")

    return result