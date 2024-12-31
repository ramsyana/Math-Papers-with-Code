from typing import Optional
import logging
from .log_laurent_series import LogLaurentSeries
from .log_laurent_derivatives import d_dz, d_dzeta, D_zeta
from .graded_differential import GradedDifferential

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def witt_action(f: LogLaurentSeries, g: GradedDifferential) -> GradedDifferential:
    print("\n--- Witt Action Detailed Debug ---")
    print(f"Input vector field f: {f}")
    print(f"Input differential g: {g._series}")
    print(f"Differential grade j: {g._j}")
    print(f"Is f constant? {f.is_constant()}")
    print(f"Is g zero? {g._series.is_zero()}")

    # Detailed zero input handling
    if f.is_constant() or g._series.is_zero():
        print("Returning zero differential due to constant field or zero series")
        return GradedDifferential(LogLaurentSeries(), g._j)
    
    # Compute derivatives with logging
    D_zeta_g = D_zeta(g._series)
    print(f"D_ζ(g): {D_zeta_g}")

    D_zeta_D_zeta_g = D_zeta(D_zeta_g)
    print(f"D_ζ²(g): {D_zeta_D_zeta_g}")

    # Precise commutator computation
    term1 = f * D_zeta_D_zeta_g
    print(f"Term 1 (f * D_ζ²(g)): {term1}")

    f_D_zeta_g = f * D_zeta_g
    print(f"f * D_ζ(g): {f_D_zeta_g}")

    term2 = D_zeta(f_D_zeta_g)
    print(f"Term 2 (D_ζ(f * D_ζ(g))): {term2}")

    # Algebraically correct commutator
    commutator = term1 - term2
    print(f"Commutator: {commutator}")

    # Correct scale term computation
    df_dz = d_dz(f)
    print(f"df/dz: {df_dz}")

    scale_term = (df_dz * g._series) * g._j
    print(f"Scale term (multiplied by full j): {scale_term}")

    # Combine terms
    result_series = commutator + scale_term
    print(f"Result series: {result_series}")
    print(f"Coefficients of z²: {result_series._even_terms.get(0, {}).get(2, 0)}")

    return GradedDifferential(result_series, g._j)


def witt_bracket(f: LogLaurentSeries, h: LogLaurentSeries) -> LogLaurentSeries:
    """
    Compute the Witt algebra bracket [[fDζ, Dζ], [hDζ, Dζ]].

    Args:
        f: First vector field coefficient
        h: Second vector field coefficient

    Returns:
        LogLaurentSeries: Coefficient of the resulting vector field
    """
    logger.debug("\n==== Starting Witt bracket computation ====")
    logger.debug(f"Input f: {f}")
    logger.debug(f"Input h: {h}")

    # Compute derivatives
    D_zeta_h = D_zeta(h)
    D_zeta_f = D_zeta(f)

    # Compute first term with Leibniz terms
    first_action = (
        f * D_zeta(D_zeta_h)
        - D_zeta(f * D_zeta_h)
        + h * D_zeta(D_zeta_f) 
    )

    second_action = (
        h * D_zeta(D_zeta_f)
        - D_zeta(h * D_zeta_f)
        + f * D_zeta(D_zeta_h)
    )

    # Return the bracket result
    result = first_action - second_action
    logger.debug(f"Result: {result}")
    return result


def verify_jacobi(f: LogLaurentSeries, g: LogLaurentSeries, h: LogLaurentSeries) -> bool:
    """
    Verify the Jacobi identity for three Witt algebra elements.
    
    Args:
        f, g, h: Vector field coefficients
        
    Returns:
        bool: True if Jacobi identity is satisfied
    """
    # Compute cyclic sum [X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]]
    term1 = witt_bracket(f, witt_bracket(g, h))
    term2 = witt_bracket(g, witt_bracket(h, f))
    term3 = witt_bracket(h, witt_bracket(f, g))
    
    result = term1 + term2 + term3
    
    # Check if result is effectively zero
    return all(abs(coeff) < 1e-10 
              for terms in result._even_terms.values() 
              for coeff in terms.values()) and \
           all(abs(coeff) < 1e-10 
              for terms in result._odd_terms.values() 
              for coeff in terms.values())