from typing import Optional
import logging
from .log_laurent_series import LogLaurentSeries
from .log_laurent_derivatives import d_dz, d_dzeta, D_zeta
from .graded_differential import GradedDifferential

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def witt_action(f: LogLaurentSeries, g: GradedDifferential) -> GradedDifferential:
    """
    Compute the action of a super Witt algebra element [fDζ, Dζ] on g[dz|dζ]⊗j.
    
    The action is given by:
    ρ([fDζ, Dζ])g[dz|dζ]⊗j = ([fDζ, Dζ]g + (j/2)∂f/∂z g)[dz|dζ]⊗j
    
    Args:
        f: The coefficient series of the vector field [fDζ, Dζ]
        g: The graded j/2-differential being acted upon
        
    Returns:
        GradedDifferential: Result of the Witt algebra action
    """
    logger.debug("\n==== Starting Witt algebra action computation ====")
    logger.debug(f"Input vector field f: {f}")
    logger.debug(f"Input differential g: {g._series}")
    logger.debug(f"Grade j/2: {g.grade}")
    
    g_series = g._series
    
    # First compute the commutator action [fDζ, Dζ]g
    # This involves computing D_ζ(g) and then f·D_ζ²(g) - D_ζ(f·D_ζg)
    D_zeta_g = D_zeta(g_series)
    logger.debug(f"\nStep 1: D_ζ(g) = {D_zeta_g}")
    
    # Add check for constant vector field
    if f.is_constant():
        return GradedDifferential(LogLaurentSeries(), g._j)
    
    D_zeta_D_zeta_g = D_zeta(D_zeta_g)
    logger.debug(f"Step 2: D_ζ²(g) = {D_zeta_D_zeta_g}")
    
    term1 = f * D_zeta_D_zeta_g
    logger.debug(f"Step 3: f·D_ζ²(g) = {term1}")
    
    f_D_zeta_g = f * D_zeta_g
    logger.debug(f"Step 4: f·D_ζ(g) = {f_D_zeta_g}")
    
    term2 = D_zeta(f_D_zeta_g)
    logger.debug(f"Step 5: D_ζ(f·D_ζ(g)) = {term2}")
    
    commutator = term2 - term1
    logger.debug(f"\nCommutator [fDζ, Dζ]g = {commutator}")
    
    # Compute the scaling term (j/2)∂f/∂z g
    df_dz = d_dz(f)
    scale_term = (df_dz * g_series) * (g._j / 2)
    logger.debug(f"\nScale term (j/2)·∂f/∂z·g = {scale_term}")
    
    # Combine terms
    result_series = commutator + scale_term
    logger.debug(f"\nFinal result = {result_series}")
    
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