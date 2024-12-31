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
    Compute the action of a Witt algebra element [fDζ, Dζ] on a graded differential g.
    """
    logger.info("Starting Witt action computation")
    logger.debug("[Input] f (vector field): %s", f)
    logger.debug("[Input] g (differential): %s | Grade: %s", g._series, g._j)

    if f.is_constant() or g._series.is_zero():
        logger.info("[Shortcut] Returning zero differential: Constant field or zero series")
        return GradedDifferential(LogLaurentSeries(), g._j)

    logger.info("[Step] Computing D_ζ(g)")
    D_zeta_g = D_zeta(g._series)
    logger.debug("[Result] D_ζ(g): %s", D_zeta_g)

    logger.info("[Step] Computing terms of the commutator")
    term1 = f * D_zeta(D_zeta_g)
    term2 = D_zeta(f) * D_zeta_g + f * D_zeta(D_zeta_g)
    commutator = term1 - term2
    logger.debug("[Intermediate] Term 1: %s", term1)
    logger.debug("[Intermediate] Term 2: %s", term2)
    logger.debug("[Result] Commutator: %s", commutator)

    logger.info("[Step] Computing scale term")
    df_dz = d_dz(f)
    scale_term = (df_dz * g._series) * g._j
    logger.debug("[Intermediate] df/dz: %s", df_dz)
    logger.debug("[Result] Scale term: %s", scale_term)

    logger.info("[Step] Combining terms into result series")
    result_series = commutator + scale_term
    logger.debug("[Result] Combined series: %s", result_series)

    logger.info("[Summary] Computed Witt action result")
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