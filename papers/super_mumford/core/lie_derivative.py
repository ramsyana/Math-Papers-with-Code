from typing import Optional
import logging
from .log_laurent_series import LogLaurentSeries
from .log_laurent_derivatives import d_dz, d_dzeta, D_zeta
from .graded_differential import GradedDifferential

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def lie_derivative(f: LogLaurentSeries, g: GradedDifferential) -> GradedDifferential:
    """
    Compute Lie derivative of g[dz|dζ]⊗j with respect to [fDζ, Dζ].
    Formula: ρ([fDζ, Dζ])g[dz|dζ]⊗j = ([fDζ, Dζ]g + (j/2)∂f/∂z g)[dz|dζ]⊗j
    """
    logger.debug("\n==== Starting lie_derivative computation ====")
    logger.debug(f"Input f: {f}")
    logger.debug(f"Input g: {g._series}")
    logger.debug(f"Grade j/2: {g.grade}")

    g_series = g._series
    
    # First compute [fDζ, Dζ]g:
    # Log all intermediate steps in commutator calculation
    D_zeta_g = D_zeta(g_series)
    logger.debug(f"\nStep 1: D_ζ(g) = {D_zeta_g}")
    
    D_zeta_D_zeta_g = D_zeta(D_zeta_g)
    logger.debug(f"Step 2: D_ζ²(g) = {D_zeta_D_zeta_g}")
    
    term1 = f * D_zeta_D_zeta_g
    logger.debug(f"Step 3: f·D_ζ²(g) = {term1}")
    
    # Second term
    f_D_zeta_g = f * D_zeta_g
    logger.debug(f"\nStep 4: f·D_ζ(g) = {f_D_zeta_g}")
    
    term2 = D_zeta(f_D_zeta_g)
    logger.debug(f"Step 5: D_ζ(f·D_ζ(g)) = {term2}")
    
    commutator = term1 - term2
    logger.debug(f"\nCommutator [fDζ, Dζ]g = {commutator}")
    logger.debug("Commutator coefficients:")
    for k in commutator._even_terms:
        for p, c in commutator._even_terms[k].items():
            logger.debug(f"  z^{p}: {c}")
    
    # Scale term calculation
    df_dz = d_dz(f)
    logger.debug(f"\nDEBUG: Scale Term Components:")
    logger.debug(f"df_dz terms: {df_dz._even_terms}")
    logger.debug(f"g_series terms: {g_series._even_terms}")
    logger.debug(f"g.grade: {g.grade}")
    
    scale_term = df_dz * g_series * g._j  # This is right, but need to fix d_dz
    logger.debug(f"Scale term before final multiply: {df_dz * g_series}")
    logger.debug(f"Step 7: Scale term (j/2)·∂f/∂z·g = {scale_term}")
    logger.debug("Scale term coefficients:")
    for k in scale_term._even_terms:
        for p, c in scale_term._even_terms[k].items():
            logger.debug(f"  z^{p}: {c}")
    
    # Combine terms
    result_series = commutator + scale_term
    logger.debug(f"\nFinal result = {result_series}")
    logger.debug("Final coefficients:")
    for k in result_series._even_terms:
        for p, c in result_series._even_terms[k].items():
            logger.debug(f"  z^{p}: {c}")
    
    return GradedDifferential(result_series, g._j)

def is_zero_series(series: LogLaurentSeries) -> bool:
    """Check if a series is effectively zero"""
    # Check even terms
    if any(abs(coeff) > 1e-15 
           for terms in series._even_terms.values()
           for coeff in terms.values()):
        return False
        
    # Check odd terms 
    if any(abs(coeff) > 1e-15
           for terms in series._odd_terms.values()
           for coeff in terms.values()):
        return False
            
    return True

def lie_bracket(f: LogLaurentSeries, h: LogLaurentSeries) -> LogLaurentSeries:
    """
    Compute [[fDζ, Dζ], [hDζ, Dζ]] acting on H•.
    """
    logger.info("\n==== Starting lie_bracket computation ====")
    logger.info(f"Input vectors: f = {f}, h = {h}")
    
    if is_zero_series(f):
        logger.info("f is zero, returning h")
        return h
    if is_zero_series(h):
        logger.info("h is zero, returning f")
        return f
    
    # First bracket computation
    logger.info("\nComputing first bracket [fDζ, Dζ]h:")
    D_zeta_h = D_zeta(h)
    D_zeta_f = D_zeta(f)
    logger.info(f"D_ζ(h) = {D_zeta_h}")
    
    D_zeta_D_zeta_h = D_zeta(D_zeta_h)
    logger.info(f"D_ζ²(h) = {D_zeta_D_zeta_h}")
    
    term1 = f * D_zeta_D_zeta_h
    logger.info(f"f·D_ζ²(h) = {term1}")
    
    term2 = D_zeta(f * D_zeta_h)
    logger.info(f"D_ζ(f·D_ζ(h)) = {term2}")
    
    term3 = D_zeta_h * 2 * D_zeta_f
    logger.info(f"2D_ζ(h)·D_ζ(f) = {term3}")
    
    first_action = term1 - term2 + term3
    logger.info(f"First bracket complete = {first_action}")
    
    # Second bracket computation  
    logger.info("\nComputing second bracket [hDζ, Dζ]f:")
    D_zeta_D_zeta_f = D_zeta(D_zeta_f)
    logger.info(f"D_ζ²(f) = {D_zeta_D_zeta_f}")
    
    term4 = h * D_zeta_D_zeta_f
    term5 = D_zeta(h * D_zeta_f) 
    term6 = D_zeta_f * 2 * D_zeta_h
    second_action = term4 - term5 - term6
    logger.info(f"Second bracket complete = {second_action}")
    
    # Complete bracket
    result = first_action - second_action
    logger.info(f"\nFinal result = {result}")
    logger.info("Final coefficients:")
    for k in result._even_terms:
        for p, c in result._even_terms[k].items():
            logger.info(f"z^{p}: {c}")
    
    return result

class LogLaurentSeries:
    def __str__(self):
        """String representation matching expected test format"""
        return ' + '.join(f"{k}: {dict(v)}" for k, v in self._even_terms.items()) or "0"