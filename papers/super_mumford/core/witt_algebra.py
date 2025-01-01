from typing import Optional
import logging
from .log_laurent_series import LogLaurentSeries
from .log_laurent_derivatives import d_dz, d_dzeta, D_zeta
from .graded_differential import GradedDifferential

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def witt_action(f: LogLaurentSeries, g: GradedDifferential) -> GradedDifferential:
    if f.is_constant() or g._series.is_zero():
        return GradedDifferential(LogLaurentSeries(), g._j)

    D_zeta_g = D_zeta(g._series)
    
    # Compute [fDζ, Dζ]g
    term1 = f * D_zeta(D_zeta_g) 
    term2 = D_zeta(f * D_zeta_g)
    commutator = term1 - term2
    
    # Scale term should use j/2 not j
    df_dz = d_dz(f)
    scale_term = df_dz * g._series * (g._j/2)  # Changed from g._j to g._j/2
    
    return GradedDifferential(commutator + scale_term, g._j)

def witt_bracket(f: LogLaurentSeries, h: LogLaurentSeries) -> LogLaurentSeries:
    """Compute brackets for super Witt algebra"""
    f_is_odd = bool(f._odd_terms)
    h_is_odd = bool(h._odd_terms)

    if not f_is_odd and not h_is_odd:
        # [Lp, Lq] = (p-q)Lp+q 
        p = next(iter(f._even_terms[0].keys()))
        q = next(iter(h._even_terms[0].keys()))
        return LogLaurentSeries(log_terms={0: {p + q: p - q}})

    elif not f_is_odd and h_is_odd:
        # [Lp, Gr] = (p/2 - r)Gp+r
        p = next(iter(f._even_terms[0].keys())) 
        r = next(iter(h._odd_terms[0].keys()))
        coeff = p/2 - r  
        return LogLaurentSeries(odd_log_terms={0: {p + r: coeff}})

    elif f_is_odd and h_is_odd:
        # [Gr, Gs] = 2Lr+s
        r = next(iter(f._odd_terms[0].keys()))
        s = next(iter(h._odd_terms[0].keys()))  
        return LogLaurentSeries(log_terms={0: {r + s: 2}})

    else:
        # [Gr, Lp] = -(p/2 - r)Gr+p
        r = next(iter(f._odd_terms[0].keys()))
        p = next(iter(h._even_terms[0].keys()))
        coeff = -(p/2 - r)
        return LogLaurentSeries(odd_log_terms={0: {r + p: coeff}})
