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
   """Compute [[fDζ, Dζ], [hDζ, Dζ]]"""
   
   # First compute super derivatives
   D_zeta_h = D_zeta(h)
   D_zeta_f = D_zeta(f)
   
   # [fDζ, Dζ]h
   bracket1 = f * D_zeta(D_zeta_h) - D_zeta(f * D_zeta_h)
   
   # [hDζ, Dζ]f  
   bracket2 = h * D_zeta(D_zeta_f) - D_zeta(h * D_zeta_f)
   
   # Return [[fDζ, Dζ], [hDζ, Dζ]]
   return bracket1 - bracket2