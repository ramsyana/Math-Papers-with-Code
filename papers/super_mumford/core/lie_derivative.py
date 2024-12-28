from typing import Optional
from .log_laurent_series import LogLaurentSeries
from .log_laurent_derivatives import d_dz, d_dzeta, D_zeta
from .graded_differential import GradedDifferential

def lie_derivative(f: LogLaurentSeries, g: GradedDifferential) -> GradedDifferential:
    """
    Compute the Lie derivative of g[dz|dζ]⊗j with respect to [fDζ, Dζ].
    The formula is:
    ρ([fDζ, Dζ])g[dz|dζ]⊗j = ([fDζ, Dζ]g + (j/2)∂f/∂z g)[dz|dζ]⊗j
    """
    g_series = g._series
    
    # Compute [fDζ, Dζ]g = D_ζ(f·D_ζg) - f·D_ζ²(g)
    f_D_ζg = f * D_zeta(g_series)
    term1 = D_zeta(f_D_ζg)
    
    D_ζD_ζg = D_zeta(D_zeta(g_series))
    term2 = f * D_ζD_ζg
    
    commutator = term1 - term2
    
    # Compute (j/2)∂f/∂z g 
    df_dz = d_dz(f)
    scale_term = df_dz * g_series * g.grade
    
    # Add the terms correctly
    result_series = commutator + scale_term
    
    return GradedDifferential(result_series, g._j)

def lie_bracket(f: LogLaurentSeries, h: LogLaurentSeries) -> LogLaurentSeries:
    """
    Compute the Lie bracket [[fDζ, Dζ], [hDζ, Dζ]] acting on H•.
    """
    # Compute [fDζ, Dζ]h = D_ζ(f·D_ζh) - f·D_ζ²(h)
    f_D_ζh = f * D_zeta(h)
    term1 = D_zeta(f_D_ζh)
    
    D_ζD_ζh = D_zeta(D_zeta(h))  
    term2 = f * D_ζD_ζh
    
    first_action = term1 - term2
    
    # Compute [hDζ, Dζ]f similarly
    h_D_ζf = h * D_zeta(f)
    term3 = D_zeta(h_D_ζf)
    
    D_ζD_ζf = D_zeta(D_zeta(f))
    term4 = h * D_ζD_ζf
    
    second_action = term3 - term4
    
    # The bracket is the difference of these actions
    result = first_action - second_action
    
    return result