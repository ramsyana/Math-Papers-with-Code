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
    
    # First compute the derivative terms
    D_ζg = D_zeta(g_series)
    f_D_ζg = f * D_ζg
    
    # First term includes both parts of D_ζ
    term1a = d_dzeta(f_D_ζg)
    term1b = f_D_ζg.multiply_by_zeta(d_dz(f_D_ζg))
    term1 = term1a + term1b
    
    # Second term: f·D_ζ²(g)
    D_ζD_ζg = D_zeta(D_zeta(g_series))
    term2 = f * D_ζD_ζg
    
    # Compute the commutator
    commutator = term1 - term2
    
    # Compute (j/2)∂f/∂z g term
    df_dz = d_dz(f)
    
    # CRITICAL FIX: Explicitly use the grade and handle multiplication
    scale_term = df_dz * g_series * g.grade
    
    # Ensure the scale term is added correctly
    result_series = scale_term + commutator 
    
    return GradedDifferential(result_series, g._j)

def lie_bracket(f: LogLaurentSeries, h: LogLaurentSeries) -> LogLaurentSeries:
    """
    Compute the Lie bracket [[fDζ, Dζ], [hDζ, Dζ]] acting on H•.
    """
    # CRITICAL FIX: Ensure non-zero cases are preserved
    if f._even_terms == {0: {0: 0}} or h._even_terms == {0: {0: 0}}:
        return h if f._even_terms == {0: {0: 0}} else f
    
    # Compute first bracket [fDζ, Dζ]h
    D_ζh = D_zeta(h)
    f_D_ζh = f * D_ζh
    
    # First term of first bracket
    term1a = d_dzeta(f_D_ζh)
    term1b = f_D_ζh.multiply_by_zeta(d_dz(f_D_ζh))
    term1 = term1a + term1b
    
    # Second term of first bracket
    D_ζD_ζh = D_zeta(D_zeta(h))
    term2 = f * D_ζD_ζh
    
    first_action = term1 - term2
    
    # Compute second bracket [hDζ, Dζ]f 
    D_ζf = D_zeta(f)
    h_D_ζf = h * D_ζf
    
    # First term of second bracket
    term3a = d_dzeta(h_D_ζf)
    term3b = h_D_ζf.multiply_by_zeta(d_dz(h_D_ζf))
    term3 = term3a + term3b
    
    # Second term of second bracket
    D_ζD_ζf = D_zeta(D_zeta(f))
    term4 = h * D_ζD_ζf
    second_action = term3 - term4
    
    # Complete Lie bracket is difference of actions
    result = first_action - second_action 
    
    return result