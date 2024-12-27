import pytest
from super_mumford.core.log_laurent_series import LogLaurentSeries

def test_log_laurent_series_creation():
    # Test empty series
    empty = LogLaurentSeries()
    assert str(empty) == "0"
    
    # Test even series with log terms
    even = LogLaurentSeries(log_terms={
        0: {0: 1, 1: 2},  # 1 + 2z
        1: {0: 3},        # 3log(z)
        2: {1: 4}         # 4z*log²(z)
    })
    assert "1" in str(even)
    assert "2z" in str(even)
    assert "3log(z)" in str(even)
    assert "4z*log^2(z)" in str(even)
    
    # Test odd series with log terms
    odd = LogLaurentSeries(odd_log_terms={
        0: {0: 1},    # ζ
        1: {1: 2}     # 2ζz*log(z)
    })
    assert "ζ" in str(odd)
    assert "2ζ*z*log(z)" in str(odd)

def test_max_log_power():
    series = LogLaurentSeries(
        log_terms={0: {1: 1}, 2: {1: 1}},
        odd_log_terms={1: {1: 1}, 3: {1: 1}}
    )
    assert series.max_log_power == 3

def test_log_laurent_series_addition():
    s1 = LogLaurentSeries(
        log_terms={
            0: {0: 1},    # 1
            1: {0: 2}     # 2log(z)
        },
        odd_log_terms={
            0: {0: 3}     # 3ζ
        }
    )
    
    s2 = LogLaurentSeries(
        log_terms={
            1: {0: 1}     # log(z)
        },
        odd_log_terms={
            0: {0: -3}    # -3ζ
        }
    )
    
    result = s1 + s2
    assert result._even_terms[0][0] == 1    # Constant term
    assert result._even_terms[1][0] == 3    # log(z) term
    assert result._odd_terms[0][0] == 0     # ζ terms cancel

def test_log_laurent_series_multiplication():
    # Test even * even with log terms
    s1 = LogLaurentSeries(log_terms={
        0: {0: 2},    # 2
        1: {0: 3}     # 3log(z)
    })
    s2 = LogLaurentSeries(log_terms={
        1: {0: 1}     # log(z)
    })
    
    result = s1 * s2
    # Should get: 2log(z) + 3log²(z)
    assert result._even_terms[1][0] == 2    # 2 * 1 * log(z)
    assert result._even_terms[2][0] == 3    # 3 * 1 * log(z) * log(z)