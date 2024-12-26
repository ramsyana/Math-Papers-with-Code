"""
Test suite for Relative Sizes of Iterated Sumsets implementation
=============================================================

This module contains tests for the implementation of concepts from the paper
"Relative Sizes of Iterated Sumsets" by Noah Kravitz.

Tests cover:
- Alpha sequence generation
- ArithmeticSet operations
- Sumset computations
- Complete theorem verification
"""

import pytest
from iterated_sumsets import ArithmeticSet, generate_alpha_sequences

def test_generate_alpha_sequences():
    """Test alpha sequence generation with different parameters"""
    # Test case 1: Basic sequence generation
    sequences = generate_alpha_sequences(n=2, R=3)
    assert len(sequences) == 2  # Should generate 2 sequences
    assert all(0 in seq for seq in sequences)  # All sequences should contain 0
    
    # Test case 2: Single sequence
    sequences = generate_alpha_sequences(n=1, R=2)
    assert len(sequences) == 1
    assert sequences[0] == [0, 2, 3]  # Based on gamma(r) = r + 1

def test_arithmetic_set_creation():
    """Test ArithmeticSet initialization and basic properties"""
    # Test case 1: Simple set creation
    alphas = [0, 2, 3]
    M = 3
    arith_set = ArithmeticSet(M, alphas)
    assert arith_set.M == M
    assert arith_set.alphas == sorted(alphas)

def test_interval_at_scale():
    """Test interval generation at different scales"""
    arith_set = ArithmeticSet(M=3, alphas=[0, 1])
    
    # Test scale 0 (M^0)
    scale_0 = arith_set.interval_at_scale(0)
    assert scale_0 == {0, 1, 2, 3}
    
    # Test scale 1 (M^1)
    scale_1 = arith_set.interval_at_scale(1)
    assert scale_1 == {0, 3, 6, 9}

def test_h_fold_sumset():
    """Test h-fold sumset computation"""
    arith_set = ArithmeticSet(M=2, alphas=[0, 1])
    
    # Test 1-fold sumset (original set)
    sumset_1 = arith_set.h_fold_sumset(1)
    elements = arith_set.get_elements()
    assert sumset_1 == elements
    
    # Test 2-fold sumset
    sumset_2 = arith_set.h_fold_sumset(2)
    assert len(sumset_2) >= len(sumset_1)  # 2-fold should be at least as large

def test_nathanson_property():
    """Test the main theorem property with specific h values"""
    # Generate sequences and create sets
    alpha_sequences = generate_alpha_sequences(n=2, R=3)
    set1 = ArithmeticSet(M=3, alphas=alpha_sequences[0])
    set2 = ArithmeticSet(M=3, alphas=alpha_sequences[1])
    
    # Test sizes for h = [1, 2, 3]
    h_values = [1, 2, 3]
    sizes1 = [len(set1.h_fold_sumset(h)) for h in h_values]
    sizes2 = [len(set2.h_fold_sumset(h)) for h in h_values]
    
    # Verify the alternating size relationship
    assert sizes1[0] < sizes2[0]  # |h₁A| < |h₁B|
    assert sizes2[1] < sizes1[1]  # |h₂B| < |h₂A|
    assert sizes1[2] < sizes2[2]  # |h₃A| < |h₃B|

def test_edge_cases():
    """Test edge cases and potential error conditions"""
    # Test empty alphas list
    with pytest.raises(ValueError):
        ArithmeticSet(M=3, alphas=[])
    
    # Test negative M
    with pytest.raises(ValueError):
        ArithmeticSet(M=-1, alphas=[0, 1])
    
    # Test negative h value
    arith_set = ArithmeticSet(M=2, alphas=[0, 1])
    with pytest.raises(ValueError):
        arith_set.h_fold_sumset(-1)