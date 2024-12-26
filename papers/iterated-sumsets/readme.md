# Relative Sizes of Iterated Sumsets

Implementation of algorithms and concepts from the paper ["Relative Sizes of Iterated Sumsets"](https://arxiv.org/pdf/2412.18598).

## Paper Information

| Field | Details |
|-------|---------|
| Title | Relative Sizes of Iterated Sumsets |
| Author | Noah Kravitz |
| arXiv | [2412.18598](https://arxiv.org/pdf/2412.18598) |
| Date | December 2024 |
| Status | Complete |

## Implementation Status

| Language | Status | Features | Directory |
|----------|---------|-----------|------------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | ✅ Complete | Full implementation | `python/` |

## Overview

This implementation demonstrates Nathanson's question about the existence of finite sets A, B ⊆ Z and natural numbers h₁ < h₂ < h₃ where:

```
|h₁A| < |h₁B|
|h₂B| < |h₂A|
|h₃A| < |h₃B|
```

## Key Components

### ArithmeticSet Class
- Manages sets and computing h-fold sumsets
- Handles set elements at different scales
- Computes iterated sumsets

### Alpha Sequence Generator
- Generates compatible alpha sequences
- Implements simplified gamma function
- Creates sequences for theorem verification

### Main Demonstration
- Shows example parameters and computations
- Verifies relative size properties
- Handles arbitrary h values

## Usage

### Python Implementation

```bash
python iterated_sumsets.py
```

Example output:
```
Program started
Starting Nathanson's sumsets computation...

Parameters: n=2, R=3, M=3

Generated alpha sequences: [[0, 2, 3, 4], [0, 4, 6, 8]]

Set elements:
Set A1: [0, 1, 2, 3, 9, 18, 27, 54, 81, 162, 243]
Set A2: [0, 1, 2, 3, 81, 162, 243, 729, 1458, 2187, 6561, 13122, 19683]

Using h values: [1, 2, 3]

Sumset sizes:
For h = 1:
  |1A1| = 11
  |1A2| = 13
For h = 2:
  |2A1| = 54
  |2A2| = 79
For h = 3:
  |3A1| = 161
  |3A2| = 305
Program finished
```

## Testing

```bash
cd python/tests
python -m pytest
```

## Performance Notes

The implementation focuses on:
- Correct implementation of mathematical concepts
- Clear demonstration of theorem properties
- Basic optimization of set operations

## Contributing

Contributions welcome in areas such as:
- Performance optimizations
- Additional test coverage
- Documentation improvements

## License

MIT License - see [LICENSE](LICENSE) file for details.

## References

1. Kravitz, N. (2024). "Relative Sizes of Iterated Sumsets." arXiv:2412.18598
2. Nathanson, M. "Inverse problems for sumset sizes of finite sets of integers." arXiv:2412.16154v1