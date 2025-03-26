# Math Papers with Code

A collection of implementations of mathematical algorithms and concepts from various academic papers in multiple programming languages.

## Implemented Papers

### Current Implementations

| Paper Title | Author(s) | arXiv | Implementations | Status | Directory |
|------------|-----------|--------|-----------------|---------|-----------|
| Relative Sizes of Iterated Sumsets | Noah Kravitz | [2412.18598](https://arxiv.org/pdf/2412.18598) | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | ✅ Complete | `papers/iterated-sumsets/` |
| A Remark on an Explicit Formula for the Sums of Powers of Integers | José L. Cereceda | [2503.14508v2](https://arxiv.org/pdf/2503.14508v2) | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | ✅ Complete | `papers/stirling-numbers-power-sums/` |
| The Neveu-Schwarz Group and Schwarz's Extended Super Mumford Form | Katherine A. Maxwell & Alexander A. Voronov | [2412.18585](https://arxiv.org/pdf/2412.18585) | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | ⏸️ On Hold | `papers/super-mumford/` |
| Derivative Polynomials and Infinite Series for Squigonometric Functions | Bart S. Van Lith | [2503.19624](https://arxiv.org/abs/2503.19624) | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | ✅ Complete | `papers/squigonometry/` |

### Implementation Status Legend

| Status | Description |
|--------|-------------|
| ✅ Complete | Implementation finished and tested |
| 🚧 In Progress | Currently being implemented |
| 📝 Planned | On roadmap for implementation |
| ⏸️ On Hold | Implementation paused |

### Coming Soon

Future papers will be added to this collection. Suggestions for new implementations are welcome through issues or pull requests.

## Repository Structure

Each paper implementation is organized in its own directory with its implementation:

```
.
├── README.md
├── papers/
│   ├── iterated-sumsets/
│   │   ├── README.md
│   │   └── python/
│   │       ├── iterated_sumsets.py
│   │       └── tests/
│   ├── super-mumford/
│   │   ├── README.md
│   │   └── python/
│   │       ├── core/
│   │       │   ├── __init__.py
│   │       │   ├── laurent_series.py
│   │       │   ├── matrix_ops.py
│   │       │   └── vector_spaces.py
│   │       ├── geometry/
│   │       │   ├── __init__.py
│   │       │   ├── grassmannian.py
│   │       │   └── line_bundles.py
│   │       ├── groups/
│   │       │   ├── __init__.py
│   │       │   ├── heisenberg.py
│   │       │   ├── neveu_schwarz.py
│   │       │   └── witt.py
│   │       ├── tests/
│   │       │   ├── __init__.py
│   │       │   ├── test_laurent_series.py
│   │       │   ├── test_matrix_ops.py
│   │       │   └── test_vector_spaces.py
│   │       ├── utils/
│   │       │   ├── __init__.py
│   │       │   └── validation.py
│   │       ├── README.md
│   │       └── pyproject.toml
│   └── future-papers/
│       ├── README.md
│       └── python/
└── common/
    ├── testing/
    └── benchmarks/
```

## Using the Implementations

Each paper implementation includes its own README with specific instructions. For Python implementations:

```bash
# Example for Super Mumford project
cd papers/super-mumford/python
pip install -r requirements.txt
python -m pytest tests/
```

## Contributing

Contributions are welcome! To contribute:

1. Select a mathematics paper to implement
2. Create a new directory under `papers/`
3. Implement the paper's concepts
4. Include:
   - README.md with paper details
   - Source code
   - Tests (if applicable)
   - Docker support (if applicable)
   - Documentation (if applicable)
   - Performance benchmarks (optional)

Please see CONTRIBUTING.md for detailed guidelines.

## Paper Implementation Guidelines

Each paper implementation should:

1. **Documentation**
   - Include link to original paper
   - Explain key concepts
   - Provide usage examples
   - Document any assumptions or limitations

2. **Code Structure**
   - Clear organization
   - Well-commented code
   - Tests
   - Docker support (where applicable)

3. **Performance**
   - Efficient implementations
   - Benchmarking (optional)
   - Optimization notes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- All original paper authors
- Contributors to the implementations
- Open source community
