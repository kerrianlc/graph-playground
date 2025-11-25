# graph-playground

A collection of graph theory and matrix algorithms.

## Birkhoff-von Neumann Algorithm

This repository includes an implementation of the Birkhoff-von Neumann decomposition algorithm, which decomposes a doubly stochastic matrix into a convex combination of permutation matrices.

### What is a Doubly Stochastic Matrix?

A doubly stochastic matrix is a square matrix where:
- All entries are non-negative
- Each row sums to 1
- Each column sums to 1

### The Birkhoff-von Neumann Theorem

The theorem states that any doubly stochastic matrix can be expressed as a convex combination of permutation matrices:

```
M = c₁P₁ + c₂P₂ + ... + cₖPₖ
```

where P₁, P₂, ..., Pₖ are permutation matrices and c₁ + c₂ + ... + cₖ = 1.

### Installation

```bash
pip install numpy scipy
```

### Usage

```python
import numpy as np
from birkhoff import birkhoff_decomposition, verify_decomposition

# Create a doubly stochastic matrix
matrix = np.array([
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5]
])

# Decompose into permutation matrices
decomposition = birkhoff_decomposition(matrix)

# Print the decomposition
for coeff, perm in decomposition:
    print(f"Coefficient: {coeff}")
    print(perm)

# Verify the decomposition
is_valid = verify_decomposition(matrix, decomposition)
print(f"Valid decomposition: {is_valid}")
```

### API Reference

#### `is_doubly_stochastic(matrix, tol=1e-9)`
Check if a matrix is doubly stochastic.

#### `birkhoff_decomposition(matrix, tol=1e-9, max_iterations=None)`
Decompose a doubly stochastic matrix into a convex combination of permutation matrices.

Returns a list of `(coefficient, permutation_matrix)` tuples.

#### `reconstruct_from_decomposition(decomposition)`
Reconstruct a matrix from its Birkhoff decomposition.

#### `verify_decomposition(original, decomposition, tol=1e-9)`
Verify that a decomposition correctly represents the original matrix.

### Running Tests

```bash
pip install pytest
pytest test_birkhoff.py -v
```

## License

MIT License