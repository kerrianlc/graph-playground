"""
Birkhoff-von Neumann Algorithm Implementation.

This module implements the Birkhoff-von Neumann decomposition algorithm,
which decomposes a doubly stochastic matrix into a convex combination
of permutation matrices.

A doubly stochastic matrix is a square matrix where each row and column
sums to 1, and all entries are non-negative.

The Birkhoff-von Neumann theorem states that any doubly stochastic matrix
can be expressed as a convex combination of permutation matrices.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def is_doubly_stochastic(matrix, tol=1e-9):
    """
    Check if a matrix is doubly stochastic.

    A doubly stochastic matrix is a square matrix where:
    - All entries are non-negative
    - Each row sums to 1
    - Each column sums to 1

    Args:
        matrix: A 2D numpy array.
        tol: Tolerance for floating point comparisons.

    Returns:
        bool: True if the matrix is doubly stochastic, False otherwise.
    """
    matrix = np.asarray(matrix, dtype=float)

    # Check if matrix is square
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False

    # Check if all entries are non-negative
    if np.any(matrix < -tol):
        return False

    # Check if rows sum to 1
    row_sums = np.sum(matrix, axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        return False

    # Check if columns sum to 1
    col_sums = np.sum(matrix, axis=0)
    if not np.allclose(col_sums, 1.0, atol=tol):
        return False

    return True


def birkhoff_decomposition(matrix, tol=1e-9, max_iterations=None):
    """
    Decompose a doubly stochastic matrix into a convex combination of permutation matrices.

    Uses the Birkhoff-von Neumann algorithm to express the input matrix as:
        matrix = sum_i (coeff_i * P_i)

    where P_i are permutation matrices and coeff_i are positive coefficients
    that sum to 1.

    Args:
        matrix: A doubly stochastic matrix (2D numpy array or list of lists).
        tol: Tolerance for floating point comparisons and determining when
             entries are effectively zero.
        max_iterations: Maximum number of iterations (default: n^2 where n is matrix size).

    Returns:
        A list of tuples (coefficient, permutation_matrix) where:
        - coefficient is a positive float
        - permutation_matrix is an n x n numpy array with exactly one 1 in each row and column

    Raises:
        ValueError: If the input is not a valid doubly stochastic matrix.

    Example:
        >>> matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        >>> decomposition = birkhoff_decomposition(matrix)
        >>> for coeff, perm in decomposition:
        ...     print(f"Coefficient: {coeff}")
        ...     print(perm)
    """
    matrix = np.asarray(matrix, dtype=float)

    if not is_doubly_stochastic(matrix, tol):
        raise ValueError("Input matrix must be doubly stochastic")

    n = matrix.shape[0]
    if max_iterations is None:
        max_iterations = n * n + 10  # n^2 should be enough, add buffer

    # Make a copy to avoid modifying the original
    remaining = matrix.copy()

    # Clip small negative values that might arise from floating point errors
    remaining = np.maximum(remaining, 0)

    decomposition = []

    for _ in range(max_iterations):
        # Check if we're done (matrix is essentially zero)
        if np.max(remaining) < tol:
            break

        # Find a perfect matching in the support of the remaining matrix
        # We use the Hungarian algorithm on the negative of the remaining matrix
        # to find a maximum weight matching in the support
        support = (remaining > tol).astype(float)

        # Use linear_sum_assignment to find a perfect matching
        # We maximize the matching weight by minimizing the negative
        row_ind, col_ind = linear_sum_assignment(-support)

        # Create the permutation matrix from the matching
        permutation = np.zeros((n, n), dtype=float)
        permutation[row_ind, col_ind] = 1.0

        # Find the minimum positive coefficient in this matching
        # This is the weight we can subtract
        matching_values = remaining[row_ind, col_ind]
        coeff = np.min(matching_values)

        if coeff < tol:
            # This shouldn't happen for a valid doubly stochastic matrix
            # but we handle it gracefully
            break

        # Record this permutation and coefficient
        decomposition.append((coeff, permutation))

        # Subtract this weighted permutation from the remaining matrix
        remaining -= coeff * permutation

        # Clip small values to zero to avoid numerical issues
        remaining = np.maximum(remaining, 0)

    return decomposition


def reconstruct_from_decomposition(decomposition):
    """
    Reconstruct a matrix from its Birkhoff decomposition.

    Args:
        decomposition: A list of (coefficient, permutation_matrix) tuples
                       as returned by birkhoff_decomposition.

    Returns:
        The reconstructed matrix as a numpy array.

    Example:
        >>> matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        >>> decomp = birkhoff_decomposition(matrix)
        >>> reconstructed = reconstruct_from_decomposition(decomp)
        >>> np.allclose(matrix, reconstructed)
        True
    """
    if not decomposition:
        raise ValueError("Decomposition cannot be empty")

    n = decomposition[0][1].shape[0]
    result = np.zeros((n, n), dtype=float)

    for coeff, perm in decomposition:
        result += coeff * perm

    return result


def verify_decomposition(original, decomposition, tol=1e-9):
    """
    Verify that a decomposition correctly represents the original matrix.

    Checks that:
    1. All coefficients are positive
    2. All matrices are valid permutation matrices
    3. Coefficients sum to approximately 1
    4. The weighted sum equals the original matrix

    Args:
        original: The original doubly stochastic matrix.
        decomposition: A list of (coefficient, permutation_matrix) tuples.
        tol: Tolerance for floating point comparisons.

    Returns:
        bool: True if the decomposition is valid, False otherwise.
    """
    original = np.asarray(original, dtype=float)

    if not decomposition:
        return False

    n = original.shape[0]
    total_coeff = 0.0
    reconstructed = np.zeros_like(original)

    for coeff, perm in decomposition:
        # Check coefficient is positive
        if coeff < -tol:
            return False

        total_coeff += coeff

        # Check permutation matrix shape
        if perm.shape != (n, n):
            return False

        # Check it's a valid permutation matrix
        # Each row and column should sum to 1, with only 0s and 1s
        if not np.allclose(np.sum(perm, axis=0), 1.0, atol=tol):
            return False
        if not np.allclose(np.sum(perm, axis=1), 1.0, atol=tol):
            return False
        if not np.all((perm >= -tol) & (perm <= 1 + tol)):
            return False

        reconstructed += coeff * perm

    # Check coefficients sum to 1
    if not np.isclose(total_coeff, 1.0, atol=tol):
        return False

    # Check reconstruction matches original
    if not np.allclose(original, reconstructed, atol=tol):
        return False

    return True


if __name__ == "__main__":
    # Example usage
    print("Birkhoff-von Neumann Decomposition Example")
    print("=" * 50)

    # Create a simple doubly stochastic matrix
    matrix = np.array([
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])

    print("\nOriginal doubly stochastic matrix:")
    print(matrix)

    print("\nIs doubly stochastic:", is_doubly_stochastic(matrix))

    print("\nDecomposition into permutation matrices:")
    decomposition = birkhoff_decomposition(matrix)

    for i, (coeff, perm) in enumerate(decomposition):
        print(f"\nPermutation {i + 1} (coefficient = {coeff:.4f}):")
        print(perm.astype(int))

    # Verify the decomposition
    print("\n" + "=" * 50)
    reconstructed = reconstruct_from_decomposition(decomposition)
    print("\nReconstructed matrix:")
    print(reconstructed)

    print("\nReconstruction error:", np.max(np.abs(matrix - reconstructed)))
    print("Decomposition is valid:", verify_decomposition(matrix, decomposition))
