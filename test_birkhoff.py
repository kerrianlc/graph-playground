"""
Unit tests for the Birkhoff-von Neumann decomposition algorithm.
"""

import numpy as np
import pytest
from birkhoff import (
    is_doubly_stochastic,
    birkhoff_decomposition,
    reconstruct_from_decomposition,
    verify_decomposition,
)


class TestIsDoublyStochastic:
    """Tests for the is_doubly_stochastic function."""

    def test_identity_matrix(self):
        """Identity matrix is doubly stochastic."""
        matrix = np.eye(3)
        assert is_doubly_stochastic(matrix)

    def test_uniform_matrix(self):
        """Uniform n x n matrix with entries 1/n is doubly stochastic."""
        n = 4
        matrix = np.ones((n, n)) / n
        assert is_doubly_stochastic(matrix)

    def test_2x2_doubly_stochastic(self):
        """2x2 doubly stochastic matrix."""
        matrix = np.array([[0.3, 0.7], [0.7, 0.3]])
        assert is_doubly_stochastic(matrix)

    def test_non_square_matrix(self):
        """Non-square matrix is not doubly stochastic."""
        matrix = np.ones((2, 3)) / 2
        assert not is_doubly_stochastic(matrix)

    def test_negative_entries(self):
        """Matrix with negative entries is not doubly stochastic."""
        matrix = np.array([[1.5, -0.5], [-0.5, 1.5]])
        assert not is_doubly_stochastic(matrix)

    def test_rows_not_summing_to_one(self):
        """Matrix with rows not summing to 1."""
        matrix = np.array([[0.5, 0.5], [0.3, 0.3]])
        assert not is_doubly_stochastic(matrix)

    def test_columns_not_summing_to_one(self):
        """Matrix with columns not summing to 1."""
        matrix = np.array([[0.5, 0.3], [0.5, 0.3]])
        assert not is_doubly_stochastic(matrix)

    def test_permutation_matrix(self):
        """Permutation matrix is doubly stochastic."""
        matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        assert is_doubly_stochastic(matrix)


class TestBirkhoffDecomposition:
    """Tests for the birkhoff_decomposition function."""

    def test_identity_matrix(self):
        """Identity matrix decomposes to itself."""
        matrix = np.eye(3)
        decomposition = birkhoff_decomposition(matrix)

        assert len(decomposition) == 1
        coeff, perm = decomposition[0]
        assert np.isclose(coeff, 1.0)
        assert np.allclose(perm, matrix)

    def test_permutation_matrix(self):
        """Permutation matrix decomposes to itself."""
        matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        decomposition = birkhoff_decomposition(matrix)

        assert len(decomposition) == 1
        coeff, perm = decomposition[0]
        assert np.isclose(coeff, 1.0)
        assert np.allclose(perm, matrix)

    def test_uniform_2x2(self):
        """Uniform 2x2 matrix decomposes into 2 permutations."""
        matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        decomposition = birkhoff_decomposition(matrix)

        # Should decompose into identity and swap permutation
        assert len(decomposition) == 2

        # Verify reconstruction
        assert verify_decomposition(matrix, decomposition)

    def test_3x3_matrix(self):
        """3x3 doubly stochastic matrix decomposition."""
        matrix = np.array([
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ])
        decomposition = birkhoff_decomposition(matrix)

        # Verify reconstruction
        assert verify_decomposition(matrix, decomposition)

        # Coefficients should sum to 1
        total_coeff = sum(coeff for coeff, _ in decomposition)
        assert np.isclose(total_coeff, 1.0)

    def test_uniform_nxn(self):
        """Uniform n x n matrix with entries 1/n."""
        n = 4
        matrix = np.ones((n, n)) / n
        decomposition = birkhoff_decomposition(matrix)

        # Verify reconstruction
        assert verify_decomposition(matrix, decomposition)

    def test_random_doubly_stochastic(self):
        """Random doubly stochastic matrix."""
        # Create a random doubly stochastic matrix using Sinkhorn iteration
        n = 5
        np.random.seed(42)
        A = np.random.rand(n, n)

        # Sinkhorn iteration to make it doubly stochastic
        for _ in range(100):
            A = A / A.sum(axis=1, keepdims=True)
            A = A / A.sum(axis=0, keepdims=True)

        assert is_doubly_stochastic(A)

        decomposition = birkhoff_decomposition(A)
        assert verify_decomposition(A, decomposition)

    def test_invalid_matrix_raises(self):
        """Non-doubly stochastic matrix raises ValueError."""
        matrix = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="doubly stochastic"):
            birkhoff_decomposition(matrix)

    def test_coefficients_positive(self):
        """All coefficients in decomposition are positive."""
        matrix = np.array([
            [0.25, 0.25, 0.5],
            [0.5, 0.25, 0.25],
            [0.25, 0.5, 0.25]
        ])
        decomposition = birkhoff_decomposition(matrix)

        for coeff, _ in decomposition:
            assert coeff > 0


class TestReconstructFromDecomposition:
    """Tests for the reconstruct_from_decomposition function."""

    def test_reconstruction_accuracy(self):
        """Reconstructed matrix matches original."""
        matrix = np.array([
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ])
        decomposition = birkhoff_decomposition(matrix)
        reconstructed = reconstruct_from_decomposition(decomposition)

        assert np.allclose(matrix, reconstructed)

    def test_empty_decomposition_raises(self):
        """Empty decomposition raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            reconstruct_from_decomposition([])


class TestVerifyDecomposition:
    """Tests for the verify_decomposition function."""

    def test_valid_decomposition(self):
        """Valid decomposition is verified correctly."""
        matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        decomposition = birkhoff_decomposition(matrix)

        assert verify_decomposition(matrix, decomposition)

    def test_wrong_coefficients(self):
        """Decomposition with wrong coefficients fails."""
        matrix = np.eye(2)
        # Create incorrect decomposition with wrong coefficient
        decomposition = [(0.5, np.eye(2))]

        assert not verify_decomposition(matrix, decomposition)

    def test_invalid_permutation(self):
        """Decomposition with invalid permutation matrix fails."""
        matrix = np.eye(2)
        # Create decomposition with non-permutation matrix
        decomposition = [(1.0, np.ones((2, 2)) / 2)]

        assert not verify_decomposition(matrix, decomposition)

    def test_empty_decomposition(self):
        """Empty decomposition fails verification."""
        matrix = np.eye(2)

        assert not verify_decomposition(matrix, [])


class TestEdgeCases:
    """Tests for edge cases."""

    def test_1x1_matrix(self):
        """1x1 doubly stochastic matrix (just [[1]])."""
        matrix = np.array([[1.0]])
        assert is_doubly_stochastic(matrix)

        decomposition = birkhoff_decomposition(matrix)
        assert len(decomposition) == 1
        assert verify_decomposition(matrix, decomposition)

    def test_list_input(self):
        """Function accepts list of lists as input."""
        matrix = [[0.5, 0.5], [0.5, 0.5]]
        decomposition = birkhoff_decomposition(matrix)

        assert verify_decomposition(matrix, decomposition)

    def test_large_matrix(self):
        """Larger matrix (10x10) decomposition."""
        n = 10
        np.random.seed(123)
        A = np.random.rand(n, n)

        # Sinkhorn iteration
        for _ in range(100):
            A = A / A.sum(axis=1, keepdims=True)
            A = A / A.sum(axis=0, keepdims=True)

        decomposition = birkhoff_decomposition(A)
        assert verify_decomposition(A, decomposition)

    def test_sparse_doubly_stochastic(self):
        """Sparse doubly stochastic matrix."""
        # Circulant permutation average
        n = 4
        matrix = np.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 0.5
            matrix[i, (i + 1) % n] = 0.5

        assert is_doubly_stochastic(matrix)
        decomposition = birkhoff_decomposition(matrix)
        assert verify_decomposition(matrix, decomposition)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
