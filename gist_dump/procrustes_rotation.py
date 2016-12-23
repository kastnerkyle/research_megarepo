# (C) Kyle Kastner, June 2014
# License: BSD 3 clause

import numpy as np
from scipy import linalg

from sklearn.utils import array2d, as_float_array
from sklearn.utils.extmath import svd_flip
from sklearn.utils.testing import assert_array_almost_equal


def procrustes_rotation(X1, X2, copy=True):
    """Apply optimal rotation and scaling matrix between two matrices.

    Parameters
    ----------
        X1, X2: array-likes with the same shape (n_samples, n_features)

    Returns
    -------
        X2_t : array-like, shape (n_samples, n_features)
    """
    X1 = as_float_array(array2d(X1), copy=copy)
    X2 = as_float_array(array2d(X2), copy=copy)

    X1_mean = X1.mean(0)
    X2_mean = X2.mean(0)

    X1 -= X1_mean
    X2 -= X2_mean

    X1_norm = linalg.norm(X1, 'fro')
    X2_norm = linalg.norm(X1, 'fro')

    X1 /= X1_norm
    X2 /= X2_norm

    U, S, V = linalg.svd(np.dot(X1.T, X2), full_matrices=False)
    U, V = svd_flip(U, V)
    R = np.dot(V.T, U.T)
    X2_t = np.sum(S) * X1_norm * np.dot(X2, R) + X1_mean
    return X2_t


def test_procrustes_rotation():
    """Test that Procrustes rotation works correctly."""
    for i in range(3):
        rng = np.random.RandomState(i)
        n_samples = 100
        n_features = 2
        X1 = rng.randn(n_samples, n_features) + 5 * rng.rand(1, n_features)
        # Simple reflection matrix
        X_reflect = np.array([[1, 0],
                              [0, -1]])
        Y_reflect = np.array([[-1, 0],
                              [0, 1]])
        arb_rotation = np.array([[np.cos(np.pi / 3), -np.sin(np.pi / 3)],
                                 [np.sin(np.pi / 3), np.cos(np.pi / 3)]])
        for R in [X_reflect, Y_reflect, arb_rotation]:
            X2 = np.dot(X1, R)
            X3 = procrustes_rotation(X1, X2)
            assert_array_almost_equal(X1, X3, decimal=6)


if __name__ == "__main__":
    test_procrustes_rotation()