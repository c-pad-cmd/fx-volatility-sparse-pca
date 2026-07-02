"""Regression tests for src.sparse_pca -- specifically the alpha=1.0 bug
where sklearn's SparsePCA collapsed every loading to exactly zero on this
correlation-scale-normalized data, which then silently corrupted the
information-criteria table and crashed the downstream GARCH fit.

Run with:  python -m unittest discover -v
"""
import unittest

import numpy as np

from src.data_utils import normalize_columns
from src.sparse_pca import fit_sparse_pca


def _synthetic_correlated_data(rng: np.random.Generator, n: int = 500, p: int = 10):
    """A handful of correlated columns, normalized the same way as the FX
    pipeline (unit column norm, not unit variance) -- this is what made
    alpha=1.0 catastrophically too large in the original bug.
    """
    latent = rng.normal(size=(n, 3))
    loadings = rng.normal(size=(3, p))
    X = latent @ loadings + 0.1 * rng.normal(size=(n, p))
    X_norm, _, _ = normalize_columns(X)
    return X_norm


class TestFitSparsePCA(unittest.TestCase):
    def setUp(self):
        self.X = _synthetic_correlated_data(np.random.default_rng(0))

    def test_default_alpha_does_not_collapse_to_zero(self):
        result = fit_sparse_pca(self.X, n_components=3)
        nonzeros = (result.loadings != 0).sum(axis=0)
        self.assertTrue(np.all(nonzeros > 0),
                         "default alpha produced an all-zero component")
        self.assertTrue(np.all(np.isfinite(result.adjusted_variance)))

    def test_too_large_alpha_raises_instead_of_silently_collapsing(self):
        # alpha=1.0 is the original (buggy) default; on unit-column-norm
        # data this drives every loading to zero. Assert it's now caught
        # loudly rather than propagating NaNs downstream.
        with self.assertRaises(ValueError):
            fit_sparse_pca(self.X, n_components=3, alpha=1.0)


if __name__ == "__main__":
    unittest.main()
