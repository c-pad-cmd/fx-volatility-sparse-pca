"""Unit tests for src.data_utils -- the parts of the pipeline that don't
require sklearn/arch, so they run with only numpy/pandas/openpyxl installed.

Run with:  python -m unittest discover -v
"""
import unittest
from pathlib import Path

import numpy as np

from src.data_utils import load_fx_data, normalize_columns

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "FxData.xlsx"


class TestNormalizeColumns(unittest.TestCase):
    def test_synthetic_matrix_is_centered_and_unit_norm(self):
        rng = np.random.default_rng(0)
        X = rng.normal(loc=5.0, scale=2.0, size=(50, 4))
        X_norm, mu, d = normalize_columns(X)

        np.testing.assert_allclose(X_norm.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(
            np.sqrt((X_norm ** 2).sum(axis=0)), 1.0, atol=1e-10
        )
        np.testing.assert_allclose(mu, X.mean(axis=0))

    def test_constant_column_does_not_divide_by_zero(self):
        X = np.column_stack([np.ones(10), np.arange(10, dtype=float)])
        X_norm, mu, d = normalize_columns(X)

        self.assertTrue(np.all(np.isfinite(X_norm)))
        self.assertEqual(d[0], 1.0)  # zero-norm column falls back to 1.0
        np.testing.assert_allclose(X_norm[:, 0], 0.0)


class TestLoadFxData(unittest.TestCase):
    def setUp(self):
        if not DATA_PATH.exists():
            self.skipTest(f"FxData.xlsx not found at {DATA_PATH}")

    def test_shape_and_columns(self):
        df = load_fx_data(DATA_PATH)
        self.assertEqual(df.shape, (3202, 20))
        self.assertEqual(
            list(df.columns),
            ["EUR", "JPY", "GBP", "CHF", "AUD", "NZD", "CAD", "NOK", "SEK",
             "PLN", "CZK", "HUF", "TRY", "ILS", "ZAR", "MXN", "BRL", "KRW",
             "IDR", "SGD"],
        )

    def test_values_are_numeric_and_finite(self):
        df = load_fx_data(DATA_PATH)
        X = df.to_numpy(dtype=float)
        self.assertTrue(np.all(np.isfinite(X)))
        self.assertTrue(np.all(X > 0))  # FX levels are strictly positive


if __name__ == "__main__":
    unittest.main()
