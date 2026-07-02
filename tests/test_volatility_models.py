"""Regression tests for src.volatility_models -- specifically the bug where
`_fit_one`/`grid_search_garch` scaled the input series before fitting (as
`arch` recommends for numerical stability) but never scaled the reported
log-likelihood/AIC/BIC/parameter-covariance back down. Model *selection*
was unaffected (the same constant shift applied to every model compared),
but the printed absolute numbers were off by a large, scale-dependent
constant, and the fitted `omega` was left in the wrong units.

The regression test below fits the same series at several different (but
all well-conditioned) `scale` values and checks the *unscaled* AIC/BIC/
ICOMP/omega agree closely -- if the unscaling math were wrong, or missing,
these would differ by orders of magnitude depending on `scale`.

Run with:  python -m unittest discover -v
"""
import unittest
import warnings

import numpy as np
import pandas as pd

from src.volatility_models import _fit_one, compare_garch_family


def _simulate_garch11(rng: np.random.Generator, n: int, omega: float,
                       alpha: float, beta: float) -> pd.Series:
    sigma2 = np.empty(n)
    r = np.empty(n)
    sigma2[0] = omega / (1 - alpha - beta)
    r[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    return pd.Series(r)


class TestScaleInvariance(unittest.TestCase):
    """`scale` is an internal numerical-conditioning knob; results in the
    original units of `r` should not depend on which (well-conditioned)
    value is used.
    """

    def setUp(self):
        warnings.filterwarnings("ignore")
        rng = np.random.default_rng(0)
        self.r = _simulate_garch11(rng, n=2000, omega=1e-6, alpha=0.1, beta=0.85)
        # 50-500 all keep std(r)*scale within arch's well-conditioned range
        self.scales = [50.0, 100.0, 200.0, 500.0]

    def test_garch_aic_bic_icomp_are_scale_invariant(self):
        fits = [_fit_one(self.r, "GARCH(1,1)", vol="Garch", p=1, q=1, scale=s)
                for s in self.scales]
        aics = [f.aic for f in fits]
        bics = [f.bic for f in fits]
        icomps = [f.icomp for f in fits]
        self.assertAlmostEqual(max(aics) - min(aics), 0.0, places=2)
        self.assertAlmostEqual(max(bics) - min(bics), 0.0, places=2)
        self.assertAlmostEqual(max(icomps) - min(icomps), 0.0, places=1)

    def test_garch_omega_is_scale_invariant(self):
        omegas = [_fit_one(self.r, "GARCH(1,1)", vol="Garch", p=1, q=1,
                            scale=s).params["omega"] for s in self.scales]
        self.assertAlmostEqual(max(omegas), min(omegas), places=8)

    def test_egarch_aic_and_omega_are_scale_invariant(self):
        fits = [_fit_one(self.r, "EGARCH(1,1)", vol="EGARCH", p=1, o=1, q=1,
                          scale=s) for s in self.scales]
        aics = [f.aic for f in fits]
        omegas = [f.params["omega"] for f in fits]
        self.assertAlmostEqual(max(aics) - min(aics), 0.0, places=1)
        self.assertAlmostEqual(max(omegas), min(omegas), places=2)


class TestCompareGarchFamily(unittest.TestCase):
    def test_runs_and_picks_a_finite_best_model(self):
        warnings.filterwarnings("ignore")
        rng = np.random.default_rng(1)
        r = _simulate_garch11(rng, n=1500, omega=5e-7, alpha=0.08, beta=0.88)
        table, fits = compare_garch_family(pd.Series(r))
        self.assertTrue(np.all(np.isfinite(table.to_numpy())))
        best = table["AIC"].idxmin()
        self.assertIn(best, fits)


if __name__ == "__main__":
    unittest.main()
