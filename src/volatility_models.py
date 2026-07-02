"""
GARCH-family volatility modeling for a single sparse-PC score series.

Ports the MATLAB Econometrics Toolbox workflow (``garch``, ``egarch``,
``gjr`` + ``estimate``) to Python using the ``arch`` package, which is the
standard library for conditional-volatility modeling in Python.

Model correspondence:
    MATLAB garch(1,1)   ->  arch_model(r, mean="Zero", vol="Garch",  p=1, q=1)
    MATLAB egarch(1,1)  ->  arch_model(r, mean="Zero", vol="EGARCH", p=1, o=1, q=1)
    MATLAB gjr(1,1)     ->  arch_model(r, mean="Zero", vol="Garch",  p=1, o=1, q=1)

All three are fit with a zero-mean equation, matching the original code,
which passed the raw SPC score series straight to ``estimate`` without an
ARMA mean model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult


@dataclass
class FittedVolModel:
    name: str
    result: ARCHModelResult
    aic: float
    bic: float
    icomp: float
    n_params: int


def _icomp(result: ARCHModelResult, n: int) -> float:
    """Bozdogan's ICOMP: -2*logL + 2*log(n)*C1F(param_cov).

    C1F is the same "information complexity" penalty used throughout the
    course: it grows with how unevenly the eigenvalues of the parameter
    covariance matrix are spread out (i.e. penalizes ill-conditioned /
    imprecisely-estimated fits more than a flat parameter count would).
    """
    cov = np.asarray(result.param_cov)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 0]
    if eigvals.size == 0:
        return np.nan
    mean_eig = eigvals.mean()
    c1f = (1.0 / (4 * mean_eig ** 2)) * np.sum((eigvals - mean_eig) ** 2)
    return -2 * result.loglikelihood + 2 * np.log(n) * c1f


def _fit_one(r: pd.Series, name: str, vol: str, p: int, q: int, o: int = 0,
             scale: float = 100.0) -> FittedVolModel:
    model = arch_model(r * scale, mean="Zero", vol=vol, p=p, o=o, q=q,
                        dist="normal")
    result = model.fit(disp="off", cov_type="robust")
    n = len(r)
    return FittedVolModel(
        name=name,
        result=result,
        aic=result.aic,
        bic=result.bic,
        icomp=_icomp(result, n),
        n_params=result.params.shape[0],
    )


def compare_garch_family(r: pd.Series) -> tuple[pd.DataFrame, dict[str, FittedVolModel]]:
    """Fit GARCH(1,1), EGARCH(1,1) and GJR-GARCH(1,1) and score with AIC/BIC/ICOMP."""
    fits = {
        "GARCH(1,1)": _fit_one(r, "GARCH(1,1)", vol="Garch", p=1, q=1),
        "EGARCH(1,1)": _fit_one(r, "EGARCH(1,1)", vol="EGARCH", p=1, o=1, q=1),
        "GJR-GARCH(1,1)": _fit_one(r, "GJR-GARCH(1,1)", vol="Garch", p=1, o=1, q=1),
    }
    table = pd.DataFrame(
        [{"Model": name, "AIC": f.aic, "BIC": f.bic, "ICOMP": f.icomp}
         for name, f in fits.items()]
    ).set_index("Model")
    return table, fits


def grid_search_garch(r: pd.Series, p_max: int = 4, q_max: int = 4,
                       scale: float = 100.0) -> pd.DataFrame:
    """Fit GARCH(p, q) for p in 1..p_max, q in 1..q_max, scoring each fit.

    Mirrors the ``pmax = 4, qmax = 4`` grid search in the write-up. Models
    that fail to converge are recorded as NaN rather than raising, matching
    the try/catch behavior of the original script.
    """
    rows = []
    n = len(r)
    for p in range(1, p_max + 1):
        for q in range(1, q_max + 1):
            try:
                model = arch_model(r * scale, mean="Zero", vol="Garch",
                                    p=p, q=q, dist="normal")
                result = model.fit(disp="off", cov_type="robust")
                aic, bic = result.aic, result.bic
                icomp = _icomp(result, n)
            except Exception:
                aic = bic = icomp = np.nan
            rows.append({"p": p, "q": q, "AIC": aic, "BIC": bic, "ICOMP": icomp})
    return pd.DataFrame(rows)


def best_by_criterion(table: pd.DataFrame, criterion: str) -> pd.Series:
    """Return the row of ``table`` that minimizes ``criterion`` (ignoring NaNs)."""
    valid = table.dropna(subset=[criterion])
    return valid.loc[valid[criterion].idxmin()]
