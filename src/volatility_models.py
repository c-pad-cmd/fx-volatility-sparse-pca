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

Scaling note
------------
SPC score series have tiny variance (~1e-3 to 1e-6), which `arch`'s default
optimizer handles poorly -- it can silently return the unoptimized starting
values instead of the MLE. We fit on ``r * scale`` as `arch` recommends, then
analytically map the log-likelihood, parameters, and parameter covariance
back to the original scale of ``r``, so AIC/BIC/ICOMP and the fitted
parameters are all reported in ``r``'s original units regardless of
``scale``. (An earlier version of this module scaled the input but never
unscaled the reported log-likelihood/AIC/BIC/parameter-covariance -- the
model *selection* was still correct, since the same constant scale shift
applied to every model being compared, but the printed absolute numbers
were off by a large constant.)
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
    loglikelihood: float
    params: pd.Series
    param_cov: pd.DataFrame
    aic: float
    bic: float
    icomp: float
    n_params: int


def _rescale_transform(family: str, params: pd.Series, scale: float):
    """Map parameters fit on ``scale * r`` back to the original scale of
    ``r``, and return the Jacobian d(params_orig)/d(params_scaled) so the
    parameter covariance matrix can be transformed consistently.

    family: "garch" (plain or GJR -- omega multiplies squared-residual
            terms, so it scales like 1/scale**2) or "egarch" (omega enters
            additively in the log-variance recursion, so it shifts instead
            of scaling; alpha/gamma/beta are scale-invariant in both cases
            since they multiply/are standardized residuals).
    """
    names = list(params.index)
    n = len(names)
    jac = np.eye(n)
    orig = params.copy()
    omega_idx = names.index("omega")

    if family == "garch":
        orig.iloc[omega_idx] = params.iloc[omega_idx] / scale ** 2
        jac[omega_idx, :] = 0.0
        jac[omega_idx, omega_idx] = 1.0 / scale ** 2
    elif family == "egarch":
        beta_idx = names.index("beta[1]")
        beta = params.iloc[beta_idx]
        orig.iloc[omega_idx] = params.iloc[omega_idx] - 2 * np.log(scale) * (1 - beta)
        jac[omega_idx, :] = 0.0
        jac[omega_idx, omega_idx] = 1.0
        jac[omega_idx, beta_idx] = 2 * np.log(scale)
    else:
        raise ValueError(f"unknown family {family!r}")

    return orig, jac


def _icomp(loglikelihood: float, n: int, param_cov: np.ndarray) -> float:
    """Bozdogan's ICOMP: -2*logL + 2*log(n)*C1F(param_cov).

    C1F is the same "information complexity" penalty used throughout the
    course: it grows with how unevenly the eigenvalues of the parameter
    covariance matrix are spread out (i.e. penalizes ill-conditioned /
    imprecisely-estimated fits more than a flat parameter count would).
    """
    eigvals = np.linalg.eigvalsh(param_cov)
    eigvals = eigvals[eigvals > 0]
    if eigvals.size == 0:
        return np.nan
    mean_eig = eigvals.mean()
    c1f = (1.0 / (4 * mean_eig ** 2)) * np.sum((eigvals - mean_eig) ** 2)
    return -2 * loglikelihood + 2 * np.log(n) * c1f


def _fit_one(r: pd.Series, name: str, vol: str, p: int, q: int, o: int = 0,
             scale: float = 100.0) -> FittedVolModel:
    model = arch_model(r * scale, mean="Zero", vol=vol, p=p, o=o, q=q,
                        dist="normal")
    result = model.fit(disp="off", cov_type="robust", show_warning=False)
    n = len(r)

    family = "egarch" if vol.lower() == "egarch" else "garch"
    loglik = result.loglikelihood + n * np.log(scale)
    params_orig, jac = _rescale_transform(family, result.params, scale)
    param_cov_orig = jac @ np.asarray(result.param_cov) @ jac.T

    k = result.params.shape[0]
    aic = -2 * loglik + 2 * k
    bic = -2 * loglik + np.log(n) * k
    icomp = _icomp(loglik, n, param_cov_orig)

    return FittedVolModel(
        name=name,
        result=result,
        loglikelihood=loglik,
        params=params_orig,
        param_cov=pd.DataFrame(param_cov_orig, index=params_orig.index,
                                columns=params_orig.index),
        aic=aic,
        bic=bic,
        icomp=icomp,
        n_params=k,
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
                fit = _fit_one(r, f"GARCH({p},{q})", vol="Garch", p=p, q=q,
                                scale=scale)
                aic, bic, icomp = fit.aic, fit.bic, fit.icomp
            except Exception:
                aic = bic = icomp = np.nan
            rows.append({"p": p, "q": q, "AIC": aic, "BIC": bic, "ICOMP": icomp})
    return pd.DataFrame(rows)


def fit_garch(r: pd.Series, p: int, q: int, scale: float = 100.0) -> FittedVolModel:
    """Fit a single GARCH(p, q) model and return the full result, correctly
    unscaled -- used for the Q3(b) forecast step, which needs the fitted
    `arch` result object itself (for `.forecast()`/`.simulate()`), not just
    the AIC/BIC/ICOMP table that `grid_search_garch` returns.
    """
    return _fit_one(r, f"GARCH({p},{q})", vol="Garch", p=p, q=q, scale=scale)


def best_by_criterion(table: pd.DataFrame, criterion: str) -> pd.Series:
    """Return the row of ``table`` that minimizes ``criterion`` (ignoring NaNs)."""
    valid = table.dropna(subset=[criterion])
    return valid.loc[valid[criterion].idxmin()]
