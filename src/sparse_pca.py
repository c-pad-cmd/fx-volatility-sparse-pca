"""
Sparse PCA dimension reduction + information-criteria order selection.

Methodology notes
------------------
The original coursework used a MATLAB implementation of sparse PCA via the
inverse power method for nonlinear eigenproblems (Hein & Buhler, NIPS 2010).
That code is a third-party, GPL-licensed research toolbox and is not
reproduced here. This module instead uses ``sklearn.decomposition.SparsePCA``
(coordinate-descent / elastic-net formulation, Zou, Hastie & Tibshirani 2006)
to compute sparse loadings -- a different algorithm for the same conceptual
problem, so the exact loading values will differ from the original MATLAB
run, but the methodology around it (adjusted variance, order selection via
AIC/CAIC/SBC/ICOMP) is reimplemented from the published formulas rather than
copied from anyone's code.

The order-selection criteria follow the "sparse-PCA-as-a-factor-model"
heuristic used in the course (Prof. H. Bozdogan, Stat 575/579): treat the
implied covariance of a k-component sparse PCA fit as

    Cov_k = F F' + diag(adjusted_variance padded to length p)

and score it with Gaussian AIC/CAIC/SBC/ICOMP, where ICOMP substitutes
Bozdogan's information complexity penalty (2 log n * C1F) for the usual
parameter-count penalty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import SparsePCA


@dataclass
class SparsePCAResult:
    n_components: int
    loadings: np.ndarray          # (p, k) sparse loading matrix F
    scores: np.ndarray            # (n, k) SPC scores, X @ F
    adjusted_variance: np.ndarray  # (k,) non-redundant variance per SPC


def fit_sparse_pca(X: np.ndarray, n_components: int, alpha: float = 1.0,
                    random_state: int = 0) -> SparsePCAResult:
    """Fit a k-component sparse PCA model and compute adjusted variance.

    ``alpha`` controls the L1 sparsity penalty (higher = sparser loadings),
    playing the same role as the ``card`` (target cardinality) parameter in
    the original MATLAB call.
    """
    model = SparsePCA(n_components=n_components, alpha=alpha,
                       random_state=random_state)
    model.fit(X)
    F = model.components_.T  # sklearn returns (k, p); we want (p, k)
    scores = X @ F
    adj_var = adjusted_variance(scores)
    return SparsePCAResult(n_components, F, scores, adj_var)


def adjusted_variance(scores: np.ndarray) -> np.ndarray:
    """Non-redundant variance explained by (possibly correlated) components.

    Sparse PCs are generally not orthogonal, so naively summing their
    variances double-counts shared information. Zou, Hastie & Tibshirani
    (2006, Sec. 3.4) fix this with a QR decomposition of the score matrix:
    the diagonal of R gives the "adjusted" (non-redundant) variance
    contributed by each component, in the order they're taken.
    """
    _, R = np.linalg.qr(scores)
    return np.diag(R) ** 2


def information_criteria(X: np.ndarray, F: np.ndarray,
                          adj_var: np.ndarray) -> dict[str, float]:
    """AIC / CAIC / SBC / ICOMP for a k-component sparse PCA fit.

    Direct port of the scoring block from the course write-up:

        Cov_SPCA = F F' + diag(adj_var padded to length p)
        Lackoffit = n*p*log(2*pi) + n*log(det(Cov_SPCA)) + n*p
        params = k*p - k*(k-1)/2 + 1
        C1F = (1 / (4*mean(eig)^2)) * sum((eig - mean(eig))^2)

        AIC   = Lackoffit + 2*params
        CAIC  = Lackoffit + params*(log(n) + 1)
        SBC   = Lackoffit + log(n)*params
        ICOMP = CAIC + 2*C1F
    """
    n, p = X.shape
    k = F.shape[1]

    padded = np.zeros(p)
    padded[:k] = adj_var
    cov = F @ F.T + np.diag(padded)

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        # Numerically singular / not PD -- regularize slightly, same spirit
        # as the course code silently tolerating ill-conditioned fits.
        cov = cov + np.eye(p) * 1e-8
        sign, logdet = np.linalg.slogdet(cov)

    lack_of_fit = n * p * np.log(2 * np.pi) + n * logdet + n * p
    params = k * p - k * (k - 1) / 2 + 1

    eigvals = np.linalg.eigvalsh(cov)
    mean_eig = eigvals.mean()
    c1f = (1.0 / (4 * mean_eig ** 2)) * np.sum((eigvals - mean_eig) ** 2)

    aic = lack_of_fit + 2 * params
    caic = lack_of_fit + params * (np.log(n) + 1)
    sbc = lack_of_fit + np.log(n) * params
    icomp = caic + 2 * c1f

    return {"AIC": aic, "CAIC": caic, "SBC": sbc, "ICOMP": icomp}


def select_number_of_components(X: np.ndarray, k_max: int, alpha: float = 1.0,
                                 random_state: int = 0) -> pd.DataFrame:
    """Score k = 1..k_max sparse PCA fits and return an IC comparison table."""
    rows = []
    for k in range(1, k_max + 1):
        result = fit_sparse_pca(X, k, alpha=alpha, random_state=random_state)
        ic = information_criteria(X, result.loadings, result.adjusted_variance)
        rows.append({"Component": k, **ic})
    table = pd.DataFrame(rows).set_index("Component")
    return table
