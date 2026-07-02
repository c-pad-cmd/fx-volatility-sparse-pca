"""
Data loading utilities for the FX exchange-rate dimension-reduction /
volatility-modeling pipeline.

The original data (``FxData.xlsx``) contains daily levels (not returns) for
p = 20 exchange rates against a base currency, T = n = 3202 trading days:

    EUR, JPY, GBP, CHF, AUD, NZD, CAD, NOK, SEK, PLN,
    CZK, HUF, TRY, ILS, ZAR, MXN, BRL, KRW, IDR, SGD

Row 1 of the sheet is a title cell, row 2 holds the currency tickers, and the
data starts on row 3.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_fx_data(path: str | Path, sheet_name: str = "Sheet 1 - FxData") -> pd.DataFrame:
    """Load the raw FX level data into a tidy DataFrame (n_obs x 20 currencies).

    Mirrors ``X = xlsread('FxData.xlsx')`` from the original MATLAB script,
    which reads only the numeric block and drops the title row.
    """
    df = pd.read_excel(path, sheet_name=sheet_name, header=1)
    df = df.dropna(axis=1, how="all")  # drop the trailing empty columns
    df = df.dropna(axis=0, how="all")
    df = df.reset_index(drop=True)
    return df


def normalize_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center each column and rescale it to unit Euclidean length.

    This is a direct port of the course's ``normalize.m`` helper: after this
    transform, ``X.T @ X`` is exactly the sample correlation matrix of the
    original variables, which is the scaling the sparse-PCA / information-
    criteria formulas in the write-up assume.

    Returns
    -------
    X_norm : the transformed matrix
    mu     : the column means that were subtracted
    d      : the Euclidean column norms used to rescale (post-centering)
    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    Xc = X - mu
    d = np.sqrt((Xc ** 2).sum(axis=0))
    d[d == 0] = 1.0
    X_norm = Xc / d
    return X_norm, mu, d
