"""
End-to-end pipeline: sparse PCA dimension reduction of 20 FX exchange-rate
series, followed by GARCH-family volatility modeling of the selected
components.

Python port of "Stat 575: Time Series Analysis, Project #2" (Fall 2024).
See README.md for methodology notes and what was changed vs. the original
MATLAB analysis.

Usage:
    python run_analysis.py --data data/FxData.xlsx --out results/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_utils import load_fx_data, normalize_columns
from src.sparse_pca import (fit_sparse_pca, information_criteria,
                             select_number_of_components)
from src.volatility_models import (best_by_criterion, compare_garch_family,
                                    grid_search_garch)
from src.plotting import acf_pacf_plots, forecast_plot, stem_plots

N_COMPONENTS = 3          # number of SPCs carried forward, per the write-up
SPARSITY_ALPHA = 1.0      # L1 penalty for sklearn's SparsePCA
NUM_LAGS = 19
P_MAX = Q_MAX = 4


def main(data_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"

    # ---- 1(a): load data, choose number of SPCs via information criteria ----
    df = load_fx_data(data_path)
    currency_labels = list(df.columns)
    X_raw = df.to_numpy(dtype=float)
    X, mu, d = normalize_columns(X_raw)
    n, p = X.shape
    print(f"Loaded FX data: n={n} observations, p={p} currencies")

    k_max = min(p - 1, 10)  # full sweep to p-1 like the original; capped for runtime
    ic_table = select_number_of_components(X, k_max=k_max, alpha=SPARSITY_ALPHA)
    ic_table.to_csv(out_dir / "sparse_pca_information_criteria.csv")
    print("\nInformation criteria by number of SPCs:")
    print(ic_table.round(2))

    best_k = {crit: int(ic_table[crit].idxmin()) for crit in ic_table.columns}
    print(f"\nBest k by criterion: {best_k}")

    # ---- 1(b): fit the chosen model, stem-plot the loadings ----
    result = fit_sparse_pca(X, N_COMPONENTS, alpha=SPARSITY_ALPHA)
    stem_plots(result.loadings, currency_labels, fig_dir)

    explained_pct = 100 * result.adjusted_variance / result.adjusted_variance.sum()
    print(f"\nPercent adjusted variance explained by first {N_COMPONENTS} SPCs:")
    for j, pct in enumerate(explained_pct, start=1):
        print(f"  SPC{j}: {pct:.2f}%")

    # ---- 1(c): ACF / PACF of the SPC scores ----
    acf_pacf_plots(result.scores, N_COMPONENTS, NUM_LAGS, fig_dir)

    # ---- 2: GARCH / EGARCH / GJR-GARCH model comparison per SPC ----
    all_garch_tables = {}
    for j in range(N_COMPONENTS):
        r = pd.Series(result.scores[:, j])
        table, fits = compare_garch_family(r)
        all_garch_tables[f"SPC{j + 1}"] = table
        table.to_csv(out_dir / f"spc{j + 1}_garch_family_ic.csv")
        print(f"\nGARCH-family comparison for SPC{j + 1}:")
        print(table.round(2))
        for crit in ("AIC", "BIC", "ICOMP"):
            best_model = table[crit].idxmin()
            print(f"  Best by {crit}: {best_model}")

    # ---- 3(a): GARCH(p,q) grid search on SPC1 (largest explained variance) ----
    r1 = pd.Series(result.scores[:, 0])
    grid = grid_search_garch(r1, p_max=P_MAX, q_max=Q_MAX)
    grid.to_csv(out_dir / "spc1_garch_grid_search.csv", index=False)
    print("\nGARCH(p, q) grid search on SPC1:")
    print(grid.round(2))

    best_aic = best_by_criterion(grid, "AIC")
    print(f"\nBest GARCH(p,q) by AIC: p={int(best_aic.p)}, q={int(best_aic.q)}")

    # ---- 3(b): forecast with the best-fitting GARCH model ----
    from arch import arch_model

    p_best, q_best = int(best_aic.p), int(best_aic.q)
    scale = 100.0
    model = arch_model(r1 * scale, mean="Zero", vol="Garch", p=p_best, q=q_best)
    fit = model.fit(disp="off")
    print(fit.summary())

    horizon = 200
    fc = fit.forecast(horizon=horizon, reindex=False)
    forecast_variance = fc.variance.values[-1] / (scale ** 2)

    omega = fit.params["omega"]
    alpha_sum = sum(v for k, v in fit.params.items() if k.startswith("alpha"))
    beta_sum = sum(v for k, v in fit.params.items() if k.startswith("beta"))
    theoretical_variance = (omega / (scale ** 2)) / (1 - alpha_sum - beta_sum)

    sim = model.simulate(fit.params, nobs=20)
    simulated_returns = sim["data"].to_numpy() / scale

    forecast_plot(simulated_returns, forecast_variance, theoretical_variance,
                  fig_dir / "spc1_garch_forecast.png")

    print(f"\nDone. Figures and CSV tables written to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/FxData.xlsx"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    main(args.data, args.out)
