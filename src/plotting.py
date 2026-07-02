"""Plot helpers mirroring the figures in the original write-up."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def stem_plots(loadings: np.ndarray, currency_labels: list[str], out_dir: Path) -> None:
    """One stem plot per component, plus an overlay of all components."""
    out_dir.mkdir(parents=True, exist_ok=True)
    k = loadings.shape[1]
    colors = ["r", "b", "m", "g", "c", "y"]

    for j in range(k):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.stem(np.arange(1, loadings.shape[0] + 1), loadings[:, j],
                linefmt=colors[j % len(colors)], basefmt=" ")
        ax.set_title(f"SPC{j + 1}")
        ax.set_xticks(range(1, loadings.shape[0] + 1))
        ax.set_xticklabels(currency_labels, rotation=90, fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"spc{j + 1}_stem.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j in range(k):
        ax.stem(np.arange(1, loadings.shape[0] + 1), loadings[:, j],
                linefmt=colors[j % len(colors)], basefmt=" ",
                label=f"SPC{j + 1}")
    ax.set_title(f"First {k} SPCs")
    ax.set_xticks(range(1, loadings.shape[0] + 1))
    ax.set_xticklabels(currency_labels, rotation=90, fontsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "spc_overlay_stem.png", dpi=150)
    plt.close(fig)


def loadings_biplot(loadings: np.ndarray, currency_labels: list[str], out_path: Path,
                     i: int = 0, j: int = 1) -> None:
    """Scatter the currencies by their loadings on two components (default
    SPC1 vs. SPC2). Complements the per-component stem plots by showing the
    developed/emerging-market split as spatial clusters rather than as two
    separate bar charts.
    """
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    x, y = loadings[:, i], loadings[:, j]
    ax.axhline(0, color="0.75", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.75", linewidth=0.8, zorder=0)
    ax.scatter(x, y, s=45, color="#1f77b4", zorder=2)
    for label, xi, yi in zip(currency_labels, x, y):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(5, 5),
                    fontsize=8)
    ax.set_xlabel(f"SPC{i + 1} loading")
    ax.set_ylabel(f"SPC{j + 1} loading")
    ax.set_title(f"Currency loadings: SPC{i + 1} vs. SPC{j + 1}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def ic_vs_k_plot(ic_table: pd.DataFrame, out_path: Path) -> None:
    """Line plot of AIC/CAIC/SBC/ICOMP across k=1..k_max, visualizing the
    component-count sweep behind ``select_number_of_components`` (see the
    README caveat on why this curve isn't smooth: each k is an independent,
    non-convex SparsePCA refit with no shared warm start).
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for col in ic_table.columns:
        ax.plot(ic_table.index, ic_table[col], marker="o", label=col)
    ax.set_xlabel("Number of components (k)")
    ax.set_ylabel("Information criterion value")
    ax.set_title("Sparse PCA order selection: IC vs. k")
    ax.set_xticks(list(ic_table.index))
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def garch_family_bars(tables: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Grouped delta-IC bars per model family (GARCH/EGARCH/GJR-GARCH), one
    panel per SPC. Visualizes the "GARCH(1,1) wins unanimously" result that
    ``compare_garch_family`` otherwise only reports as a printed table.

    Plots each criterion relative to its own best (minimum) value within the
    panel, not the raw AIC/BIC/ICOMP magnitude: the three model families
    differ by only a few points out of a shared ~11,000 baseline, so raw
    values render as visually identical bars. The relative gap is what
    actually drives model selection.
    """
    names = list(tables.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        table = tables[name]
        delta = table - table.min()
        models = list(delta.index)
        criteria = list(delta.columns)
        x = np.arange(len(models))
        width = 0.8 / len(criteria)
        for k, crit in enumerate(criteria):
            ax.bar(x + k * width, delta[crit].to_numpy(), width, label=crit)
        ax.set_xticks(x + width * (len(criteria) - 1) / 2)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
        ax.set_title(name)
        ax.set_ylabel("Delta vs. best (lower = better)")
        ax.legend(fontsize=7)
    fig.suptitle("GARCH-family model comparison")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def garch_grid_heatmap(grid: pd.DataFrame, out_path: Path,
                        criterion: str = "AIC") -> None:
    """Heatmap of a GARCH(p,q) grid-search criterion over the (p, q) grid.
    Visualizes the "every larger model overfits" claim from the p,q<=4 sweep
    that ``grid_search_garch`` otherwise only reports as a CSV table.
    """
    pivot = grid.pivot(index="p", columns="q", values=criterion)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.to_numpy(), cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_title(f"GARCH(p,q) grid search on SPC1: {criterion}")
    for r_idx, p_val in enumerate(pivot.index):
        for c_idx, q_val in enumerate(pivot.columns):
            val = pivot.loc[p_val, q_val]
            if not np.isnan(val):
                ax.text(c_idx, r_idx, f"{val:.0f}", ha="center", va="center",
                        color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label=criterion)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def acf_pacf_plots(scores: np.ndarray, n_components: int, num_lags: int,
                    out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for j in range(n_components):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))
        plot_acf(scores[:, j], lags=num_lags, ax=ax1, title=f"SPC{j + 1} ACF")
        plot_pacf(scores[:, j], lags=num_lags, ax=ax2, title=f"SPC{j + 1} PACF",
                  method="ywm")
        fig.tight_layout()
        fig.savefig(out_dir / f"spc{j + 1}_acf_pacf.png", dpi=150)
        plt.close(fig)


def forecast_plot(simulated_returns: np.ndarray, forecast_variance: np.ndarray,
                   theoretical_variance: float, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    ax1.plot(np.arange(1, len(simulated_returns) + 1), simulated_returns, "b",
              linewidth=1.5)
    ax1.set_title("Simulated Returns")
    ax1.set_xlabel("Forecast Horizon")
    ax1.set_ylabel("Return")

    horizon = len(forecast_variance)
    ax2.plot(forecast_variance, "r", linewidth=2, label="Forecast")
    ax2.plot(np.full(horizon, theoretical_variance), "k--", linewidth=1.5,
              label="Theoretical")
    ax2.set_title("Forecast Conditional Variance")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
