"""Plot helpers mirroring the figures in the original write-up."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
