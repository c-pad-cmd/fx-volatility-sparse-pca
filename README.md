# FX Sparse PCA + GARCH Volatility Modeling

Dimension reduction and conditional-volatility modeling for a panel of 20
foreign-exchange rates, using sparse PCA to compress correlated currencies
into a few interpretable components and a GARCH-family model comparison to
capture the volatility clustering in each one.

This is a Python port of a graduate time-series course project (Stat 575:
Time Series Analysis, Fall 2024) originally implemented in MATLAB. The
methodology is unchanged; the implementation is not — see
["What changed from the original"](#what-changed-from-the-original) below.

## What it does

1. **Dimension reduction.** Loads 3,202 daily observations of 20 exchange
   rates (EUR, JPY, GBP, CHF, AUD, NZD, CAD, NOK, SEK, PLN, CZK, HUF, TRY,
   ILS, ZAR, MXN, BRL, KRW, IDR, SGD), normalizes them to a correlation
   scale, and fits sparse PCA for a range of component counts. The number of
   components is chosen by scoring each fit with four information criteria
   (AIC, CAIC, SBC, and ICOMP — the last one a Bozdogan information-
   complexity penalty rather than a flat parameter count).
2. **Structure check.** Stem-plots the sparse loadings and plots the ACF/PACF
   of the resulting component scores to check for the autocorrelation and
   volatility clustering that motivate a GARCH model.
3. **Volatility model comparison.** Fits GARCH(1,1), EGARCH(1,1), and
   GJR-GARCH(1,1) to each retained component and picks the best one per
   component by AIC / BIC / ICOMP.
4. **Order search + forecast.** Grid-searches GARCH(p, q) for p, q up to 4 on
   the component with the most explained variance, and forecasts the
   conditional variance and simulated returns from the best-fitting model.

## Repo structure

```
fx-volatility-sparse-pca/
├── data/FxData.xlsx           # daily FX levels, 20 currencies x 3202 days
├── src/
│   ├── data_utils.py          # loading + correlation-scale normalization
│   ├── sparse_pca.py          # sparse PCA fit + AIC/CAIC/SBC/ICOMP scoring
│   ├── volatility_models.py   # GARCH/EGARCH/GJR fitting + IC comparison
│   └── plotting.py            # stem plots, ACF/PACF, forecast plots
├── tests/test_data_utils.py   # unit tests for the dependency-light parts
├── run_analysis.py            # end-to-end driver script
├── requirements.txt
└── LICENSE                    # MIT
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_analysis.py --data data/FxData.xlsx --out results/
```

This writes information-criteria tables (CSV) and figures (PNG) to
`results/` (git-ignored — regenerate anytime by rerunning). Runtime is a few
minutes, mostly spent fitting the GARCH(p, q) grid. After a run, it's worth
copying 2-3 of the more interesting figures (e.g. the SPC stem plot and the
GARCH forecast plot) into a committed `docs/` folder and linking them here so
the results are visible without cloning the repo.

## Tests

```bash
python -m unittest discover -v
```

Covers `normalize_columns` (centering/unit-norm properties, zero-variance
column edge case) and `load_fx_data` (shape, column names, numeric sanity)
against the real dataset — the parts of the pipeline that don't require
`sklearn`/`arch`, so they run with just numpy/pandas/openpyxl installed. All
4 pass as of this commit.

## What changed from the original

The original MATLAB analysis leaned on two pieces of third-party code that
aren't mine to redistribute, so this port replaces them with standard,
appropriately-licensed Python equivalents rather than translating them
line-for-line:

- **Sparse PCA algorithm.** The MATLAB script used Hein & Bühler's inverse
  power method for nonlinear eigenproblems (NIPS 2010, GPL-licensed research
  code). This port uses `sklearn.decomposition.SparsePCA`, a coordinate-
  descent / elastic-net formulation (Zou, Hastie & Tibshirani, 2006). Same
  goal — sparse loadings — different algorithm, so the exact loading values
  and information-criteria numbers won't match the original MATLAB run.
- **GARCH estimation.** MATLAB's Econometrics Toolbox (`garch`, `egarch`,
  `gjr`) is replaced with the `arch` package, the standard Python library for
  conditional volatility models, using the equivalent (1,1) specifications
  for each model family.
- **Information-criteria formulas** (AIC, CAIC, SBC, and the ICOMP
  information-complexity criterion used throughout the course) are
  reimplemented in `src/sparse_pca.py` and `src/volatility_models.py`
  directly from their published mathematical definitions — not copied from
  the instructor-provided MATLAB template, which carries its own
  proprietary/no-redistribution notice.
- The "adjusted variance" used in the sparse-PCA information criteria is
  computed generically via QR decomposition of the component scores (Zou,
  Hastie & Tibshirani, 2006, Sec. 3.4), so it works with any sparse PCA
  algorithm rather than depending on values a specific solver happens to
  report internally.

Because of the algorithm swap, don't expect this port's numbers to match the
original write-up's Table 1 (e.g. AIC = -971939.87 for the first component).
The point of the port is a clean, reusable, appropriately-licensed
implementation of the same modeling workflow, not a bit-for-bit
reproduction.

## Data note

`data/FxData.xlsx` contains historical FX rate levels compiled for the
original coursework. These are factual exchange-rate levels rather than
copyrightable content, but if you'd rather source your own, any daily FX
series from a public source (e.g. FRED, ECB) with the same 20-currency,
long-panel structure will work as a drop-in replacement.

## Verification status

The data loading, correlation-scale normalization, and information-criteria
math (`src/data_utils.py`, `src/sparse_pca.py`'s scoring functions) were
tested end-to-end against the real 3,202 x 20 FX dataset and produce
correctly-shaped, finite, sane-magnitude results. The `sklearn`/`arch`
-dependent model-fitting calls were validated for control flow and API usage
against the documented interfaces, but not executed against live network-
installed packages in the environment this was built in — run
`pip install -r requirements.txt && python run_analysis.py` once locally to
confirm before you rely on the numbers.

## License

MIT (see `LICENSE`). Unlike the original coursework, which depended on a
GPL-licensed MATLAB toolbox for the sparse PCA step, everything in this repo
is either original code or built on permissively-licensed libraries
(scikit-learn, `arch`, statsmodels), so it's freely reusable.
