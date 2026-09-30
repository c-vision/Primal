# python_examples

Python examples for **PrimalSolver** (the C99 convex optimization solver),
using the `primalsolver` Python package (ctypes bindings).

> These examples are part of the C repository and are versioned here. The Python
> *packaging* itself lives in a separate, locally-untracked project; only the
> examples are kept in-tree so they are published with the solver.

## Install the package

```sh
pip install primalsolver
```

until the package is on PyPI, build it from the packaging project
(`python/`, not tracked in this repo) with `pip install -e python`.

## Examples

- `quickstart.py` — a tiny LP, the API at a glance.
- `option_pricing.py` — no-arbitrage call-price bounds (two LPs).
- `transaction_cost.py` — transaction-cost LP (trading); optimum on a face.
- `multi_period.py` — two-period portfolio with turnover costs (LP).
- `index_tracking.py` — two-period index tracking with turnover (LP).
- `robust_cvar.py` — worst-case CVaR over a return box (LP).
- `cvar_portfolio.py` — CVaR portfolio (Rockafellar-Uryasev) as an LP.
- `mean_cvar.py` — mean-CVaR portfolio, optimum at the kink (LP).
- `evar_portfolio.py` — entropic value-at-risk portfolio (exponential cone).
- `markowitz_conic.py` — Markowitz factor model as a second-order cone program.
- `market_neutral.py` — market-neutral long-short with a quadratic risk cap (QP).
- `portfolio_large.py` — large Markowitz QP (quadratic objective).
- `factor_model.py` — minimum variance under a factor model (QP).
- `cardinality_portfolio.py` — min variance with a cardinality limit (MIQP).
- `regression_ls.py` — least-squares regression (SOCP + normal equations).
- `lsq_pos.py` — the `lsq_pos` SOCP form, cross-checked against a QP.
- `sharpe_ratio.py` — max-Sharpe via Charnes-Cooper (quadratic constraint).
- `risk_parity.py` — equal risk contribution (exponential cones).
- `market_impact.py` — Markowitz with market impact (power cones).
- `regression_regularized.py` — ridge / LASSO / 3-2 regression (conic penalties).
- `socp_robust.py` — least-squares regression (SOCP; complex).
- `portfolio_mgmt.py` — five conic portfolio models (min-var, tracking, max-return, 130/30, robust).
- `big_portfolio.py` — mean-variance + correlation-stress pipeline (SOCP + SDP).
- `risk_budgeting.py` / `risk_budgeting_ls.py` — risk budgeting (exp cones; the second is long-short MIO).
- `lsq_l1_penalty.py` — positivity LS with a weighted L1 penalty (rotated cone + per-asset cones).
- `sharpe_ratio_sectors.py` — long-only max-Sharpe with a sector cap.

### From the books ("Optimization Methods in Finance")

`omf_ex21`, `omf_ex25`, `omf_ex217`, `omf_duality` (LP), `omf_qp_grg` (QP),
`omf_ex312` and `omf_dedication` (dedication LPs; shadow prices + reduced costs),
`omf_ex410` (3 LPs), `omf_ex411` (arbitrage LP), `omf_financing`, `omf_workforce`
(LP + sensitivities), `omf_appendix_d` (LP), `omf_bb`, `omf_capital` (MILP),
`omf_ex118` (MIP + GMI cut), `omf_ex2001`/`omf_ex207` (tracking-error SOCP),
`omf_mvo`, `omf_bl` (Black-Litterman) frontier QPs, `omf_nearestcorr` (SDP).

### Other books and MOSEK ports

- MOSEK comparison: `lo1` (LP), `qo1` (QP), `mi1` (MIP).
- Books: `intlo_simplex` (Nemirovski, LP), `boyd_qp_box` (Boyd, QP), `hdb_kkt`
  (Handbook, LP dual).

Each builds its model, solves it and checks the result, printing `OK`/`FAIL` and
exiting non-zero on failure:

```sh
pip install primalsolver
for f in python_examples/*.py; do python "$f"; done
```

## Other example folders

- [`lp_examples/`](../lp_examples/README.md) — one model file per supported format
  (LP / OPF / CBF), solved with the `primal` command-line tool.
- [`c_examples/`](../c_examples/) — 171 C samples (LP/QP/SOCP/SDP/exp-power/MIP),
  each checking its own result.
