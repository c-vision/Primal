# examples/python

Runnable Python examples for **PrimalSolver**, the dependency-free convex
optimization solver in C99 (LP / QP / SOCP / SDP / exp-power / MIP). They use the
`primalsolver` package, a thin `ctypes` binding over the C library: **no compiler
is needed** and the whole public API (`PRIMAL_*`) is exposed.

Each example builds a model, solves it and **checks its own result**, printing
`OK`/`FAIL` and exiting non-zero on failure. They are development tests as much
as documentation.

## Install the package

From [PyPI](https://pypi.org/project/primalsolver/):

```sh
pip install primalsolver
```

Prebuilt wheels cover **macOS (universal2: Intel + Apple Silicon)** and **Linux
(x86_64, aarch64)**; Windows wheels are produced by the CI. The wheels are also
attached to the [latest release](https://github.com/c-vision/Primal/releases) if
you prefer to install a file directly.

From a checkout, build and install locally instead:

```sh
cd ../python                 # the packaging project (not tracked in the C repo)
python -m venv .venv && . .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .             # compiles the shared library and installs the package
```

Requirements: Python >= 3.8, no runtime dependencies beyond the standard library.
Recommended: run inside a virtual environment (`python -m venv .venv`).

## Run the examples

```sh
pip install primalsolver                 # or the wheel above
for f in examples/python/*.py; do python "$f"; done
```

or one at a time:

```sh
python examples/python/omf_dedication.py
```

## The API at a glance

```python
from primalsolver import Model, BK, CT, SENSE, SOLSTA, VAR_TYPE

with Model(maxcon=2, maxvar=2) as m:
    m.obj([0, 1], [-3.0, -2.0])          # objective coefficients
    m.a_row(0, [0, 1], [1.0, 1.0])       # a sparse row (or a_ij for one entry)
    m.con_bounds(0, BK.UP, up=4.0)       # row bound
    m.var_bounds(0, BK.RA, 0.0, 3.0)     # variable bound
    m.sense(SENSE.MIN)                   # or MAX
    r = m.solve()                        # r.x, r.y, r.objective, r.dual, r.solsta, r.pinf
```

Beyond the basics the wrapper covers quadratic objectives (`qobj`, `qobj_ij`) and
constraints (`qconk`), cones (`cone` with `CT.QUAD/RQUAD/PEXP/DEXP/PPOW/RPOW`),
MIP variable types (`var_type`, `VAR_TYPE`), semi-continuous/-integer, SDP bar
variables (`barvars`, `sparsesymmat`, `baraij`, `barcj`), parameters
(`int_param`/`dou_param`), warm start (`warm_start`), SOS (`sos`), AFE/ACC/DJC
(`afes`, `afe_entry`, `afe_g`, `domain`, `acc`, `djcs`, `djc`), and the data
getters (`getcj`, `getaij`, `getvarbound`, `getconbound`, `getreducedcosts`,
`getbarxj`, `getbarsj`, …). All ~547 `PRIMAL_*` functions are also reachable
directly as `primalsolver.PRIMAL_*`.

## Examples

### Basics

- `quickstart.py` — the smallest LP, the whole API in one screen
  (`max` written as `min`: `x=(3,1)`, obj `-11`).

### Trading and portfolio (from `examples/c/finance/`)

- `transaction_cost.py` — transaction-cost LP (StackOverflow 37586543); the
  optimum is a **face**, checked by value `0.8` and `x_i >= x0_i`, `e'x = 0`.
- `option_pricing.py` — no-arbitrage call-price bounds as two LPs; `[0, 0.25]`.
- `multi_period.py` — two-period portfolio with turnover costs; value `0.9`.
- `index_tracking.py` — two-period index tracking with turnover; value `0.08`.
- `robust_cvar.py` — worst-case CVaR over a return box; robust `-0.10` vs nominal
  `-0.15`.
- `cvar_portfolio.py` — CVaR portfolio (Rockafellar-Uryasev, LP): the objective
  equals the empirical CVaR of `x*`.
- `mean_cvar.py` — mean-CVaR; optimum at the kink `r0'w = r1'w`, `12/35`.
- `evar_portfolio.py` — entropic value-at-risk via **exponential cones**; checked
  against a golden-section brute force.
- `markowitz_conic.py` — Markowitz factor model as a second-order cone program;
  risk bound and inactive-constraint optimum.
- `market_neutral.py` — market-neutral long-short with a **quadratic** risk cap
  (`qconk`); checked against a 1-D brute force.
- `factor_model.py` — minimum variance under a factor model (QP); value `0.5`.
- `cardinality_portfolio.py` — min variance with a cardinality limit (**MIQP**).
- `portfolio_large.py` — large Markowitz QP (quadratic objective) with strong
  duality (`pobj ≈ dobj`).
- `regression_ls.py` — least-squares regression as an SOCP, checked against the
  normal equations.
- `lsq_pos.py` — the `lsq_pos` SOCP form (`0 <= w <= 1`), cross-checked vs a QP.
- `sharpe_ratio.py` — max-Sharpe via Charnes-Cooper (quadratic constraint);
  `sqrt(10)`.
- `sharpe_ratio_sectors.py` — long-only max-Sharpe with a sector diversification
  cap.
- `risk_parity.py` — equal risk contribution via **exponential cones**.
- `risk_budgeting.py` — risk budgeting (Spinu): target risk contributions
  `(0.8, 0.2)`.
- `risk_budgeting_ls.py` — **long-short** risk budgeting, mixed-integer conic.
- `market_impact.py` — Markowitz with market impact via a **power cone**
  (`t_i >= x_i^{3/2}`).
- `regression_regularized.py` — ridge / LASSO / 3-2 penalised regression with
  conic penalties.
- `lsq_l1_penalty.py` — positivity LS with a weighted L1 penalty (rotated cone +
  one cone per asset).
- `portfolio_mgmt.py` — five conic portfolio models (min-variance, tracking,
  max-return, 130/30 leverage, robust).
- `big_portfolio.py` — the "BIG portfolio optimisation" pipeline on real market
  moments: min-risk and max-return **SOCPs** plus a correlation-stress **SDP**;
  checks the case-study values (`vol0`, multiplier, stressed vol, …).

### Books — "Optimization Methods in Finance" (Cornuejols & Tutuncu)

Each checks the book's published optimum (or, where the book gives none, an
independent recomputation).

- `omf_ex21` — bond-portfolio LP: `(50,50)`, `Z=350`.
- `omf_ex25` — duality LP: `(3,2)`, `12`.
- `omf_ex217` — simplex LP: `(3,0,0)`, `12`.
- `omf_duality` — first duality example: `(5,2)`, `-7`.
- `omf_qp_grg` — GRG quadratic: `(1.5,1.5)`, `f=2`.
- `omf_ex312` — dedication (cash-flow matching) LP; **no published optimum**, the
  cash-flow recurrence is recomputed.
- `omf_dedication` — the dedication LP with published **shadow prices and reduced
  costs** (cost `93944.5`).
- `omf_ex410` — asset/liability: three LPs (expected profit, every-scenario
  profit, riskless profit).
- `omf_ex411` — currency arbitrage LP; the arbitrage cap and the cycle products.
- `omf_financing` — short-term financing LP; checks the book's shadow prices.
- `omf_workforce` — workforce planning LP; schedule, shadow prices and the
  reduced cost of the unused shift.
- `omf_appendix_d` — revised-simplex worked LP: `x1=3, x2=5, x6=3`, `Z=13`.
- `omf_bb` — branch-and-bound MILPs: `(1,3)->4` and `(2,1)->7`.
- `omf_capital` — capital budgeting (0-1) MIP: `(0,1,1,1)`, `21000`.
- `omf_ex118` — **Gomory mixed-integer cut**: reproduces the LP relaxation, the
  `k=3` cut `3x1+4x2<=12`, and the `40` at `(4,0)` for the cut LP and the MIP.
- `omf_ex2001` — tracking-error SOCP: `5.662` and a continuum case `5.0`.
- `omf_ex207` — relative robust portfolio SOCP: `(0.5,0.5,0)`, `t=0.662`.
- `omf_mvo` — Markowitz frontier QP; Table 8.3.
- `omf_bl` — Black-Litterman; posterior mean and Table 8.5.
- `omf_nearestcorr` — nearest correlation matrix (**SDP**), checked against
  Higham's projection.

### Other books and MOSEK ports

- `lo1` — MOSEK `lo1` LP: `(0.5,0.5)`, `2.5`.
- `qo1` — MOSEK `qo1` QP: `(0.5,0.5)`, `0.5`.
- `mi1` — MOSEK `mi1` MIP: `x1=5`, `5`.
- `intlo_simplex` — Nemirovski, *Introduction to Linear Optimization*: simplex
  LP `(6,12,6)`, `102`.
- `boyd_qp_box` — Boyd & Vandenberghe, *Convex Optimization*, Ex 4.3: box QP
  `(1,1/2,-1)`, `f=-21.625`.
- `hdb_kkt` — *Handbook of Algorithms for Physical Design Automation*: duality LP
  `(0,1)`, `p*=1`, `|nu|=1/2`.
- `socp_robust.py` — least-squares regression as an SOCP (a larger, scaling
  example): residual equals the optimal `t` and `A'r ≈ 0`.

### Ports of the remaining C samples

Every sample in [`examples/c/`](../examples/c/) now has a Python port with the
**same output** as the C program: the same text, numbers and exit code. Lines with a
wall-clock time differ only in the time. The pseudo-random instances use the same
generators. Three samples differ in the last printed digit of a rounding-noise
residual from the C build used as reference: `sos_m1_certificate`, `secure_ee_sdma`
and `logcontrast_lasso`. That build lets clang fuse multiply-adds (FMA). The C
programs compiled with `-ffp-contract=off` print exactly the Python values.

MOSEK API ports:

- `hello`, `simple`, `lo2`, `parallel`, `concurrent1` — basic LPs, several tasks,
  the four optimizers agreeing on `250/3`.
- `pinfeas`, `mioinfeas1`, `feasrepairex1` — infeasibility verdicts and the
  minimum-violation repair.
- `reoptimization`, `solvelinear`, `sensitivity`, `solutionquality`,
  `parameters` — re-solve after a change, a linear system as an LP, cost ranges,
  violations, parameter round-trip.
- `solvebasis` — write the optimal basis, read it into a new task, `solvebasis`.
- `response`, `callback` — log stream and progress callbacks (ctypes `CFUNCTYPE`).
- `mil1`, `mil2`, `milo1`, `mioinitsol`, `mico1`, `sched` — MIPs (pure,
  semi-continuous/semi-integer, warm start, mixed-integer conic, scheduling).
- `sos1`, `sos2` — SOS1 / SOS2 constraints.
- `cqo1`, `qcqo1`, `ceo1`, `pow1`, `acc1`, `acc1_max`, `djc1`, `gp1`,
  `logistic` — quadratic, rotated, exponential and power cones, affine conic and
  disjunctive constraints, geometric programming, sum-of-exp regression.
- `portfolio_1` … `portfolio_6` — Markowitz basic, frontier, market impact,
  transaction costs, cardinality, factor model.
- `sdo1`, `sdo2`, `sdo_lmi`, `dual_sdo_l1`, `nearestcorrelation`,
  `sparsecholesky` — semidefinite programs.

Books:

- `intlo_bigm`, `intlo_transport` — Nemirovski: Big-M LP, transportation
  counterexample.
- `boyd_lp_unique`, `boyd_qp_box2`, `boyd_l1`, `boyd_detector`, `boyd_corr_sdp` —
  Boyd & Vandenberghe exercises (unique LP optimum, box QP, l1 approximation,
  minimax detector, correlation bounds SDP).
- `pca_alloc`, `pca_cashflow` — *Portfolio Construction and Analytics*.
- `oa_feed` — *Optimization Algorithms*: animal feed mix.
- `npo_lad`, `npo_minimax` — *Nonlinear Parameter Optimization Using R*: L1 and
  L-infinity regression.
- `lmco_lovasz` — Ben-Tal & Nemirovski: Lovasz capacity of C5, `sqrt(5)`.

Applications and scaling tests:

- LP / MILP: `transport`, `assignment`, `network_flow`, `lp_large`,
  `lp_ineq_large`, `cvx_regression`, `diet`, `max_flow_min_cut`, `lp_mincut`,
  `wasserstein`, `dist_robust`, `surface_cycles`, `knapsack`, `facility`,
  `unit_commitment`, `exact_cover`, `nesting_dotted_board`, `poly_date`, `sudoku`,
  `tsp`, `lpt`, `sinr_balancing`.
- QP: `qp_sparse`, `eco_driving_qp`, `mpc_linear`, `barqcqp`.
- SOCP: `equilibrium`, `min_enclosing_ball`, `min_circle`, `welzl`,
  `facility_location`, `pwl_convex`, `total_variation`, `primal_svm`,
  `secure_beamforming`, `limit_analysis_dp`, `steering_robustness`.
- Exponential / power cones and GPs: `logistic_large`, `mle_density`,
  `hard_uncertain`, `f_sparc`, `kmeans`, `rank_one`, `logcontrast_lasso`,
  `gp_toolbox`, `gate_sizing_gp`, `transformer_design`, `max_volume_cuboid`,
  `secure_ee_sdma`.
- SDP: `maxcut_sdp`, `binary_quadratic`, `lovasz_theta`, `quantum_separability`,
  `truss_design`, `filter_design`, `lmi_lyapunov`, `bounded_real_lmi`,
  `lownerjohn_ellipsoid`, `qcqp_sdo_relaxation`, `lyapunov_roa`.
- `sos_m1_certificate` — re-verifies an exact rational SOS certificate (pure
  arithmetic, no solve).

## See also

- [`examples/lp/`](../examples/lp/README.md) — one model file per supported
  format (LP / OPF / CBF / MPS), solved with the `primal` command-line tool.
- [`examples/c/`](../examples/c/) — 171 C samples, the source of many of these
  Python ports.
