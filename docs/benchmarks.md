# Benchmark methodology

The README table dated 2026-11-05 is a fresh snapshot of the current working
tree, based on `06f767b` plus the mixed SOC/PEXP corrector. It contains 32
instances: four LP, three QP, three MILP, three SOCP and nineteen SDP sizes.
Every timing cell is the median of five complete runs. The harness objective
checks passed on all five runs and the printed objective column was stable.
The canonical table does not contain exponential-cone models; the paired
corrector measurements are reported separately below it.

## Environment

| Component | Measured configuration |
|---|---|
| Machine | Apple M5 Max, arm64, macOS 27.2 |
| C compiler | Apple clang 21.0.0, C99 `-O2 -Wall -Wextra -pedantic` |
| NumPy / SciPy | 2.5.3 / 1.18.1 |
| HiGHS used by SciPy | 1.12.0 |
| Clarabel | 0.11.1 |
| SCS | 3.3.1 |
| SCIP / PySCIPOpt | 10.0 / 6.2.1 |

These are CPU runs with each solver's default settings. Thread counts and
tolerances were not normalized across solvers. The separately installed
`highspy` package is not the HiGHS interface used for this table: LP reference
times come from SciPy's `linprog(method="highs")`.

## What the timer includes

- Primal: the internal `PRIMAL_optimize` call, including conversion to its
  internal representation; model construction or MPS reading is excluded.
- Clarabel: `solve()`, after the solver constructor. Setup is excluded.
- SCS: the `scs.solve(...)` call, including setup.
- HiGHS: the SciPy `linprog` call and its interface overhead.
- SCIP: optimization, with reading/building the model outside the timer.

Consequently this is a comparison of the existing harness boundaries, not an
equal end-to-end latency test. Sub-millisecond cells are rounded to four
decimal places in seconds and should not support fine percentage claims.
`N/A` is a missing measurement for this harness/interface; it is not a solver
capability statement. In particular, HiGHS is not measured on the MILP rows.

For SOCP, the family parameter n describes n free x coordinates and n/2 cone
epigraph variables, so the model has 3n/2 scalar variables and one explicit
equality. The README corrects the old harness dimension label accordingly.
For SDP, the d×d size describes the PSD matrix, not d scalar variables.

## Accuracy limits

The LP/QP/MILP objective cross-check allows a difference of
`1e-5*(1+abs(objective))`. SOCP uses an analytic objective check of `1e-4`
absolute; SDP uses `1e-4*(1+abs(analytic objective))`. These checks detect
large disagreements. They do not establish a common KKT tolerance, cone
feasibility or certificate quality across all timed results.

In the independent accuracy audit of the current SDP sweep, d=20 returned
`rc=0` with objective -140.19172 versus analytic reference -140.1919, about
1.27e-6 relative error. The isolated `06f767b` baseline returns the same point:
this is not a correction introduced by the new PEXP route. Orders 14, 21 and
22 also had relative objective errors above 1e-7 (about 1.06e-7, 2.23e-7 and
3.98e-7 respectively, at the sweep's printed precision). Their timings must
not be interpreted as common 1e-8-accuracy results. The table's SOCP/SDP
objective column reports the analytic reference.

## Reproduce

Build all benchmark binaries before timing; keep the machine otherwise idle:

```sh
make bench
python3 tools/bench_repeat.py --runs 5 --out out/benchmark-median.md
python3 tools/bench_repeat.py --check-rows-only
```

Use a Python environment containing NumPy, SciPy, Clarabel, SCS and PySCIPOpt.
The harness and deterministic instances live under `bench/`; the aggregator
lives in `tools/bench_repeat.py`. On the development machine these dependencies
were supplied by the existing `env-bench` environment.

For this snapshot, `make clean && make` was scoped to a fresh isolated OUT
directory, preserving previous measurements. The canonical harness binary
paths were redirected to that directory; five raw tables, their objective
check logs, the aggregated table, versions and binary hashes were retained
under `out/readme-20261105/`. No frozen baseline directory was rebuilt.

The clean release build was warning-free. The numerical change's focused
group passed 928 checks with default, forced sparse, forced dense and legacy
routes. This is not a new full-suite or sanitizer result.
