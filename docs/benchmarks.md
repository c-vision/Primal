# Benchmark methodology

The README table covers 32 instances. The SOCP and SDP rows use
medians from five runs at `bd53531` on 2026-11-07. The MILP rows use
medians from ten interleaved runs at `b452e46` on 2026-11-08. The objective
checks passed on every run. Bold marks the lowest median in each row.
The table uses six decimal places in seconds to show sub-millisecond times.
The exponential-cone corrector measurements appear separately in the README.

The LP rows use 18 interleaved measurements from 2026-11-10. Primal
eliminates eligible variable bound equations from the interior-point
linear system, reconstructs their multipliers and checks the original
system residual. LP400 factors a system of order 200 after eliminating
400 rows. Primal and Clarabel ran at one thread. Clarabel and HiGHS used
feasibility settings of `1e-9`. The three solvers passed the `1e-8`
checks for objective agreement, feasibility, stationarity and relative
gap. Raw timings and binary hashes are in `out/f143/final-pairs.json`.

The QP rows use 18 interleaved measurements from 2026-11-11. The dense
interior-point solver eliminates eligible slack columns and checks the
reconstructed direction against the original equations. The sparse
solver can factor a dense Hessian block separately from diagonal slack
columns. Primal and Clarabel ran at one thread and passed the same
`1e-8` accuracy checks as the LP comparison. Raw timings and binary
hashes are in `out/f147/final-pairs.json`.

## Environment

| Component | Measured configuration |
|---|---|
| Machine | Apple M5 Max, arm64, macOS 27.2 |
| C compiler | Apple clang 21.0.0, C99 `-O2 -Wall -Wextra -pedantic` |
| NumPy / SciPy | 2.5.3 / 1.18.1 |
| HiGHS used by SciPy | 1.12.0 |
| HiGHS used by highspy for MILP | 1.15.1 |
| Clarabel | 0.11.1 |
| SCS | 3.3.1 |
| SCIP / PySCIPOpt | 10.0 / 6.2.1 |

The SOCP and SDP comparisons use the harness defaults. The MILP
comparison uses one thread per solver and the same MPS input. It checks
optimal status and objective agreement, plus Primal's feasibility and
integrality to `1e-8`.

## What the timer includes

- Primal: the internal `PRIMAL_optimize` call, including conversion to its
  internal representation; model construction or MPS reading is excluded.
- Clarabel: `solve()`, after the solver constructor. Setup is excluded.
- SCS: the `scs.solve(...)` call, including setup.
- HiGHS: SciPy's `linprog(method="highs")` for LP, including interface
  overhead. For MILP, `highspy` times `run()` after model loading.
- SCIP: optimization, with reading/building the model outside the timer.

The bold values identify the lowest medians within those timing boundaries.
Close medians can change order across runs.
`N/A` denotes a missing measurement for that solver and instance.

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

In the accuracy audit of the SDP sweep before the fixes, d=20 returned
`rc=0` with objective -140.19172 versus analytic reference -140.1919, about
1.27e-6 relative error. The isolated `06f767b` baseline returns the same point:
this is not a correction introduced by the new PEXP route. Orders 14, 21 and
22 also had relative objective errors above 1e-7 (about 1.06e-7, 2.23e-7 and
3.98e-7 respectively, at the sweep's printed precision). Their timings must
not be interpreted as common 1e-8-accuracy results. The table's SOCP/SDP
objective column reports the analytic reference.

The subsequent SDP fix enforces symmetric directions, recovers the dual
step from the original dual equations and checks the primal direction after
Schur reconstruction. A bounded augmented solve repairs inaccurate steps.
T301 verifies orders 14, 20, 21 and 22 at `1e-8` with the default near factor
and with the factor set to one. The measured gap for order 20 fell from
`7.13e-7` to `7.97e-14`.

A follow-up termination check ends pure SDP recovery once the absolute
residuals, both cones and the gap meet the requested accuracy. In two series
of nine alternating pairs against the accuracy fix at `adb7f74`, order 20
went from 247.0 to 56.0 ms and from 235.8 to 52.6 ms. Its public relative
gap is approximately `3.13e-9`. Orders 14, 21 and 22 also pass T301 and took
6.5–7.0 ms, 37.6–38.6 ms and 46.3–47.8 ms respectively.

## Reproduce

To rerun the unified harness, build the benchmark binaries and keep the
machine otherwise idle.

```sh
make bench
python3 tools/bench_repeat.py --runs 5 --out out/benchmark-median.md
python3 tools/bench_repeat.py --check-rows-only
```

Use a Python environment containing NumPy, SciPy, Clarabel, SCS and PySCIPOpt.
The harness and deterministic instances live under `bench/`; the aggregator
lives in `tools/bench_repeat.py`. On the development machine these dependencies
were supplied by the existing `env-bench` environment.

The November 7 snapshot used a fresh isolated output directory. Its raw
tables, objective check logs, aggregated table and binary hashes were
retained under `out/readme-20261107/`. The November 5 snapshot is retained
under `out/readme-20261105/`.

The single-thread MILP comparison uses the same MPS fixtures with HiGHS
and SCIP. Its comparison script, raw timings and binary hash are retained
under `out/milp-20261108/` on the benchmark machine.

The November 7 benchmark build was warning-free. For the earlier PEXP
change, 928 focused checks passed with default, forced sparse, forced dense
and legacy routes.
