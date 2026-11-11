# Benchmark methodology

The README reports 32 instances measured on 2026-11-11. Each row
shows the median of interleaved runs and highlights the fastest solver.
The mixed SOC/PEXP comparison reports two medians from 21 interleaved
measurements each. The refreshed mixed-cone timings and binary hashes
are in `out/f157/final-pairs.json` on the benchmark machine.

| Family | Measurements per solver and instance | Accuracy checks |
|---|---:|---|
| LP and QP | 18 | Objective agreement, primal feasibility, stationarity and relative gap within `1e-8` |
| MILP | 10 | Optimal status and objective agreement, with Primal feasibility and integrality within `1e-8` |
| SOCP | 62 | Analytic objective, primal feasibility, stationarity and relative gap within `1e-7` |
| SDP | 9 | Optimal status and objective error within `1e-4 * (1 + abs(analytic objective))` |
| Mixed SOC/PEXP | 42 | Objective agreement, feasibility, stationarity and relative gap within `1e-8` |

Primal and Clarabel ran at one thread. HiGHS and SCIP used one thread
for MILP and read the same MPS fixtures as Primal. The SOCP SCS runs
used `eps_abs=eps_rel=1e-8`. SDP references used their default
tolerances.

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

The SDP timing comparison checks the analytic objective. The SOCP and
SDP objective columns show analytic reference values. See the table
above for the checks applied to each family.

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

The 2026-11-11 snapshot used an isolated build. It was refreshed the same day
on a build of the tree that carries the presolve and dual-ray corrections; the
comparison scripts, raw timings, accuracy checks and binary hashes are retained
under `out/readme-20261111r/` on the benchmark machine (the earlier measurement
is in `out/readme-20261111/`). The refresh ran on a machine with other
processes (load average 10 to 15 on 18 cores), with the solvers interleaved so
that every instance saw the same conditions. Two MILP rows are within noise of
HiGHS: on `milp_40x20` the first full run gave 53.2 ms for Primal and 48.4 ms for
HiGHS, and a later repeat of the MILP family (the one in the table) 52.3 ms and
53.2 ms; on `milp_60x30` the two runs gave 108.1 and 104.9 ms against 97.5 and
105.8 ms. The unified harness
provides a quick cross-check. The snapshot scripts add the family-specific
settings, timing precision and accuracy checks described above.
