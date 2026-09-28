# PrimalSolver

A convex optimization solver written from scratch in portable C99. No external
dependencies — `libc` and `libm`, nothing else.

It solves **LP, MILP, QP, QCQP, SOCP, SDP, exponential and power cones, and
mixed-integer conic problems**, and it hands back the primal point, the dual
point and a status you can check rather than trust.

[![Sponsor](https://img.shields.io/badge/Sponsor-❤-ea4aaa)](https://github.com/sponsors/c-vision)

## Where it fits

The niche is the **combination**, not any single capability: among solvers with
a permissive licence, none other does semidefinite programming *and*
exponential/power cones *and* mixed-integer variables with an interior-point
method, in C, with no dependencies and no licence server.

It is not a replacement for HiGHS on large LP/MIP, and it is slower than
Clarabel on SOCP and SCS on SDP. [How it compares](#how-it-compares) says by
how much.

## What it solves

| Class | Route |
|---|---|
| LP, QP | presolve → simplex or Mehrotra interior point (dense, or sparse normal equations) |
| MILP, MIQP | branch & bound: CG / cover / Gomory cuts, strong branching, probing, bound tightening, a primal heuristic |
| SOCP | primal-dual conic IPM with Nesterov–Todd scaling |
| SDP | Nesterov–Todd IPM on PSD blocks |
| Exp / power cones | native IPM with a closed-form barrier, 4–5 orders of magnitude more accurate than the cut route |
| QCQP | quadratic rows encoded exactly into rotated quadratic cones |
| Disjunctions | ACC, DJC, SOS1/SOS2, semi-continuous and semi-integer variables |

Every class is mapped onto **one** internal conic representation rather than a
family of separate solvers, so behaviour and diagnostics are consistent across
classes. Each route has its own page under [Approfondimenti](#approfondimenti).

## Certificates

Every answer comes with the evidence: the primal and dual points, so the caller
can check stationarity, both feasibilities, complementarity and
`|pobj − dobj|`; a Farkas vector `y` with `A'y ≤ 0` and `Σ y_i b_i > 0` when
the model is infeasible; a recession direction `ρ` with `Aρ = 0`, `ρ ≥ 0`,
`c'ρ < 0` when it is unbounded.

A vector is published only after it has been **measured** in the form the solver
actually solved, and a `*_CER` status never appears without the vector that
proves it. Where no vector can exist the verdict is in `PRIMAL_getprosta`, the
status is `UNKNOWN`, and the getters refuse rather than hand over a zero vector.
[Details and the exceptions](docs/certificates.md).

## Example

```c
#include <math.h>
#include "primal.h"

int main(void) {
    PRIMALenv_t env;  PRIMALtask_t task;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 0, 0, &task);
    PRIMAL_appendvars(task, 2);
    PRIMAL_appendcons(task, 2);

    for (int j = 0; j < 2; j++)                   /* x, y >= 0 */
        PRIMAL_putvarbound(task, j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putcj(task, 0, 2.0);                    /* min 2x + 3y */
    PRIMAL_putcj(task, 1, 3.0);
    PRIMAL_putobjsense(task, PRIMAL_OPTIMIZE_MINIMIZE);

    PRIMAL_putarow(task, 0, 2, (int[]){0,1}, (double[]){1, 1});   /* x + y >= 1 */
    PRIMAL_putconbound(task, 0, PRIMAL_BK_LO, 1.0, INFINITY);
    PRIMAL_putarow(task, 1, 2, (int[]){0,1}, (double[]){1,-1});   /* x - y <= 0 */
    PRIMAL_putconbound(task, 1, PRIMAL_BK_UP, -INFINITY, 0.0);

    if (PRIMAL_optimize(task) != PRIMAL_RES_OK) return 1;

    double x[2], obj;
    PRIMAL_getxx(task, PRIMAL_SOL_BAS, x);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_BAS, &obj);
    printf("x = %.6f, y = %.6f, obj = %.6f\n", x[0], x[1], obj);  /* 0.5 0.5 2.5 */

    PRIMAL_deletetask(&task);
    PRIMAL_deleteenv(&env);
    return 0;
}
```

171 runnable examples live under `samples/`, each checking its own result: 48
ports of public MOSEK examples with identical optimal values, 63 larger
deterministic models, 44 finance models, 16 book models.

## Build

```sh
make               # gcc -std=c99 -Wall -Wextra -pedantic -O2, zero warnings
make test          # reliability suite: 4891 checks
make run-samples   # the 171 examples
make clean         # remove out/
```

Everything the build produces goes to `out/` (gitignored). **Portability**: POSIX
C99 with `gcc` or `clang` on Linux, macOS and the BSDs. There is **no MSVC
project and no `nmake` path** — on Windows use MinGW
([how to build on Windows](docs/windows.md)), or compile the `LIBSRCS` list from
the Makefile with any C99 compiler. Clean under AddressSanitizer +
UndefinedBehaviorSanitizer on the suite and on all samples.

## How it compares

Capability, licence and accuracy class — **not** speed.

| | PrimalSolver | MOSEK | Gurobi | HiGHS | Clarabel | SCS |
|---|---|---|---|---|---|---|
| LP / QP | yes | yes | yes | yes | LP (QP via cones) | yes |
| SOCP | yes | yes | Gurobi only | — | yes | yes |
| **SDP** (PSD) | yes | yes | — | — | **no** | yes, first-order |
| **Exp / power** | yes, native | yes | — | — | yes | yes, first-order |
| MIP / MIQP | yes | yes | yes | yes | — | — |
| Method | interior point | interior point | simplex + IPM | simplex + IPM | interior point | **first-order** |
| Accuracy | ~1e-8 | ~1e-8 | ~1e-8 | ~1e-9 | ~1e-8 | ~1e-4 |
| Dependencies | **libm** | — | — | C++ | Rust | C |
| Licence | Apache-2.0 | commercial | commercial | MIT | Apache-2.0 | MIT |

The rows that decide it:

- **Clarabel** is the closest open conic IPM and is **~30× faster** here on
  SOCP (0.0004 s against 0.013 s at 200 cones) — but it has no PSD cone, so it
  cannot do half of what this does.
- **SCS** does PSD and exponential cones, but is first-order: it agrees with
  this solver's objective to about `1e-5`, where the IPM reaches `1e-8`.
- **HiGHS** is faster on large LP/MILP — 0.012 s against 0.091 s on a 400×200 LP.
  The crossover sits around 50–100 constraints; below that this solver wins on
  fixed overhead.
- **SDP with exponential cones and integer variables, in one dependency-free
  library**, is what nothing else permissively licensed covers. CVXOPT / SDPA /
  Sedumi have SDP but are GPL.

Needing raw LP/MIP throughput? Use HiGHS. Needing an SDP *and* an exponential
cone *and* an integer variable, in C, without linking anything? This is it.

## Benchmark

Reproduce with `make bench` and a Python that has numpy/scipy plus the optional
references (`clarabel`, `scs`, `highspy`, `pyscipopt`; a solver that is absent
shows `N/A`). `bench/bench_all.py` prints this table and cross-checks every
objective against the analytic value; it exits non-zero on a mismatch. LP/QP/
MILP are the generated MPS instances, SOCP/SDP are the closed-form families of
`bench/conic_bench.c` and `bench/sdp_sweep.c`. The SDP sweep reaches **d = 22**;
`d = 20` is the one block the interior point does not close on this family yet
(`rc = 1007`, listed, not hidden).

<div style="font-size: 0.9em">

| instance | class | vars x cons | nnz | Primal (s) | HiGHS (s) | Clarabel (s) | SCS (s) | SCIP (s) | obj |
|---|---|---|---|---|---|---|---|---|---|
| lp_50x25 | lp | 50 x 25 | 171 | 0.0010 | 0.0015 | 0.0002 | N/A | N/A | 64.1886 |
| lp_100x50 | lp | 100 x 50 | 576 | 0.0034 | 0.0017 | 0.0007 | N/A | N/A | 121.946 |
| lp_200x100 | lp | 200 x 100 | 2120 | 0.0145 | 0.0040 | 0.0020 | N/A | N/A | 284.367 |
| lp_400x200 | lp | 400 x 200 | 8026 | 0.0911 | 0.0124 | 0.0114 | N/A | N/A | 546.481 |
| qp_50x25 | qp | 50 x 25 | 1445 | 0.0016 | N/A | 0.0004 | N/A | N/A | 12.8014 |
| qp_100x50 | qp | 100 x 50 | 5631 | 0.0391 | N/A | 0.0021 | N/A | N/A | 26.4052 |
| qp_200x100 | qp | 200 x 100 | 22212 | 0.2701 | N/A | 0.0073 | N/A | N/A | 55.9929 |
| milp_40x20 | milp | 40 x 20 | 139 | 0.2995 | N/A | N/A | N/A | 0.1434 | 143.955 |
| milp_60x30 | milp | 60 x 30 | 260 | 1.0117 | N/A | N/A | N/A | 0.2583 | 200.846 |
| milp_80x40 | milp | 80 x 40 | 460 | 3.0614 | N/A | N/A | N/A | 1.0175 | 277.019 |
| socp_40 | socp | 40 x 1 | - | 0.0004 | N/A | 0.0001 | 0.0003 | N/A | 0.707107 |
| socp_120 | socp | 120 x 1 | - | 0.0042 | N/A | 0.0002 | 0.0007 | N/A | 0.707107 |
| socp_200 | socp | 200 x 1 | - | 0.0137 | N/A | 0.0004 | 0.0011 | N/A | 0.707107 |
| sdp_4 | sdp | 4 x 4 | - | 0.0002 | N/A | 0.0002 | 0.0002 | N/A | -9.83233 |
| sdp_5 | sdp | 5 x 5 | - | 0.0004 | N/A | 0.0001 | 0.0002 | N/A | -4.79948 |
| sdp_6 | sdp | 6 x 6 | - | 0.0006 | N/A | 0.0002 | 0.0002 | N/A | -10.2438 |
| sdp_7 | sdp | 7 x 7 | - | 0.0019 | N/A | 0.0002 | 0.0004 | N/A | -8.18613 |
| sdp_8 | sdp | 8 x 8 | - | 0.0014 | N/A | 0.0004 | 0.0005 | N/A | -22.165 |
| sdp_9 | sdp | 9 x 9 | - | 0.0019 | N/A | 0.0005 | 0.0004 | N/A | -17.0048 |
| sdp_10 | sdp | 10 x 10 | - | 0.0026 | N/A | 0.0007 | 0.0006 | N/A | -31.7195 |
| sdp_11 | sdp | 11 x 11 | - | 0.0035 | N/A | 0.0010 | 0.0016 | N/A | -20.5753 |
| sdp_12 | sdp | 12 x 12 | - | 0.0057 | N/A | 0.0015 | 0.0009 | N/A | -44.5174 |
| sdp_13 | sdp | 13 x 13 | - | 0.0050 | N/A | 0.0019 | 0.0010 | N/A | -38.7212 |
| sdp_14 | sdp | 14 x 14 | - | 0.0060 | N/A | 0.0019 | 0.0010 | N/A | -74.2121 |
| sdp_15 | sdp | 15 x 15 | - | 0.0091 | N/A | 0.0021 | 0.0013 | N/A | -51.0706 |
| sdp_16 | sdp | 16 x 16 | - | 0.0217 | N/A | 0.0031 | 0.0014 | N/A | -69.7722 |
| sdp_17 | sdp | 17 x 17 | - | 0.0167 | N/A | 0.0038 | 0.0017 | N/A | -38.8839 |
| sdp_18 | sdp | 18 x 18 | - | 0.0224 | N/A | 0.0042 | 0.0020 | N/A | -78.2696 |
| sdp_19 | sdp | 19 x 19 | - | 0.0655 | N/A | 0.0045 | 0.0020 | N/A | -123.432 |
| sdp_20 | sdp | 20 x 20 | - | N/A (rc=1007) | N/A | 0.0067 | 0.0025 | N/A | -140.192 |
| sdp_21 | sdp | 21 x 21 | - | 0.0341 | N/A | 0.0081 | 0.0037 | N/A | -93.0904 |
| sdp_22 | sdp | 22 x 22 | - | 0.2170 | N/A | 0.0097 | 0.0081 | N/A | -149.613 |

</div>

## Status

**4891 checks, 0 failures** at `-O0`, `-O1`, `-O2`, `-O3`. **171/171** examples
exit 0. Warning-free. ASan + UBSan clean.

## TODO

Open algorithmic work, largest first:

- **Basis warm start.** No real crash basis; `solvebasis` re-solves from scratch
  when the basis handed in does not measure, instead of re-optimizing *from* it.
- **Sparse KKT in the conic path.** The unified conic system is assembled
  densely, which caps problem size. The sparse LDLᵀ with 2×2 pivots exists and is
  validated; it is not wired in yet.
- **Symmetric KKT in the SOCP route**, so the sparse symmetric factorization can
  replace the sparse LU there.
- **Large-scale conic performance.** The gap to Clarabel is architectural — an
  interior-point method without tuned sparse kernels.
- **MIP.** No conflict analysis, no pseudo-cost propagation, no Gomory
  mixed-integer cuts.
- **Parameter coverage.** 31 declared, all but one wired to behaviour
  (`PRIMAL_IPAR_LOG` is accepted and read back, but nothing consults it).
- **I/O.** No TASK or PTF, no compression, no quadratic LP bracket syntax.

## Approfondimenti

One page per algorithm: what it is, when the solver picks it, what it was
measured at, where it stops. Start at [docs/README.md](docs/README.md).

**Foundations** — [standard form](docs/standard-form.md) ·
[scaling](docs/scaling.md) · [presolve](docs/presolve.md) ·
[linear algebra](docs/linear-algebra.md)

**Routes** — [simplex](docs/simplex.md) ·
[interior point, LP/QP](docs/interior-point.md) ·
[conic IPM, SOCP](docs/conic-socp.md) · [SDP](docs/sdp.md) ·
[exp & power cones](docs/exp-power-cones.md) ·
[tangent cuts](docs/tangent-cuts.md) · [QCQP encoding](docs/qcqp-encoding.md) ·
[branch & bound](docs/branch-and-bound.md) ·
[disjunctive constraints](docs/disjunctive-constraints.md)

**Contract** — [certificates](docs/certificates.md) ·
[API guide](docs/api.md) · [I/O formats](docs/io-formats.md)

## Diagnostics

`GMB_DBG=1` prints to **stderr** the route taken with its measured termination
triple, the worst cone or bar violation in the delivered point, each round of
the cut loop, and the MIP search's decisions:
`./out/samples/logistic_large 2>&1 | grep -E 'route|cones|mip'`

## License

Apache-2.0. All code is original; see [LICENSE](LICENSE). The 48 example ports
re-implement models published in MOSEK's public documentation. PrimalSolver is
not affiliated with, or endorsed by, MOSEK.
