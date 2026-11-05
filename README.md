# PrimalSolver

A convex optimization solver written from scratch in portable C99. No external
dependencies — `libc` and `libm`, nothing else.

It solves **LP, MILP, QP, QCQP, SOCP, SDP, exponential and power cones, and
mixed-integer conic problems**, and it hands back the primal point, the dual
point and a status you can check rather than trust.

[![Sponsor](https://img.shields.io/badge/Sponsor-❤-ea4aaa)](https://github.com/sponsors/c-vision)
[![PyPI](https://img.shields.io/pypi/v/primalsolver)](https://pypi.org/project/primalsolver/)

## What it solves

- **LP, QP** — presolve, then simplex or a Mehrotra interior point.
- **MILP, MIQP** — branch & bound with cutting planes, strong branching,
  probing and primal heuristics.
- **SOCP** — primal-dual conic interior point with Nesterov–Todd scaling.
- **SDP** — Nesterov–Todd interior point on PSD blocks.
- **Exponential and power cones** — native interior point with closed-form
  barriers.
- **QCQP** — quadratic constraints encoded exactly as rotated quadratic cones.
- **Disjunctions** — ACC, DJC, SOS1/SOS2, semi-continuous and semi-integer
  variables.

Every class is mapped onto one internal conic representation, so behaviour and
diagnostics are the same across classes.

## Certificates

Every answer comes with its evidence: the primal and dual points, so you can
check feasibility, complementarity and the duality gap yourself; a Farkas
vector when the model is infeasible; a recession direction when it is
unbounded. A certificate status is never reported without the vector that
proves it. [Details](docs/certificates.md).

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

171 runnable examples live under `examples/c/`, each checking its own result.

## Command line

`make` also builds `out/primal`, a command-line solver that reads a model file,
solves it and prints the solution. The algorithm is chosen from what the model
contains, so the same command solves every problem class.

```
Maximize
 obj: 3 x0 + 2 x1
Subject To
 c0: x0 + x1 <= 4
 c1: x0 + 3 x1 <= 6
Bounds
 x0 <= 3
 x1 <= 3
End
```

```sh
make
./out/primal diet.lp
```

It reads MPS, CPLEX LP, OPF and CBF files (detected automatically); run
`./out/primal --help` for the options. Ready-to-run inputs for every problem
class are in [`examples/lp/`](examples/lp/README.md).

## Python

On PyPI as [**`primalsolver`**](https://pypi.org/project/primalsolver/), no
compiler needed:

```sh
pip install primalsolver
python -c "import primalsolver; print(primalsolver.version())"
```

Wheels for macOS and Linux are on PyPI. Examples are in
[`examples/python/`](examples/python/README.md).

## Julia

The package **`Primal`** lives in [`julia/`](julia/) and is also a
[JuMP](https://jump.dev) solver. To install it next to JuMP (Julia 1.10 or
later and a C compiler; the first `using Primal` compiles the C library):

```sh
git clone https://github.com/c-vision/Primal.git
julia -e 'using Pkg; Pkg.develop(path="Primal/julia"); Pkg.add("JuMP")'
```

```julia
using JuMP, Primal

model = Model(Primal.Optimizer)
@variable(model, 0 <= x <= 3.5)
@variable(model, y >= 0)
@constraint(model, c, x + y <= 4)
@objective(model, Max, 3x + 2y)
optimize!(model)
value(x), value(y), objective_value(model), shadow_price(c)   # (3.5, 0.5, 11.5, 2.0)
```

It accepts linear and convex quadratic objectives, linear constraints, binary
and integer variables, and the second-order, rotated second-order, exponential
and power cones; JuMP's bridges reformulate the rest. Solver parameters are set
with `set_attribute(model, "PRIMAL_IPAR_INTPNT_MAX_ITERATIONS", 500)` (any
`PRIMAL_IPAR_*` / `PRIMAL_DPAR_*` of `primal.h`). Examples are in
[`examples/julia/`](examples/julia/).

## Build

```sh
make               # library, command line and test suite
make test          # test suite
make run-samples   # run the 171 examples
make clean         # remove out/
```

Everything the build produces goes to `out/`. It builds with `gcc` or `clang` on
Linux, macOS and the BSDs; on Windows use MinGW
([how to build on Windows](docs/windows.md)).

## How it compares

| | PrimalSolver | MOSEK | Gurobi | HiGHS | Clarabel | SCS |
|---|---|---|---|---|---|---|
| LP / QP | yes | yes | yes | yes | yes | yes |
| SOCP | yes | yes | yes | — | yes | yes |
| SDP | yes | yes | — | — | yes | yes, first-order |
| Exp / power cones | yes | yes | — | — | yes | yes, first-order |
| MIP / MIQP | yes | yes | yes | yes | — | — |
| Method | interior point | interior point | simplex + IPM | simplex + IPM | interior point | first-order |
| Typical accuracy | ~1e-8 | ~1e-8 | ~1e-8 | ~1e-9 | ~1e-8 | ~1e-4 |
| Dependencies | libm | — | — | C++ | Rust | C |
| Licence | Apache-2.0 | commercial | commercial | MIT | Apache-2.0 | MIT |

PrimalSolver is not a replacement for HiGHS on large LP/MIP, and Clarabel is
faster on most conic models. What it offers is the combination: SDP,
exponential/power cones and integer variables in one permissively licensed C
library with no dependencies.

The mixed SOC/exponential route now includes a higher-order corrector. Local
paired measurements show 24–48% less solve time on three exponential
regression sizes, including 33–35% on the 40×8 case. The focused checks pass
928/928; this is not a full-suite result. Clarabel remains substantially faster
on that case. See [measurements, scope and validation](docs/pexp-higher-correction.md).

## Benchmarks

Measured on 2026-11-05, Apple M5 Max (arm64), macOS 27.2, Apple clang 21,
C99 `-O2`. Seconds, median of **five complete runs**; printed objectives were
stable across all runs and the harness objective checks passed.

<div>

| instance | class | vars x cons | nnz | Primal (s) | HiGHS (s) | Clarabel (s) | SCS (s) | SCIP (s) | obj |
|---|---|---|---|---|---|---|---|---|---|
| lp_50x25 | lp | 50 x 25 | 171 | 0.0016 | 0.0022 | 0.0002 | N/A | N/A | 64.1886 |
| lp_100x50 | lp | 100 x 50 | 576 | 0.0021 | 0.0024 | 0.0010 | N/A | N/A | 121.946 |
| lp_200x100 | lp | 200 x 100 | 2120 | 0.0081 | 0.0055 | 0.0031 | N/A | N/A | 284.367 |
| lp_400x200 | lp | 400 x 200 | 8026 | 0.0448 | 0.0174 | 0.0199 | N/A | N/A | 546.481 |
| qp_50x25 | qp | 50 x 25 | 1445 | 0.0023 | N/A | 0.0006 | N/A | N/A | 12.8014 |
| qp_100x50 | qp | 100 x 50 | 5631 | 0.0105 | N/A | 0.0031 | N/A | N/A | 26.4052 |
| qp_200x100 | qp | 200 x 100 | 22212 | 0.0655 | N/A | 0.0125 | N/A | N/A | 55.9929 |
| milp_40x20 | milp | 40 x 20 | 139 | 0.3519 | N/A | N/A | N/A | 0.2031 | 143.955 |
| milp_60x30 | milp | 60 x 30 | 260 | 1.3803 | N/A | N/A | N/A | 0.3652 | 200.846 |
| milp_80x40 | milp | 80 x 40 | 460 | 4.1655 | N/A | N/A | N/A | 1.4121 | 277.019 |
| socp_40 | socp | 60 x 1 | - | 0.0006 | N/A | 0.0001 | 0.0004 | N/A | 0.707107 |
| socp_120 | socp | 180 x 1 | - | 0.0003 | N/A | 0.0003 | 0.0008 | N/A | 0.707107 |
| socp_200 | socp | 300 x 1 | - | 0.0005 | N/A | 0.0006 | 0.0013 | N/A | 0.707107 |
| sdp_4 | sdp | 4 x 4 | - | 0.0003 | N/A | 0.0002 | 0.0003 | N/A | -9.83233 |
| sdp_5 | sdp | 5 x 5 | - | 0.0003 | N/A | 0.0002 | 0.0003 | N/A | -4.79948 |
| sdp_6 | sdp | 6 x 6 | - | 0.0006 | N/A | 0.0003 | 0.0003 | N/A | -10.2438 |
| sdp_7 | sdp | 7 x 7 | - | 0.0018 | N/A | 0.0004 | 0.0006 | N/A | -8.18613 |
| sdp_8 | sdp | 8 x 8 | - | 0.0010 | N/A | 0.0005 | 0.0007 | N/A | -22.165 |
| sdp_9 | sdp | 9 x 9 | - | 0.0017 | N/A | 0.0007 | 0.0006 | N/A | -17.0048 |
| sdp_10 | sdp | 10 x 10 | - | 0.0021 | N/A | 0.0010 | 0.0010 | N/A | -31.7195 |
| sdp_11 | sdp | 11 x 11 | - | 0.0025 | N/A | 0.0014 | 0.0023 | N/A | -20.5753 |
| sdp_12 | sdp | 12 x 12 | - | 0.0058 | N/A | 0.0023 | 0.0013 | N/A | -44.5174 |
| sdp_13 | sdp | 13 x 13 | - | 0.0057 | N/A | 0.0027 | 0.0015 | N/A | -38.7212 |
| sdp_14 | sdp | 14 x 14 | - | 0.0059 | N/A | 0.0027 | 0.0014 | N/A | -74.2121 |
| sdp_15 | sdp | 15 x 15 | - | 0.0080 | N/A | 0.0031 | 0.0017 | N/A | -51.0706 |
| sdp_16 | sdp | 16 x 16 | - | 0.0193 | N/A | 0.0044 | 0.0018 | N/A | -69.7722 |
| sdp_17 | sdp | 17 x 17 | - | 0.0160 | N/A | 0.0058 | 0.0023 | N/A | -38.8839 |
| sdp_18 | sdp | 18 x 18 | - | 0.0187 | N/A | 0.0060 | 0.0027 | N/A | -78.2696 |
| sdp_19 | sdp | 19 x 19 | - | 0.0521 | N/A | 0.0058 | 0.0026 | N/A | -123.432 |
| sdp_20 | sdp | 20 x 20 | - | 0.0414 | N/A | 0.0112 | 0.0029 | N/A | -140.192 |
| sdp_21 | sdp | 21 x 21 | - | 0.0322 | N/A | 0.0162 | 0.0053 | N/A | -93.0904 |
| sdp_22 | sdp | 22 x 22 | - | 0.0339 | N/A | 0.0166 | 0.0113 | N/A | -149.613 |

</div>

`N/A` means not measured by this harness, not necessarily unsupported. SDP
sizes denote the PSD matrix order. SOCP models have `n+n/2` scalar variables.
For SOCP/SDP, `obj` is the analytic reference value.
Primal timings cover `PRIMAL_optimize`; Clarabel timings exclude solver
construction, while the SCS calls include setup. Default tolerances differ,
and objective agreement alone does not establish equal solution accuracy.
In particular, `sdp_20` returns `rc=0` but has relative objective error about
`1.3e-6`, so its time is **not** a result at `1e-8` accuracy.

Reference versions, accuracy limits and reproduction:
[benchmark methodology](docs/benchmarks.md).

The mixed SOC/PEXP regression family is separate from the table above. Against
isolated sources at `06f767b`, two batches of nine alternating pairs gave:

| Model | Previous solve (ms) | Current solve (ms) | Reduction |
|---|---:|---:|---:|
| Regression 20×4 | 28.00 / 27.57 | 14.95 / 14.41 | 46.6–47.7% |
| Regression 40×8 | 91.47 / 94.60 | 61.27 / 61.69 | 33.0–34.8% |
| Regression 80×12 | 121.67 / 126.28 | 92.51 / 94.34 | 24.0–25.3% |

Both medians are shown. These gains apply to the mixed-cone corrector, not to
LP/MIP or SDP. Validation and remaining Clarabel gap are documented in
[the corrector measurements](docs/pexp-higher-correction.md).

## Documentation

One page per algorithm, starting at [docs/README.md](docs/README.md):
[simplex](docs/simplex.md) · [interior point](docs/interior-point.md) ·
[SOCP](docs/conic-socp.md) · [SDP](docs/sdp.md) ·
[exp & power cones](docs/exp-power-cones.md) ·
[branch & bound](docs/branch-and-bound.md) ·
[certificates](docs/certificates.md) · [API guide](docs/api.md) ·
[I/O formats](docs/io-formats.md)

## License

Apache-2.0. All code is original; see [LICENSE](LICENSE). The examples that port
models from MOSEK's public documentation re-implement them. PrimalSolver is not
affiliated with, or endorsed by, MOSEK.
