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
- **SDP** — Nesterov–Todd interior point on PSD blocks. For a single block
  with fixed diagonal and compatible objective signs, Primal constructs
  the primal and dual solution directly.
- **Exponential and power cones** — native interior point with closed-form
  barriers. Mixed SOC and primal exponential models use a higher-order
  corrector and select a sparse KKT solve when the system is sparse.
- **QCQP** — quadratic constraints encoded exactly as rotated quadratic cones.
- **Modeling** — affine conic constraints (ACC), disjunctive constraints
  (DJC), SOS1/SOS2, semi-continuous and semi-integer variables.

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
| Continuous methods | simplex + IPM | interior point | simplex + IPM | simplex + IPM | interior point | first-order |
| Typical accuracy | ~1e-8 | ~1e-8 | ~1e-8 | ~1e-9 | ~1e-8 | ~1e-4 |
| Dependencies | libm | — | — | C++ | Rust | C |
| Licence | Apache-2.0 | commercial | commercial | MIT | Apache-2.0 | MIT |

## Benchmarks

CPU benchmarks on Apple Silicon (arm64), macOS 27.2, Apple clang 21,
C99 `-O2`. Times are in seconds. **Bold** marks the lowest median in
each row. Except for LP50, LP, QP, SOCP and SDP use five runs at
`bd53531` (2026-11-07).
MILP uses ten interleaved runs at one thread at `b452e46` (2026-11-08).
The LP50 Primal and Clarabel cells use 18 interleaved runs (2026-11-10),
with both solvers checked to `1e-8`. Its HiGHS cell uses the November 7 run.
The objective checks passed in every run.

<div>

| instance | class | vars x cons | nnz | Primal (s) | HiGHS (s) | Clarabel (s) | SCS (s) | SCIP (s) | obj |
|---|---|---|---|---|---|---|---|---|---|
| lp_50x25 | lp | 50 x 25 | 171 | 0.000326 | 0.001461 | **0.000179** | N/A | N/A | 64.1886 |
| lp_100x50 | lp | 100 x 50 | 576 | 0.001388 | 0.001837 | **0.000648** | N/A | N/A | 121.946 |
| lp_200x100 | lp | 200 x 100 | 2120 | 0.006212 | 0.003626 | **0.002039** | N/A | N/A | 284.367 |
| lp_400x200 | lp | 400 x 200 | 8026 | 0.031574 | 0.012659 | **0.011852** | N/A | N/A | 546.481 |
| qp_50x25 | qp | 50 x 25 | 1445 | 0.001604 | N/A | **0.000392** | N/A | N/A | 12.8014 |
| qp_100x50 | qp | 100 x 50 | 5631 | 0.006800 | N/A | **0.002093** | N/A | N/A | 26.4052 |
| qp_200x100 | qp | 200 x 100 | 22212 | 0.044115 | N/A | **0.007664** | N/A | N/A | 55.9929 |
| milp_40x20 | milp | 40 x 20 | 139 | 0.045042 | **0.042070** | N/A | N/A | 0.134153 | 143.955 |
| milp_60x30 | milp | 60 x 30 | 260 | 0.091407 | **0.085865** | N/A | N/A | 0.238516 | 200.846 |
| milp_80x40 | milp | 80 x 40 | 460 | **0.199763** | 0.317508 | N/A | N/A | 0.953599 | 277.019 |
| socp_40 | socp | 60 x 1 | - | 0.000378 | N/A | **0.000097** | 0.000257 | N/A | 0.707107 |
| socp_120 | socp | 180 x 1 | - | 0.000235 | N/A | **0.000219** | 0.000701 | N/A | 0.707107 |
| socp_200 | socp | 300 x 1 | - | **0.000340** | N/A | 0.000423 | 0.001048 | N/A | 0.707107 |
| sdp_4 | sdp | 4 x 4 | - | **0.000025** | N/A | 0.000123 | 0.000200 | N/A | -9.83233 |
| sdp_5 | sdp | 5 x 5 | - | **0.000010** | N/A | 0.000134 | 0.000192 | N/A | -4.79948 |
| sdp_6 | sdp | 6 x 6 | - | **0.000010** | N/A | 0.000163 | 0.000215 | N/A | -10.2438 |
| sdp_7 | sdp | 7 x 7 | - | **0.000011** | N/A | 0.000252 | 0.000403 | N/A | -8.18613 |
| sdp_8 | sdp | 8 x 8 | - | **0.000012** | N/A | 0.000404 | 0.000461 | N/A | -22.165 |
| sdp_9 | sdp | 9 x 9 | - | **0.000014** | N/A | 0.000539 | 0.000380 | N/A | -17.0048 |
| sdp_10 | sdp | 10 x 10 | - | **0.000015** | N/A | 0.000678 | 0.000682 | N/A | -31.7195 |
| sdp_11 | sdp | 11 x 11 | - | **0.000018** | N/A | 0.001086 | 0.001724 | N/A | -20.5753 |
| sdp_12 | sdp | 12 x 12 | - | **0.000019** | N/A | 0.001677 | 0.000890 | N/A | -44.5174 |
| sdp_13 | sdp | 13 x 13 | - | **0.000023** | N/A | 0.001840 | 0.000974 | N/A | -38.7212 |
| sdp_14 | sdp | 14 x 14 | - | **0.000026** | N/A | 0.001892 | 0.001008 | N/A | -74.2121 |
| sdp_15 | sdp | 15 x 15 | - | **0.000030** | N/A | 0.002332 | 0.001260 | N/A | -51.0706 |
| sdp_16 | sdp | 16 x 16 | - | **0.000032** | N/A | 0.003333 | 0.001454 | N/A | -69.7722 |
| sdp_17 | sdp | 17 x 17 | - | **0.000039** | N/A | 0.004184 | 0.001667 | N/A | -38.8839 |
| sdp_18 | sdp | 18 x 18 | - | **0.000045** | N/A | 0.004503 | 0.001992 | N/A | -78.2696 |
| sdp_19 | sdp | 19 x 19 | - | **0.000046** | N/A | 0.004267 | 0.001919 | N/A | -123.432 |
| sdp_20 | sdp | 20 x 20 | - | **0.000060** | N/A | 0.006773 | 0.002362 | N/A | -140.192 |
| sdp_21 | sdp | 21 x 21 | - | **0.000058** | N/A | 0.009405 | 0.003735 | N/A | -93.0904 |
| sdp_22 | sdp | 22 x 22 | - | **0.000069** | N/A | 0.010784 | 0.008129 | N/A | -149.613 |

</div>

`N/A` means not measured by this harness, not necessarily unsupported. SDP
sizes denote the PSD matrix order. SOCP models have `n+n/2` scalar variables.
For SOCP/SDP, `obj` is the analytic reference value.
Primal timings cover `PRIMAL_optimize`; Clarabel timings exclude solver
construction, while the SCS calls include setup. Default tolerances differ,
and objective agreement alone does not establish equal solution accuracy.

The SDP rows use fixed-diagonal models with compatible objective signs.
See the [benchmark methodology](docs/benchmarks.md) for reference versions,
accuracy checks and reproduction.

### Mixed SOC and exponential cones

The regression models combine one rotated quadratic cone with a primal
exponential cone per observation. Sizes give observations × features.

| Instance | Primal (ms) | Clarabel (ms) |
|---|---:|---:|
| 20×4 | 3.39 / 3.46 | **0.26 / 0.19** |
| 40×8 | 10.50 / 10.10 | **0.58 / 0.58** |
| 80×12 | 43.38 / 43.31 | **2.62 / 2.58** |

Measured on 2026-11-07. Each cell shows two medians from nine alternating
pairs per series. Primal timings cover `PRIMAL_optimize`. Clarabel 0.11.1
runs at one thread and includes solver construction. The results met
the `1e-8` benchmark checks for objective agreement, feasibility and
relative gap. See the [mixed-cone benchmark notes](docs/pexp-higher-correction.md#sparse-kkt-selection).

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
