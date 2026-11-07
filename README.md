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

PrimalSolver combines SDP, exponential and power cones, and integer
variables in one permissively licensed C library. The benchmarks below
include a fixed-diagonal SDP family where Primal can construct the
solution directly. Clarabel remains faster on the mixed SOC/PEXP case.

The mixed SOC/exponential route now includes a higher-order corrector. Local
paired measurements show 24–48% less solve time on three exponential
regression sizes, including 33–35% on the 40×8 case. The focused checks pass
928/928; this is not a full-suite result. Clarabel remains substantially faster
on that case. See [measurements, scope and validation](docs/pexp-higher-correction.md).

## Benchmarks

Measured on 2026-11-07 at `bd53531`, Apple Silicon (arm64), macOS 27.2,
Apple clang 21, C99 `-O2`. Times are in seconds and show the median of five
complete runs. **Bold** marks the lowest measured median in each row.
The objective checks passed in every run.

<div>

| instance | class | vars x cons | nnz | Primal (s) | HiGHS (s) | Clarabel (s) | SCS (s) | SCIP (s) | obj |
|---|---|---|---|---|---|---|---|---|---|
| lp_50x25 | lp | 50 x 25 | 171 | 0.001053 | 0.001461 | **0.000166** | N/A | N/A | 64.1886 |
| lp_100x50 | lp | 100 x 50 | 576 | 0.001388 | 0.001837 | **0.000648** | N/A | N/A | 121.946 |
| lp_200x100 | lp | 200 x 100 | 2120 | 0.006212 | 0.003626 | **0.002039** | N/A | N/A | 284.367 |
| lp_400x200 | lp | 400 x 200 | 8026 | 0.031574 | 0.012659 | **0.011852** | N/A | N/A | 546.481 |
| qp_50x25 | qp | 50 x 25 | 1445 | 0.001604 | N/A | **0.000392** | N/A | N/A | 12.8014 |
| qp_100x50 | qp | 100 x 50 | 5631 | 0.006800 | N/A | **0.002093** | N/A | N/A | 26.4052 |
| qp_200x100 | qp | 200 x 100 | 22212 | 0.044115 | N/A | **0.007664** | N/A | N/A | 55.9929 |
| milp_40x20 | milp | 40 x 20 | 139 | 0.289115 | N/A | N/A | N/A | **0.150934** | 143.955 |
| milp_60x30 | milp | 60 x 30 | 260 | 1.031619 | N/A | N/A | N/A | **0.267775** | 200.846 |
| milp_80x40 | milp | 80 x 40 | 460 | 3.066108 | N/A | N/A | N/A | **1.010305** | 277.019 |
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

### Earlier SDP accuracy and performance measurements

In the November 5 snapshot, before the accuracy fix, SDP20's
relative objective error was about `1.3e-6`. At `314d4f0`, all 19 published SDP
orders (4–22) meet `1e-8` for relative objective error, reported primal and
dual infeasibility, and relative gap. SDP20's relative gap is `3.13e-9`.

The first optimization reduced the time spent recovering an accurate
solution. These medians come from two series of nine alternating pairs.
The baseline is the accuracy fix at `adb7f74`.

| Model | Accuracy fix (ms) | Optimized solve (ms) |
|---|---:|---:|
| SDP14 | 47.83 / 48.20 | 7.03 / 6.55 |
| SDP20 | 247.02 / 235.77 | 55.98 / 52.63 |
| SDP21 | 137.55 / 137.47 | 38.61 / 37.63 |
| SDP22 | 187.72 / 184.24 | 47.80 / 46.28 |

Combining faster PSD step checks with elimination of fixed coordinates
from the recovery system reduced the sum of the four median times by
6.9–7.1% against `314d4f0`. The 19 published SDP orders still meet
the accuracy limits above.

| Model | `314d4f0` (ms) | Combined optimizations (ms) |
|---|---:|---:|
| SDP14 | 5.14 / 5.57 | 4.70 / 5.04 |
| SDP20 | 45.40 / 42.19 | 41.97 / 39.67 |
| SDP21 | 35.32 / 28.92 | 31.75 / 26.16 |
| SDP22 | 41.24 / 35.63 | 39.62 / 33.66 |

A separate comparison on SDP20 (seed 220) at `314d4f0` gave 59.67 / 61.44 ms for
`PRIMAL_optimize` and 10.07 / 10.19 ms for Clarabel 0.11.1, including solver
construction and solve at one thread. These are medians of two series of
nine alternating pairs. Both relative objective errors were below `1e-8`.
The relative gaps were `3.13e-9` for Primal and `2.01e-11` for Clarabel.

For a single PSD block with every diagonal fixed and compatible objective
signs, Primal now constructs a primal and dual certificate directly. The
interior-point solver handles the other cases. On this family, two new
series of nine alternating pairs gave these solve times against `1bf265c`.

| Model | `1bf265c` (ms) | Direct certificate (ms) |
|---|---:|---:|
| SDP20 | 36.50 / 40.79 | 0.083 / 0.086 |
| SDP21 | 25.54 / 28.47 | 0.086 / 0.090 |
| SDP22 | 36.14 / 36.86 | 0.098 / 0.097 |

All 19 orders (4–22) met the accuracy limits above with this change.

See the [benchmark methodology](docs/benchmarks.md) for reference versions,
accuracy limits and reproduction.

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
