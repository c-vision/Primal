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
proves it. [Details](algorithms/certificates.md).

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

The package **`PrimalSolver`** lives in [`julia/`](julia/) and is also a
[JuMP](https://jump.dev) solver. To install it next to JuMP (Julia 1.10 or
later and a C compiler; the first `using PrimalSolver` compiles the C library):

```sh
git clone https://github.com/c-vision/Primal.git
julia -e 'using Pkg; Pkg.develop(path="Primal/julia"); Pkg.add("JuMP")'
```

```julia
using JuMP, PrimalSolver

model = Model(PrimalSolver.Optimizer)
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

Measured on **2026-11-12** on Apple Silicon (arm64), macOS 27.2,
with Apple clang 21 and C99 `-O2`. Times are in seconds. **Bold**
marks the lowest median in each row. Each cell is the median of **15**
independent aggregations of `bench/bench_all.py`
(`tools/bench_repeat.py --runs 15`); the objective column is identical in all
5, and every aggregation runs the cross-solver objective check.

<div>

| instance | class | vars x cons | nnz | Primal (s) | HiGHS (s) | Clarabel (s) | SCS (s) | SCIP (s) | obj |
|---|---|---|---|---|---|---|---|---|---|
| lp_50x25 | lp | 50 x 25 | 171 | 0.0002 | 0.0018469 | **0.0001729** | N/A | N/A | 64.1886 |
| lp_100x50 | lp | 100 x 50 | 576 | **0.0006** | 0.0020062 | 0.000723 | N/A | N/A | 121.946 |
| lp_200x100 | lp | 200 x 100 | 2120 | **0.0016** | 0.0039667 | 0.0021485 | N/A | N/A | 284.367 |
| lp_400x200 | lp | 400 x 200 | 8026 | **0.0073** | 0.013067 | 0.0133544 | N/A | N/A | 546.481 |
| qp_50x25 | qp | 50 x 25 | 1445 | **0.0004** | N/A | 0.0004198 | N/A | N/A | 12.8014 |
| qp_100x50 | qp | 100 x 50 | 5631 | **0.0016** | N/A | 0.0022352 | N/A | N/A | 26.4052 |
| qp_200x100 | qp | 200 x 100 | 22212 | **0.0065** | N/A | 0.0078989 | N/A | N/A | 55.9929 |
| milp_40x20 | milp | 40 x 20 | 139 | **0.0497** | N/A | N/A | N/A | 0.1614309 | 143.955 |
| milp_60x30 | milp | 60 x 30 | 260 | **0.1026** | N/A | N/A | N/A | 0.2892254 | 200.846 |
| milp_80x40 | milp | 80 x 40 | 460 | **0.2235** | N/A | N/A | N/A | 1.1191591 | 277.019 |
| socp_40 | socp | 60 x 1 | - | 0.000136 | N/A | **0.0001001** | 0.0003133 | N/A | 0.707107 |
| socp_120 | socp | 180 x 1 | - | **0.000198** | N/A | 0.0002286 | 0.0007305 | N/A | 0.707107 |
| socp_200 | socp | 300 x 1 | - | **0.000298** | N/A | 0.0004684 | 0.0011015 | N/A | 0.707107 |
| sdp_4 | sdp | 4 x 4 | - | **0.000027** | N/A | 0.0001677 | 0.000218 | N/A | -9.83233 |
| sdp_5 | sdp | 5 x 5 | - | **0.00001** | N/A | 0.0001435 | 0.0002089 | N/A | -4.79948 |
| sdp_6 | sdp | 6 x 6 | - | **0.00001** | N/A | 0.0001958 | 0.0002428 | N/A | -10.2438 |
| sdp_7 | sdp | 7 x 7 | - | **0.000012** | N/A | 0.000265 | 0.0004205 | N/A | -8.18613 |
| sdp_8 | sdp | 8 x 8 | - | **0.000014** | N/A | 0.0004052 | 0.0004866 | N/A | -22.165 |
| sdp_9 | sdp | 9 x 9 | - | **0.000016** | N/A | 0.0005773 | 0.0003954 | N/A | -17.0048 |
| sdp_10 | sdp | 10 x 10 | - | **0.000017** | N/A | 0.0007208 | 0.0007054 | N/A | -31.7195 |
| sdp_11 | sdp | 11 x 11 | - | **0.00002** | N/A | 0.0011122 | 0.0017001 | N/A | -20.5753 |
| sdp_12 | sdp | 12 x 12 | - | **0.000021** | N/A | 0.0017985 | 0.0009701 | N/A | -44.5174 |
| sdp_13 | sdp | 13 x 13 | - | **0.000026** | N/A | 0.0020243 | 0.0011125 | N/A | -38.7212 |
| sdp_14 | sdp | 14 x 14 | - | **0.000029** | N/A | 0.0020907 | 0.0010707 | N/A | -74.2121 |
| sdp_15 | sdp | 15 x 15 | - | **0.000032** | N/A | 0.002432 | 0.001329 | N/A | -51.0706 |
| sdp_16 | sdp | 16 x 16 | - | **0.000032** | N/A | 0.0035239 | 0.0014924 | N/A | -69.7722 |
| sdp_17 | sdp | 17 x 17 | - | **0.000041** | N/A | 0.0043605 | 0.0018023 | N/A | -38.8839 |
| sdp_18 | sdp | 18 x 18 | - | **0.000048** | N/A | 0.0045971 | 0.0020848 | N/A | -78.2696 |
| sdp_19 | sdp | 19 x 19 | - | **0.000047** | N/A | 0.004302 | 0.0020414 | N/A | -123.432 |
| sdp_20 | sdp | 20 x 20 | - | **0.000066** | N/A | 0.0066506 | 0.0024955 | N/A | -140.192 |
| sdp_21 | sdp | 21 x 21 | - | **0.00006** | N/A | 0.0092842 | 0.0039946 | N/A | -93.0904 |
| sdp_22 | sdp | 22 x 22 | - | **0.000077** | N/A | 0.0107491 | 0.0086255 | N/A | -149.613 |

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

Each regression model uses one rotated quadratic cone for the quadratic
penalty and one primal exponential cone per observation. Sizes give
observations × features.

| Instance | Primal (ms) | Clarabel (ms) | ratio |
|---|---:|---:|---:|
| 20×4 | 0.59 | **0.29** | 2.1× |
| 40×8 | 1.37 | **0.63** | 2.2× |
| 80×12 | 3.73 | **2.72** | 1.4× |

Measured on 2026-11-12 with `bench/pexp_bench.py --runs 21`, repeated three
times: each cell is the median of the three medians, from interleaved runs of
both solvers. The instance is generated by
`bench/pexp_bench.c` from its own seed, so the table is reproducible from a
clean checkout; the script asserts every answer against the known optima
(`18.718151648415301`, `47.09337729630259`, `81.84315908327622`) and the
`1e-8` relative triple. Primal timings cover `PRIMAL_optimize`; Clarabel 0.11.1
runs at one thread and includes solver construction. See the
[mixed-cone benchmark notes](algorithms/pexp-higher-correction.md#mixed-soc-and-exponential-cones-2026-11-12)
for what moved the numbers.

## Documentation

The [algorithm guide](algorithms/README.md) follows the numerical methods
from model formulation through solution verification. See the
[API guide](docs/api.md) and [I/O formats](docs/io-formats.md) for reference
material.

## License

Apache-2.0. All code is original; see [LICENSE](LICENSE). The examples that port
models from MOSEK's public documentation re-implement them. PrimalSolver is not
affiliated with, or endorsed by, MOSEK.
