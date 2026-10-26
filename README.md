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
