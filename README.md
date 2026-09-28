# PrimalSolver

A convex optimization solver written from scratch in portable C99. No external
numerical dependencies — `libc`, `libm` and POSIX threads.

It solves **LP, MILP, QP, QCQP, SOCP, SDP, exponential and power cones, and
mixed-integer conic problems**, and it hands back the primal point, the dual
point and a status you can check rather than trust.

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

Model edits invalidate result statuses and objectives until the next solve;
imported solution vectors are unverified. For MIP, use `MIO_OBJ_BOUND` rather
than `getdualobj`, and check for a feasible incumbent when a limit is reached.

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
make test          # reliability suite: 4881 checks
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
- **HiGHS** is faster on large LP/MILP — 0.013 s against 0.10 s on a 400×200 LP.
  The crossover sits around 50–100 constraints; below that this solver wins on
  fixed overhead.
- **SDP with exponential cones and integer variables, in one dependency-free
  library**, is what nothing else permissively licensed covers. CVXOPT / SDPA /
  Sedumi have SDP but are GPL.

Needing raw LP/MIP throughput? Use HiGHS. Needing an SDP *and* an exponential
cone *and* an integer variable, in C, without linking anything? This is it.

## Status

**4881 checks, 0 failures** at `-O0`, `-O1`, `-O2`, `-O3`. **171/171** examples
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
- **Parameter coverage.** 31 declared, ~16 wired to behaviour.
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
