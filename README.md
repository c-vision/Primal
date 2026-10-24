# PrimalSolver

A convex optimization solver written from scratch in portable C99. No external
dependencies — `libc` and `libm`, nothing else.

It solves **LP, MILP, QP, QCQP, SOCP, SDP, exponential and power cones, and
mixed-integer conic problems**, and it hands back the primal point, the dual
point and a status you can check rather than trust.

[![Sponsor](https://img.shields.io/badge/Sponsor-❤-ea4aaa)](https://github.com/sponsors/c-vision)
[![PyPI](https://img.shields.io/pypi/v/primalsolver)](https://pypi.org/project/primalsolver/)

## Where it fits

The niche is the **combination**, not any single capability: among solvers with
a permissive licence, none other does semidefinite programming *and*
exponential/power cones *and* mixed-integer variables with an interior-point
method, in C, with no dependencies and no licence server.

It is not a replacement for HiGHS on large LP/MIP. Conic performance depends
on the family; SOCP (rotated cones included) still trails Clarabel and SDP trails SCS. [How it compares](#how-it-compares) says by
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

171 runnable examples live under `examples/c/`, each checking its own result: 48
ports of public MOSEK examples with identical optimal values, 63 larger
deterministic models, 44 finance models, 16 book models.

## Command line

The default `make` also builds `out/primal`, a small command-line solver that
reads a model file, solves it and prints the solution. **The file is the
interface**: the algorithm is chosen from what the model contains, so the same
command solves LP, QP, SOCP, SDP, exponential/power and mixed-integer models.

An input file (CPLEX LP format, the easiest to write by hand):

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
```
PrimalSolver 0.1.0
file    : diet.lp   (numvar=2  numcon=2)
status  : OPTIMAL   prosta=PRIM_AND_DUAL_FEAS  solsta=OPTIMAL
obj     : primal = 11   dual = 11
viol    : primal = 0.000e+00
x[x0] = 3
x[x1] = 1
```

It reads MPS (`.mps`), CPLEX LP (`.lp`), OPF (`.opf`) and CBF (`.cbf`) — detected
automatically — and accepts `--brief`, `--full` (also the duals and slacks),
`--sensitivity` (LP cost/RHS ranges), `--solution-file FILE`, `--write FILE`,
`--max-iter N`,
`--tol-pfeas/--tol-dfeas/--tol-gap V` and `--param NAME=VALUE`
(run `./out/primal --help`).

Ready-to-run inputs for every problem class — LP, MIP, QP, SOCP, an exponential
cone and an SDP — live in [`examples/lp/`](examples/lp/README.md), each with its
expected output; `make run-examples` solves them all.

## Python

On PyPI: [**`primalsolver`**](https://pypi.org/project/primalsolver/) — `ctypes`
bindings over the C library, no compiler needed:

```sh
pip install primalsolver
python -c "import primalsolver; print(primalsolver.version())"
```

Wheels for macOS (universal2) and Linux (x86_64/aarch64) are published to PyPI and
also attached to the [latest
release](https://github.com/c-vision/Primal/releases/latest); Windows wheels are
built by the CI. **172 runnable examples** live in
[`examples/python/`](examples/python/README.md): one per C sample, with the same
output as the C program, plus a quickstart. The README covers installation, usage,
and one line per example.

## Julia

Package **`Primal`** in [`julia/`](julia/) — native Julia bindings over the C
library via `ccall`, exposing the full public `PRIMAL_*` API:

```sh
# from the PrimalSolver checkout
julia --project=julia -e 'include("julia/deps/build.jl"); using Pkg; Pkg.instantiate()'
julia --project=julia -e 'using Primal; println(Primal.version())'
```

### JuMP

`Primal.Optimizer` is a [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl)
solver, so Primal can be used from [JuMP](https://jump.dev) as any other. To install
it next to JuMP in any Julia environment (Julia 1.10 or later and a C compiler; the
first `using Primal` compiles the C library):

```sh
git clone https://github.com/c-vision/Primal.git
julia -e 'using Pkg; Pkg.develop(path="Primal/julia"); Pkg.add("JuMP")'
```

(`Pkg.add(url=..., subdir="julia")` does not work yet: it copies `julia/` alone,
without the C sources the build needs.)

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

It accepts linear and convex quadratic objectives, linear constraints, binary and
integer variables, and the second-order, rotated second-order, exponential and power
cones (primal and dual); JuMP's bridges reformulate the rest (norm cones, geometric
mean, relative entropy, quadratic constraints). `set_silent`, `set_time_limit_sec`
and `set_attribute(model, "PRIMAL_IPAR_INTPNT_MAX_ITERATIONS", 500)` (any
`PRIMAL_IPAR_*` / `PRIMAL_DPAR_*` parameter of `primal.h`) set the solver options.
The wrapper passes MathOptInterface's conformance suite (571 tests; one excluded and
five at a looser tolerance, each documented in `julia/test/MOI_wrapper.jl`):

```sh
julia --project=julia -e 'using Pkg; Pkg.test()'
```

**171 runnable examples** live in [`examples/julia/`](examples/julia/), one per C sample,
with the same output as the C program. Each checks its own result and exits
non-zero on failure. Run them all with:

```sh
julia --project=julia julia/runtests.jl
```

The Julia package lives in this same repository under `julia/`; it compiles the C
shared library into `julia/deps/` on first use (no system compiler needed beyond
`gcc`/`clang`, already required to build the C library).

## Build

```sh
make               # gcc -std=c99 -Wall -Wextra -pedantic -O2, zero warnings
make test          # reliability suite: 5260 checks
make run-samples   # the 171 examples
make clean         # remove out/
```

Everything the build produces goes to `out/` (gitignored). **Portability**: POSIX
C99 with `gcc` or `clang` on Linux, macOS and the BSDs. There is **no MSVC
project and no `nmake` path** — on Windows use MinGW
([how to build on Windows](docs/windows.md)), or compile the `LIBSRCS` list from
the Makefile with any C99 compiler. The full suite passes at `-O0` to `-O3`,
without FMA contraction (`-ffp-contract=off`, which is what plain gcc on
x86-64 does) and under ASan + UBSan, with no sanitizer diagnostic.

## How it compares

Capability, licence and accuracy class — **not** speed.

| | PrimalSolver | MOSEK | Gurobi | HiGHS | Clarabel | SCS |
|---|---|---|---|---|---|---|
| LP / QP | yes | yes | yes | yes | yes, native QP | yes |
| SOCP | yes | yes | Gurobi only | — | yes | yes |
| **SDP** (PSD) | yes | yes | — | — | yes | yes, first-order |
| **Exp / power** | yes, native | yes | — | — | yes | yes, first-order |
| MIP / MIQP | yes | yes | yes | yes | — | — |
| Method | interior point | interior point | simplex + IPM | simplex + IPM | interior point | **first-order** |
| Accuracy | ~1e-8 | ~1e-8 | ~1e-8 | ~1e-9 | ~1e-8 | ~1e-4 |
| Dependencies | **libm** | — | — | C++ | Rust | C |
| Licence | Apache-2.0 | commercial | commercial | MIT | Apache-2.0 | MIT |

The rows that decide it:

- **Clarabel** is the closest open conic IPM. It was **~7× faster** in the
  historical SOCP case (0.0004 s against 0.0027 s for `socp_200`: 100 cones,
  300 scalar variables, down from 0.014 s via sparse Cholesky, a fill-reducing
  ordering, and caching that ordering across iterations).
  It supports PSD cones and quadratic objectives directly; see its
  [documented capabilities](https://clarabel.org/stable/).
  These timings are indicative: the harness excludes Clarabel's solver
  construction but includes PrimalSolver's conversion inside `PRIMAL_optimize`.
- **SCS** does PSD and exponential cones, but is first-order: it agrees with
  this solver's objective to about `1e-5`, where the IPM reaches `1e-8`.
- **HiGHS** is faster on large LP/MILP — 0.011 s against 0.030 s on a 400×200 LP.
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
shows `N/A`). On this machine they are all installed in one interpreter, so:

```sh
make bench && /Users/gaetano/ai/env-bench/bin/python3.14 bench/bench_all.py
```

That script times **every** solver in the table (PrimalSolver, HiGHS, Clarabel,
SCS, SCIP) and cross-checks each objective. It takes **one run per cell**, so the
table it prints is a single-run snapshot. For anything finer than ~10% use
`make bench-repeat`, which runs it N times, medians every cell and checks that
every objective is identical across the runs.

The default `python3` has **none** of them and every reference column comes out
`N/A` -- which looks like a missing benchmark rather than a missing package. `bench/bench_all.py` prints this table and cross-checks every
objective against the analytic value; it exits non-zero on a mismatch. LP/QP/
MILP are the generated MPS instances, SOCP/SDP are the closed-form families of
`bench/conic_bench.c` and `bench/sdp_sweep.c`. The SDP sweep reaches **d = 22**.

The SDP eigenvalue kernel (`dmat_eig_jacobi`, `min_eig`) now updates the
symmetric matrix one-sided (half the arithmetic per Jacobi rotation). The
`d = 4..22` sweep is ~**1.4×** faster overall (interleaved base/new; `d = 22`
~1.5×, `d = 20` ~2.9×). The gain is **not uniform per size**: the ~1e-14 change
in the eigenvalues can move which trajectory stalls, so one size can be slower
(`d = 14`). That is the same route non-reproducibility `T81`/`T91` already
assert as an outcome.

**Four changes account for the SDP speed-up**. The SDP family is **23.3%
faster** in the historical comparison recorded on 2026-10-05 (0.2442 s -> 0.1873 s,
medians of 5), and **no instance in the table is slower** (27 faster, 0 slower).
`d = 22` goes 0.0349 s -> 0.0277 s. Each was measured at **`-O1`, `-O2` and `-O3` with distinct binaries**
and interleaved pairs — a gain that survives `-O1` is work reduction, not code
generation:

1. **`sym_fun` shares one eigendecomposition per matrix, and `f(λ)` is hoisted
   out of its `O(d³)` loop.** `sym_fun` is a spectral function: it takes an
   eigendecomposition of `A` and applies `f` to the eigenvalues (`kind 0` →
   √λ, `1` → 1/√λ, `2` → 1/|λ|). A profiler over the ratio test showed **6
   calls per iteration but only 4 distinct input matrices**: `Xbar^1/2`,
   `Xbar^-1` and `Xbar^-1/2` are three spectral functions of the *same* `Xbar`,
   and `dmat_eig_jacobi` is deterministic, so the decomposition is kept and
   reused — **33% of the eigendecompositions skipped**, uniformly at `d` = 4, 14,
   19, 22. Separately, `f(λ_k)` depends only on `k` but was recomputed in every
   one of the `d³` positions: **10648 square roots or divisions per call at
   `d = 22` where 22 suffice**. Together **-9.8% / -10.9% / -5.3%** at `-O1` /
   `-O2` / `-O3`, disjoint ranges at all three, 11 of 11 pairs faster at each.
   The persistent buffers also remove 3 `malloc` + 3 `free` per call.
2. **The `sym` matrices are computed once per iteration instead of twice.** The
   ratio test evaluates `Xbar[j]` and `Sbar[j]`, and neither is assigned
   anywhere inside `for (pass = 0; pass < 2; pass++)` — a checksum of both at
   the two branches is identical in every iteration. Yet `sym(Xbar[j],1)` and
   `sym(Sbar[j],1)` were recomputed on both passes: **2 of the 8 `sym_fun`
   calls per iteration were pure repeats**. `sym_fun` is 55.4% of the SDP hot
   loop, so removing a quarter of it is most of the win: **-9.0% / -10.3% /
   -7.5%** at `-O1` / `-O2` / `-O3` on `d = 22`. `sdp_sweep 4..22` keeps
   **19/19** objectives identical to the last digit.
3. **`min_eig` sign tests go through a `min_eig_sign` wrapper.** Worth
   **-5.2% / -6.0% / -5.5%**. Its Cholesky-first attempt **never succeeds** on
   the SDP corpus (0 of 112 calls at `d = 22`, 0 of 273 at `d = 19`, 0 of 106
   at `d = 14`) — these matrices sit on the PSD boundary, because the call
   sites are the step-length ratio test. **The mechanism is not identified**:
   both builds execute the same work (172 `min_eig` and 224 `sym_fun` calls,
   counted), so the gain is neither avoided decompositions nor a reduction in
   calls. Treat it as measured but unexplained, and re-measure it if the
   compiler changes.
4. **The eigenvalue kernel updates the symmetric matrix one-sided** (described
   above), worth ~1.4x on the sweep.

Change 1 is the one place where results are **not** bit-identical, and it is worth
being precise about. Objectives are bit-identical at `-O1` and `-O3`. At `-O2`
three instances move by at most **5.7e-7** relative (`d=12` -44.517447 →
-44.517445, `d=13` -38.721146 → -38.721155, `d=14` -74.212068 → -74.212062);
nothing in the family exceeds 1e-6. The cause is **FMA contraction, not
arithmetic**: compiled with `-ffp-contract=off` the two versions are
bit-identical, so they are the same computation and differ only in how the
compiler groups the multiply-add once `f` is a load rather than a computed value.
The suite passes at `-O2`, which is the level that differs.

LP, QP, MILP and SOCP move by -3.4% to +0.2% in the same comparison, which is
noise: those routes are untouched, and a change confined to one code path cannot
move a family it does not call. The **grand total is not a useful figure here** —
the SDP rows are a fraction of the time the table spends in MILP/LP/QP.

A build trap worth knowing: **`make` does not rebuild when only `CFLAGS`
changes**, so comparing two optimisation levels with the same `out/` silently
compares one binary with itself -- and that mistake is what produced a *false*
explanation once already. Use a separate `OUT=` per level, and check the two
binaries differ before measuring.

And **the table's own measurement uncertainty is 5-13%**: aggregating the
*same* build with 3 runs instead of 5 moves LP by +13.0%, QP by +10.8%, MILP by
+10.2%. Read the per-family figure against a **~10%** threshold.

Regenerate the table with `make bench-repeat` (medians, objective stability
checked) rather than a single run.

The sweep answers at `d = 20`, but **how long it takes is a build-time choice**:
the same block takes **0.12 s** on an Apple build and **18.0 s** elsewhere. That is
the stalled-point policy (`sdp.c`, `GMB_STALL_IS_ANSWER`): Apple publishes the
frozen point when it measures within the same absolute residual the suite asserts,
every other toolchain runs the tangent-cut outer approximation first. See
`docs/interior-point.md`; the two policies are both green on the suite.

**Sparse LU update (2026-10-20).** Pivot search retains each row's
column-slot index for elimination. Rows also track the boundary of their
completed L entries, so searches and updates skip that prefix. A 64-bit
presence filter rejects absent columns; collisions fall through to the exact
search. All metadata moves with row swaps. Storage remains O(n) beyond the
existing sparse factors; pivot selection and arithmetic order are unchanged.

Against `66f8bd9`, **31 interleaved triples** at `-O2` give:

| case | original | pivot-slot cache only | current | change vs original |
|---|---:|---:|---:|---:|
| SOCP 120 | 1.544 ms | 1.353 ms | 1.201 ms | −22.2% |
| SOCP 200 | 2.982 ms | 2.469 ms | 2.132 ms | −28.5% |

All eight conic benchmark outputs match except elapsed time. The additional
reduction beyond the October 9 patch is 11.2% / 13.6%, respectively. T281/T282
cover row swaps, fill, column ordering, filter collisions and repeated solves.
An exact next-column cache was also tried and rejected: its further 4.4% gain
missed the predeclared 5% minimum. This optimization **does not reach Clarabel**:
in the refreshed table SOCP 200 is about **4.6×** slower (rounded timings).
SDP convergence remains a separate unresolved gap.

**Experimental symmetric SOCP KKT (2026-10-21).** Setting `GMB_SOCP_LDL=1`
enables a direct NT system with sparse LDL and reusable symbolic analysis.
It retains the existing solver as fallback and leaves default routing unchanged.
The first 31 alternating pairs give SOCP 120: 1.566 → 1.185 ms, and SOCP 200:
2.820 → 1.087 ms. These targeted timings do not update the canonical table
below. Both backends pass the full suite at every level and under
ASan + UBSan; see the current validation note under Status.
[Equations, numerical gates and limitations](docs/socp-symmetric-kkt.md).

**Rounding-independence fixes (2026-10-21).** Building the suite without FMA
contraction or without vectorised sums exposed three defects that the default
macOS/clang `-O2` build was masking:
- the SOC/R+ ratio test of the legacy SOCP solver returned a *negative* step
  once a coordinate had rounded below zero, and the iterate was moved
  backwards out of the cone (T286 without FMA diverged to `feas_p=3.5e+260`);
  it now returns 0;
- in the unified conic IPM a split free variable pair could run away
  (`x/s = 6.6e21` on T244 at `-O0`), and the R+ normal-equations elimination
  then produced a direction with `|rp + A d| = 0.056` against `|rp| = 1e-12`.
  A direction that misses its own primal rows by more than `10|rp| +
  1e-10(1+|b|)` is now recomputed with the R+ block in augmented form, which
  stays on for the rest of the solve. This fixes T244 at every level and also
  lets the first pass of T111's PPOW model answer on its own. The check fires
  in 24 of the 171 examples in the retained run logs; nine print a different last digit (all still
  within tolerance: `qcqp_sdo_relaxation` −0.19999997 → −0.20000000,
  `risk_parity` spread 3.3e-4 → 6.2e-5, `mle_density` objective error
  4e-10 → 9e-10);
- `T83 A` asked for `u` to 1e-5 at an optimum where the objective is flat;
  the tolerance is now derived from the objective error, `sqrt(2 df / f'')`.

**SOC scaling update (2026-10-20).** The quadratic representation now computes
the nonzero products of the arrow matrix directly in O(k²), preserving their
summation order. In seven alternating pairs of 200,000 scaling setups, the
setup cost falls 20.7% at cone dimension 3 and 54.1% at dimension 8. This is a
kernel measurement: 21 alternating end-to-end runs show no resolved gain on
SOCP 200 (3.127 vs 3.147 ms). T283 checks dimensions 1, 3, 8 and 32 against
the dense solver, the exact norm optimum, stationarity, dual cone membership
and primal-dual gap. The 64 targeted checks pass before and after; no full
suite or sanitizer run was performed for this update. The canonical table
below predates this scaling change.

The **Primal** column below was refreshed on **2026-10-21** at `e90c614`
(the solver code there is that of `ee1081e`), with the C harness alone
(`out/bench/solve_mps` on the ten MPS instances, `out/bench/conic_bench`,
`out/bench/sdp_sweep 4 22`): medians of **11 rounds, interleaved** with the
previous version `2a71577` in the same session, the order alternating every
round. All 704 runs returned `rc=0`, and every objective is identical between
the two versions and equal to the `obj` column. The three **SOCP rows** were
then re-measured after the second-order SOCP corrector, the same way
(`conic_bench`, 11 interleaved rounds against `aa6b378`): see below.

**Historical comparison before the second-order SOCP corrector:** performance
is unchanged by the earlier 2026-10-21 work (symmetric SOCP KKT opt-in,
direction check, boundary recovery, HSD fix):

| family | `2a71577` (s) | current (s) | change |
|---|---:|---:|---:|
| LP (4) | 0.0425 | 0.0420 | −1.2% |
| QP (3) | 0.0540 | 0.0540 | +0.1% |
| MILP (3) | 4.4179 | 4.4105 | −0.2% |
| SOCP (3) | 0.0037 | 0.0037 | +1.1% |
| SDP (19) | 0.1925 | 0.1938 | +0.7% |
| **total** | **4.7106** | **4.7040** | **−0.1%** |

Single cells move between −3.5% and +6.2% (the latter `sdp_4`, 0.16 ms, at
the timer's resolution), most within ±2%. The only detectable effect is
SDP's +0.7%: the current version is slower in 122 and faster in 82 of the
204 paired SDP runs. Its cause is not identified (the direction check is
inactive on the sweep, which has no scalar variables). A first refresh of
this column, five non-interleaved rounds taken in a slower machine state,
had suggested up to +11% on `sdp_19`..`sdp_21`; set against the 2026-10-20
column from another session, that was machine drift, not code — the
interleaved measurement gives +0.2%, +0.8%, +0.9% there. Compare versions
only within one session.

**SOCP second-order corrector (2026-10-21).** The Mehrotra corrector of the
sparse SOCP solver used to change only the centering target `sigma*mu`; it
now also carries the affine second-order term `-ds_aff o dlambda_aff`, in
balanced NT coordinates on the NT engine and as the raw Jordan product on the
augmented-LU engine ([derivation, oracle and limits](docs/socp-mehrotra-correction.md);
`T289` checks the formula against an independent eigendecomposition). The
opt-in `GMB_SOCP_LDL` backend keeps its previous corrector. Measured with the
C worker `bench/socp_compare.c` (`PRIMAL_optimize` only), 21 rounds
interleaved with `aa6b378`, medians:

| case | before | after | change | after faster |
|---|---:|---:|---:|---:|
| SOCP 40 | 0.362 ms | 0.370 ms | +2.2% | 7/21 |
| SOCP 120 | 1.165 ms | 0.296 ms | **−74.6%** | 21/21 |
| SOCP 200 | 2.066 ms | 0.536 ms | **−74.1%** | 21/21 |
| SOCP 200, unequal costs, 2 equalities | 0.655 ms | 0.586 ms | −10.6% | 21/21 |
| SOCP 200, rotated cones | 24.70 ms | 20.09 ms | −18.6% | 21/21 |

Every run ends `OPTIMAL` with primal/dual residuals and gap below `1e-8`; the
canonical cases stop slightly earlier (`SOCP 120` gap 5e-12 → 4.4e-10), the
rotated case finishes tighter (3.3e-9 → 1.3e-11). In the table the three SOCP
rows become 0.0004 / 0.0003 / 0.0006 s (`conic_bench`, 11 interleaved rounds,
from 0.0004 / 0.0013 / 0.0023). Suite 5233/0 at `-O0`..`-O3`, without FMA and
under ASan + UBSan, with both SOCP backends; 171/171 examples exit 0, only
`logcontrast_lasso` prints different digits (β moves in the 4th decimal on a
flat objective, its own checks still pass). The comparison with Clarabel
needs `tools/socp_compare.py` and the reference Python environment; it
reports SOCP 120/200 at 0.253 / 0.491 ms against Clarabel construction+solve
0.333 / 0.612 ms, a measurement taken by the previous session and not
repeated here.

**Rotated cones written on their members (2026-10-21).** A rotated cone
`2uv >= |w|^2` used to reach the SOCP solver through `m+2` auxiliary copies of
its members and `m+2` link equalities, with the cone rows on the copies. The
members then had no cone row of their own, the NT matrix `G'Q(w)^-1 G` was
singular and every iteration ran on the augmented LU (SOCP 200 rotated:
n = 800, p = 501, 6 of 6 steps LU). The block is now the SOC
`((u+v)/sqrt2, (u-v)/sqrt2, w)` written directly on the members — an
orthogonal transform of two rows, like QUAD's identity rows — so the same
model is n = 300, p = 1 and runs on NT (6 NT + 2 LU steps). The published
member duals are the same (`G'lambda` reproduces the old link multipliers
exactly). Measured with the C worker, 21 rounds interleaved with `0180c52`:

| case | before | after | change | after faster |
|---|---:|---:|---:|---:|
| SOCP 120, rotated cones | 6.105 ms | 1.658 ms | **−72.8%** | 21/21 |
| SOCP 200, rotated cones | 19.715 ms | 3.797 ms | **−80.7%** | 21/21 |
| SOCP 40 / 120 / 200, canonical | 0.364 / 0.292 / 0.528 ms | 0.366 / 0.290 / 0.530 ms | within ±1% | — |
| SOCP 200, unequal costs | 0.571 ms | 0.576 ms | +0.9% | 11/21 |

Both versions end `OPTIMAL` on the rotated cases with objectives equal to
~1e-9; the gap gets tighter (SOCP 120 rotated 4.4e-9 → 4.8e-12). The table's
SOCP rows are unchanged (that family has no rotated cone). Against Clarabel
the rotated cases move from ~15× / ~32× to roughly ~4× / ~6× slower — derived
from the previous session's Clarabel timings (~0.41 / ~0.62 ms), not
re-measured, so indicative only.

The smaller model also exposed the dense SOCP solver used for small
relaxations: on the MISOCP `T212` (k-means, rotated cones + disjunctions) 5 of
25 node relaxations already stalled before (the optimum was found by chance)
and 10 of 32 after, losing it (2.0 for 1.0). A dense solve that fails now gets
one attempt with the sparse production engines (`socp_solve_sparse_production`,
never the opt-in LDL backend) before the round gives up; only a success
replaces its status, so infeasible models keep their verdict (`T101 C`). `T212`
then answers every node. Suite 5233/0 at `-O0`..`-O3`, without FMA and under
ASan + UBSan, with both SOCP backends at every level; 171/171 examples exit 0,
with last-digit changes in `lsq_l1_penalty` and `primal_svm`, and `rank_one`
returning the other of its two symmetric optima `b = (3/2, 0)`.

**Fixed variables substituted, sparse LU incidence (2026-10-22).** On the
sparse SOCP route a fixed variable is no longer two opposite `R+` rows (an
interior of width zero, which pushed the last iterations onto the augmented
LU): its value is substituted into the equality and cone right-hand sides,
the reduced model is solved, and the bound multipliers are lifted back from
stationarity, `max(r_j,0)` and `max(-r_j,0)` with
`r_j = c_j + (E'y)_j - (G'lambda)_j`. The reduced tolerances are only ever
tightened; an inconsistent row, an allocation failure or an unsuccessful
reduced solve falls back to the original model
([details](docs/socp-fixed-reduction.md); `T291`). The sparse LU also keeps an
exact column-to-row incidence bitset (≤ 8 MiB, portable C99) so a pivot column
visits only the rows that hold it, in the same order as before — the factors
are byte-identical (`T290`) — and the augmented fallback orders its columns
`[lambda, x, y]` instead of RCM. Measured with the C worker, 21 rounds
interleaved with `bb1beaf`:

| case | before | after | change | after faster |
|---|---:|---:|---:|---:|
| SOCP 120, rotated cones | 1.675 ms | 0.512 ms | **−69.4%** | 21/21 |
| SOCP 200, rotated cones | 3.775 ms | 0.987 ms | **−73.9%** | 21/21 |
| small rotated, n = 2 / 10 / 40 | 0.019 / 0.059 / 0.608 ms | 0.015 / 0.059 / 0.593 ms | −20.3% / −0.6% / −2.4% | 21 / 13 / 19 |
| SOCP 40 / 120 / 200, canonical | 0.368 / 0.307 / 0.561 ms | 0.366 / 0.300 / 0.550 ms | −0.5% / −2.2% / −2.1% | 14 / 15 / 14 |
| SOCP 200, unequal costs | 0.594 ms | 0.600 ms | +1.1% | 9/21 |

All runs end `OPTIMAL`; the rotated objectives agree to ~1e-9 and every gap
stays below 1e-10. Over the two rotated-cone steps of 2026-10-21/22 the
rotated SOCP 200 goes 19.7 → 0.99 ms (−95%); against the previous session's
Clarabel construction+solve timings (~0.41 / ~0.62 ms, not re-measured, so
indicative) the rotated cases are now about 1.2× / 1.6× slower instead of
~15× / ~32×. The table's SOCP rows are unchanged (`conic_bench` 0.000371 /
0.000301 / 0.000550 s against 0.000371 / 0.000300 / 0.000549 s). Suite
5254/0 at `-O0`..`-O3`, without FMA and under ASan + UBSan, with both SOCP
backends at every level; 171/171 examples exit 0, only `logcontrast_lasso`
moves β in the 4th decimal on its flat objective (its checks pass).

The **reference columns** (HiGHS, Clarabel, SCS, SCIP) are still the
**2026-10-20** snapshot: they need `bench_all.py` with the reference solvers'
Python packages, which was not run for this refresh. Read cross-solver ratios
as indicative; the two sides were not measured in the same session.
Use the interleaved measurements above for the LU gain.
The Apple stalled-point policy described above remains in effect.

The exp/power grid's October 9 result remains **SELECTOR OK but REJECTED**:
T47 radial KKT is `−6.82e-4` against `1e-4`. The current route probe produces
identical output to the pre-change LU probe, including T47. This pre-existing
failure is not counted as a green validation.

<div style="font-size: 0.9em">

| instance | class | vars x cons | nnz | Primal (s) | HiGHS (s) | Clarabel (s) | SCS (s) | SCIP (s) | obj |
|---|---|---|---|---|---|---|---|---|---|
| lp_50x25 | lp | 50 x 25 | 171 | 0.0010 | 0.0022 | 0.0002 | N/A | N/A | 64.1886 |
| lp_100x50 | lp | 100 x 50 | 576 | 0.0014 | 0.0020 | 0.0007 | N/A | N/A | 121.946 |
| lp_200x100 | lp | 200 x 100 | 2120 | 0.0062 | 0.0039 | 0.0022 | N/A | N/A | 284.367 |
| lp_400x200 | lp | 400 x 200 | 8026 | 0.0333 | 0.0138 | 0.0133 | N/A | N/A | 546.481 |
| qp_50x25 | qp | 50 x 25 | 1445 | 0.0016 | N/A | 0.0004 | N/A | N/A | 12.8014 |
| qp_100x50 | qp | 100 x 50 | 5631 | 0.0071 | N/A | 0.0025 | N/A | N/A | 26.4052 |
| qp_200x100 | qp | 200 x 100 | 22212 | 0.0453 | N/A | 0.0083 | N/A | N/A | 55.9929 |
| milp_40x20 | milp | 40 x 20 | 139 | 0.2933 | N/A | N/A | N/A | 0.1644 | 143.955 |
| milp_60x30 | milp | 60 x 30 | 260 | 1.0270 | N/A | N/A | N/A | 0.2863 | 200.846 |
| milp_80x40 | milp | 80 x 40 | 460 | 3.0903 | N/A | N/A | N/A | 1.1389 | 277.019 |
| socp_40 | socp | 40 x 1 | - | 0.0004 | N/A | 0.0001 | 0.0003 | N/A | 0.707107 |
| socp_120 | socp | 120 x 1 | - | 0.0003 | N/A | 0.0002 | 0.0007 | N/A | 0.707107 |
| socp_200 | socp | 200 x 1 | - | 0.0006 | N/A | 0.0005 | 0.0012 | N/A | 0.707107 |
| sdp_4 | sdp | 4 x 4 | - | 0.0002 | N/A | 0.0002 | 0.0002 | N/A | -9.83233 |
| sdp_5 | sdp | 5 x 5 | - | 0.0002 | N/A | 0.0001 | 0.0002 | N/A | -4.79948 |
| sdp_6 | sdp | 6 x 6 | - | 0.0004 | N/A | 0.0002 | 0.0002 | N/A | -10.2438 |
| sdp_7 | sdp | 7 x 7 | - | 0.0013 | N/A | 0.0003 | 0.0004 | N/A | -8.18613 |
| sdp_8 | sdp | 8 x 8 | - | 0.0007 | N/A | 0.0004 | 0.0005 | N/A | -22.165 |
| sdp_9 | sdp | 9 x 9 | - | 0.0012 | N/A | 0.0006 | 0.0004 | N/A | -17.0048 |
| sdp_10 | sdp | 10 x 10 | - | 0.0015 | N/A | 0.0008 | 0.0007 | N/A | -31.7195 |
| sdp_11 | sdp | 11 x 11 | - | 0.0020 | N/A | 0.0010 | 0.0017 | N/A | -20.5753 |
| sdp_12 | sdp | 12 x 12 | - | 0.0040 | N/A | 0.0017 | 0.0010 | N/A | -44.5174 |
| sdp_13 | sdp | 13 x 13 | - | 0.0039 | N/A | 0.0019 | 0.0011 | N/A | -38.7212 |
| sdp_14 | sdp | 14 x 14 | - | 0.0042 | N/A | 0.0021 | 0.0011 | N/A | -74.2121 |
| sdp_15 | sdp | 15 x 15 | - | 0.0058 | N/A | 0.0023 | 0.0014 | N/A | -51.0706 |
| sdp_16 | sdp | 16 x 16 | - | 0.0141 | N/A | 0.0034 | 0.0015 | N/A | -69.7722 |
| sdp_17 | sdp | 17 x 17 | - | 0.0116 | N/A | 0.0044 | 0.0018 | N/A | -38.8839 |
| sdp_18 | sdp | 18 x 18 | - | 0.0145 | N/A | 0.0044 | 0.0024 | N/A | -78.2696 |
| sdp_19 | sdp | 19 x 19 | - | 0.0419 | N/A | 0.0046 | 0.0021 | N/A | -123.432 |
| sdp_20 | sdp | 20 x 20 | - | 0.0331 | N/A | 0.0065 | 0.0027 | N/A | -140.192 |
| sdp_21 | sdp | 21 x 21 | - | 0.0245 | N/A | 0.0095 | 0.0042 | N/A | -93.0904 |
| sdp_22 | sdp | 22 x 22 | - | 0.0288 | N/A | 0.0110 | 0.0091 | N/A | -149.613 |

</div>

## Status

Last full validation at all four levels: **4905 checks, 0 failures** at `-O0`,
`-O1`, `-O2`, `-O3`. At checkpoint `472dfbb` the suite has **5168 checks**
(2026-10-21): 0 failures at `-O2` and `-O3`, with the default and the opt-in
SOCP route, and under ASan; 2 failures (`T197`, opt-in HSD) at `-O0`, `-O1`,
without FMA, and under UBSan or ASan+UBSan, with no sanitizer diagnostic.
**171/171** examples exit 0. Warning-free build at all four levels.

A subsequent independent review corrects the linear SOC ratio-test sign:
from `(1,0)`, direction `(1,1)` permits step 1, whereas `(-1,1)` stops at
0.5. T287 checks these exact crossings and R+ guards. The updated suite
has **5176 checks**, all passing at O2 with both backends. Full ASan+UBSan
runs report **5174 passes / 2 failures** with both backends: only the two
T197 checks fail, with no sanitizer diagnostic. The eight new geometry
checks also pass at O0/O1/O3, without FMA and under ASan+UBSan.
The earlier full build matrix and 171-example results above belong to
`472dfbb`; they were not all rerun after this follow-up.

**Current validation (2026-10-21, HSD dependent-row fix).** `T197` is fixed:
dependent equalities are removed on the opt-in HSD path, its KKT keeps the
scalar and PSD directions explicit, its stop rule measures the original
model, and the native SDP publisher now reports scalar reduced costs
([details](docs/hsd-dependent-rows.md); `T288` added). The suite has
**5196 checks**: **0 failures** at `-O0`, `-O1`, `-O2`, `-O3`, under ASan,
UBSan and ASan + UBSan, with both SOCP backends — the first fully green
sanitized run — warning-free, no sanitizer diagnostic. **171/171** examples
exit 0 with unchanged output.

**Boundary recovery in the legacy SOCP solver (2026-10-21).** Without FMA
(`-ffp-contract=off`) the 7 `T286` checks used to fail: on a 60-cone model
the LU fallback direction took `mu` from 1e-12 to 5e-17 in one step, a cone
block then sat on its boundary to within rounding, every later step had
length 0 and the solve stalled at `feas_p=1e-8` until the iteration limit.
After two consecutive blocked steps without convergence, the flattened blocks
are now pushed back inside by about ten times the current residual and the
solve continues. The suite now passes **5196/0 at every level, without FMA,
and under ASan + UBSan**, with both SOCP backends. The recovery never fires
in the FMA suite or in the 171 examples, whose output is unchanged; without
FMA it fires only on `T286`'s 60-cone variants. See
[causal evidence](docs/sanitizer-regression-analysis.md).

The SDP predictor/corrector change reuses LU factors within an iteration only
when the equilibrated matrices are identical. T272 covers a mixed SOC/PSD
problem and reoptimization; the focused conic regression run passes 184 checks.
Two before/after traces are identical apart from reuse diagnostics, with half
as many LU factorizations. Five paired timing runs show only small median
reductions with overlapping ranges; no broad speedup is claimed.

## TODO

Open algorithmic work, largest first:

- **Basis warm start.** No real crash basis; `solvebasis` re-solves from scratch
  when the basis handed in does not measure, instead of re-optimizing *from* it.
- **Sparse KKT in the conic path.** The unified conic system is assembled
  densely, which caps problem size. The sparse LDLᵀ with 2×2 pivots exists and is
  validated; it is not wired in yet.
- **Symmetric KKT in the SOCP route**, so the sparse symmetric factorization can
  replace the sparse LU there.
- **Large-scale conic performance.** The second-order SOCP corrector reduces
  the canonical-family gap and direct rotated-cone rows the rotated one.
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
`./out/examples/c/logistic_large 2>&1 | grep -E 'route|cones|mip'`

## License

Apache-2.0. All code is original; see [LICENSE](LICENSE). The 48 example ports
re-implement models published in MOSEK's public documentation. PrimalSolver is
not affiliated with, or endorsed by, MOSEK.
