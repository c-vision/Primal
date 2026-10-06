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

171 runnable examples live under `c_examples/`, each checking its own result: 48
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
cone and an SDP — live in [`lp_examples/`](lp_examples/README.md), each with its
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
built by the CI. **53 runnable examples** live in
[`python_examples/`](python_examples/README.md) — installation, usage, and one
line per example.

## Build

```sh
make               # gcc -std=c99 -Wall -Wextra -pedantic -O2, zero warnings
make test          # reliability suite: 5033 checks
make run-samples   # the 171 examples
make clean         # remove out/
```

Everything the build produces goes to `out/` (gitignored). **Portability**: POSIX
C99 with `gcc` or `clang` on Linux, macOS and the BSDs. There is **no MSVC
project and no `nmake` path** — on Windows use MinGW
([how to build on Windows](docs/windows.md)), or compile the `LIBSRCS` list from
the Makefile with any C99 compiler. Clean under AddressSanitizer +
UndefinedBehaviorSanitizer on all samples; on the suite, two checks (`T197`,
the HSD opt-in) are known red under the sanitizers — see issue #11.

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

- **Clarabel** is the closest open conic IPM and is **~7× faster** in the
  recorded SOCP case (0.0004 s against 0.0027 s for `socp_200`: 100 cones,
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
references (`clarabel`, `scs`, `highspy`, `pyscipopt`, `out/bench/scip_cbf` from
`make bench-scip` against SCIP and SCIP-SDP, and Julia with the `bench/julia`
project for Pajarito and Hypatia; a solver that is absent shows `N/A`).
`bench/bench_all.py` prints this table and cross-checks every objective against
the analytic value; it exits non-zero on a mismatch. LP/QP/MILP are the
generated MPS instances, SOCP/SDP are the closed-form families of
`bench/conic_bench.c` and `bench/sdp_sweep.c`; the SCIP column is SCIP-SDP on
the SDP rows. The SDP sweep reaches **d = 22**. Timings: one run on an Intel
i7-12700H (Linux, gcc 14), HiGHS 1.15.1, Clarabel 0.11.1, SCS 3.3.1, SCIP 10.0.2
built with PaPILO 3.0.0 and Ipopt 3.14.19 (C API on CBF models, PySCIPOpt 6.2.1
built against it on MPS files), SCIP-SDP 4.4.0 with DSDP 5.8, Pajarito 0.8.3
(integer models, over HiGHS 1.26 and Clarabel) and Hypatia 0.11.0 (continuous
ones) on Julia 1.13.1, timed on a second, already-compiled run.

That script times **every** solver in the table (PrimalSolver, HiGHS, Clarabel,
SCS, SCIP, Pajarito/Hypatia) and cross-checks each objective. It takes **one run per cell**, so the
table it prints is a single-run snapshot. For anything finer than ~10% use
`make bench-repeat`, which runs it N times, medians every cell and checks that
every objective is identical across the runs.

Each solver runs with one fixed configuration on every instance, its defaults
except where a default gave wrong answers here: Clarabel without chordal
decomposition (with it, 0.11.1 reports `Solved` at 18.0568 on SDPLIB `control1`,
optimum 17.7846; the cost is `mcp100`, 0.39 s to 13 s); SCS to `eps` 1e-6 (its
default 1e-4 lands 1e-4 to 6e-4 off); SCIP-SDP at a 1e-7 feasibility tolerance
(at its default 1e-5 it returns near-feasible "optima" for infeasible SDPs and
0.7% too good on `port`), using its LP approach on models that also have
nonlinear cones (its SDP relaxations leave those out and never move the bound);
Pajarito with Clarabel for its conic subproblems (with Hypatia, its outer
approximation cuts off the optimum of `expdesign_D_12_6` and `_16_8`).

The SDP eigenvalue kernel (`dmat_eig_jacobi`, `min_eig`) now updates the
symmetric matrix one-sided (half the arithmetic per Jacobi rotation). The
`d = 4..22` sweep is ~**1.4×** faster overall (interleaved base/new; `d = 22`
~1.5×, `d = 20` ~2.9×). The gain is **not uniform per size**: the ~1e-14 change
in the eigenvalues can move which trajectory stalls, so one size can be slower
(`d = 14`). That is the same route non-reproducibility `T81`/`T91` already
assert as an outcome.

**Four changes account for the SDP speed-up**. On the Apple build they were
measured on, the SDP family went **23.3% faster** (0.2442 s ->
0.1873 s, medians of 5), and **no instance is slower** (27 faster, 0 slower);
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

<div style="font-size: 0.9em">

| instance | class | vars x cons | nnz | Primal (s) | HiGHS (s) | Clarabel (s) | SCS (s) | SCIP (s) | Pajarito/Hypatia (s) | obj |
|---|---|---|---|---|---|---|---|---|---|---|
| lp_50x25 | lp | 50 x 25 | 171 | 0.0037 | 0.0395 | **0.0009** | 0.0035 | 0.0035 | 0.0033 | 64.1886 |
| lp_100x50 | lp | 100 x 50 | 576 | 0.0134 | **0.0021** | 0.0026 | 0.0081 | 0.0053 | 0.0068 | 121.946 |
| lp_200x100 | lp | 200 x 100 | 2120 | 0.0527 | **0.0070** | 0.0071 | 0.0183 | 0.0230 | 0.0324 | 284.367 |
| lp_400x200 | lp | 400 x 200 | 8026 | 0.1826 | **0.0293** | 0.0625 | 0.0518 | 0.1728 | 0.0964 | 546.481 |
| qp_50x25 | qp | 50 x 25 | 1445 | 0.0126 | 0.0028 | **0.0010** | 0.0018 | 0.2322 | 0.0042 | 12.8014 |
| qp_100x50 | qp | 100 x 50 | 5631 | 0.0161 | 0.0056 | 0.0040 | **0.0032** | 0.6708 | 0.0103 | 26.4052 |
| qp_200x100 | qp | 200 x 100 | 22212 | 0.1488 | 0.0773 | 0.0328 | **0.0106** | 2.5707 | 0.0461 | 55.9929 |
| milp_40x20 | milp | 40 x 20 | 139 | 0.3621 | 0.2756 | N/A | N/A | 0.3456 | **0.1503** | 143.955 |
| milp_60x30 | milp | 60 x 30 | 260 | 1.4953 | **0.2399** | N/A | N/A | 0.6365 | 0.2412 | 200.846 |
| milp_80x40 | milp | 80 x 40 | 460 | 3.5450 | 0.5603 | N/A | N/A | 2.0022 | **0.5208** | 277.019 |
| socp_40 | socp | 40 x 1 | - | 0.0012 | N/A | **0.0002** | 0.0006 | 0.0363 | 0.0017 | 0.707107 |
| socp_120 | socp | 120 x 1 | - | 0.0072 | N/A | **0.0015** | 0.0033 | 0.0809 | 0.0125 | 0.707107 |
| socp_200 | socp | 200 x 1 | - | 0.0230 | N/A | **0.0019** | 0.0041 | 0.1034 | 0.0403 | 0.707107 |
| sdp_4 | sdp | 4 x 4 | - | **0.0008** | N/A | 0.0594 | **0.0008** | 0.0055 | 0.0025 | -9.83233 |
| sdp_5 | sdp | 5 x 5 | - | 0.0014 | N/A | **0.0005** | 0.0010 | 0.0059 | 0.0016 | -4.79948 |
| sdp_6 | sdp | 6 x 6 | - | 0.0628 | N/A | **0.0007** | 0.0012 | 0.0048 | 0.0018 | -10.2438 |
| sdp_7 | sdp | 7 x 7 | - | 0.8369 | N/A | **0.0007** | 0.0011 | 0.0113 | 0.0023 | -8.18613 |
| sdp_8 | sdp | 8 x 8 | - | 0.0023 | N/A | **0.0016** | 0.0023 | 0.0114 | 0.0021 | -22.165 |
| sdp_9 | sdp | 9 x 9 | - | 0.0030 | N/A | **0.0017** | 0.0023 | 0.0102 | 0.0025 | -17.0048 |
| sdp_10 | sdp | 10 x 10 | - | 0.0040 | N/A | **0.0023** | 0.0026 | 0.0140 | 0.0030 | -31.7195 |
| sdp_11 | sdp | 11 x 11 | - | 0.0054 | N/A | 0.0043 | 0.0097 | 0.0140 | **0.0035** | -20.5753 |
| sdp_12 | sdp | 12 x 12 | - | 0.0088 | N/A | 0.0054 | **0.0034** | 0.0170 | 0.0045 | -44.5174 |
| sdp_13 | sdp | 13 x 13 | - | 4.4306 | N/A | 0.0060 | **0.0047** | 0.0188 | 0.0053 | -38.7212 |
| sdp_14 | sdp | 14 x 14 | - | 6.1125 | N/A | 0.0068 | **0.0062** | 0.0201 | 0.0063 | -74.2121 |
| sdp_15 | sdp | 15 x 15 | - | 6.2201 | N/A | 0.0108 | **0.0063** | 0.0239 | 0.0078 | -51.0706 |
| sdp_16 | sdp | 16 x 16 | - | 7.7997 | N/A | 0.0181 | **0.0074** | 0.0278 | 0.0083 | -69.7722 |
| sdp_17 | sdp | 17 x 17 | - | 10.2622 | N/A | 0.0202 | **0.0078** | 0.0365 | 0.0109 | -38.8839 |
| sdp_18 | sdp | 18 x 18 | - | 13.9289 | N/A | 0.0258 | **0.0103** | 0.0231 | 0.0139 | -78.2696 |
| sdp_19 | sdp | 19 x 19 | - | 18.3836 | N/A | 0.0196 | **0.0105** | 0.0364 | 0.0283 | -123.432 |
| sdp_20 | sdp | 20 x 20 | - | 25.4804 | N/A | 0.0306 | **0.0174** | 0.0442 | 0.0180 | -140.192 |
| sdp_21 | sdp | 21 x 21 | - | 32.6378 | N/A | 0.0452 | **0.0156** | 0.0535 | 0.0204 | -93.0904 |
| sdp_22 | sdp | 22 x 22 | - | 44.2064 | N/A | 0.0520 | 0.0288 | 0.0594 | **0.0264** | -149.613 |

</div>

### Public instances

`python3 bench/fetch_instances.py` downloads a few small instances from each
standard set (CBLIB, SDPLIB, Maros–Meszaros, Netlib, MIPLIB 2017), and
`bench_all.py` then prints this second table. CBLIB supplies the models that
mix PSD, exponential and power cones with integer variables. Every solver,
PrimalSolver included, is capped at 60 s. SCIP builds the CBF models through
its C API (`bench/scip_cbf.c`, `make bench-scip`), with SCIP-SDP for those with
PSD parts. `obj` is the set's published optimum where it has one (SDPLIB,
Netlib, MIPLIB; checked to the precision printed there), otherwise the value
the solvers agree on; for `expdesign_D_12_6` and `_16_8` it is SCIP-SDP's,
whose points we checked against the model. A ✗ marks an answer that
disagrees with it; `fail` is a solve that ended without one (including
Clarabel's reduced-accuracy `Almost*` statuses); `N/A` is a class the solver
does not take. The two ✗ left are on the ill-conditioned `hinf1`, where
Hypatia (2.03267) and SCS (2.03349) stop short; feasible points reach 2.03262
and below.

<div style="font-size: 0.9em">

| instance | set | class | vars x cons | Primal (s) | HiGHS (s) | Clarabel (s) | SCS (s) | SCIP (s) | Pajarito/Hypatia (s) | obj |
|---|---|---|---|---|---|---|---|---|---|---|
| port_12_9_3_a_1 | cblib | soc+exp+psd+int | 90 x 169 | >60s | N/A | N/A | N/A | **6.1503** | 7.5615 | -0.0332208 |
| port_12_9_3_b_1 | cblib | soc+exp+psd+int | 91 x 170 | >60s | N/A | N/A | N/A | **4.2889** | 8.0163 | -0.0347061 |
| port_16_12_4_b_1 | cblib | soc+exp+psd+int | 120 x 232 | >60s | N/A | N/A | N/A | **20.5534** | >60s | -0.0440182 |
| expdesign_D_8_4 | cblib | exp+psd+int | 61 x 148 | fail | N/A | N/A | N/A | 0.5673 | **0.0547** | 0.843307 |
| expdesign_D_12_6 | cblib | exp+psd+int | 127 x 311 | fail (rc=1007) | N/A | N/A | N/A | 28.4016 | **3.6698** | -0.520253 |
| expdesign_D_16_8 | cblib | exp+psd+int | 217 x 534 | fail (rc=1007) | N/A | N/A | N/A | >60s | fail (OTHER_ERROR) | 2.93132 |
| kpart_diw.15.4.29 | cblib | psd+int | 105 x 255 | fail (rc=1002) | N/A | N/A | N/A | **0.2699** | >60s | -95 |
| rt_2x4_3bars | cblib | psd+int | 64 x 190 | fail (rc=1002) | N/A | N/A | N/A | **1.8447** | 29.6921 | 3.08114 |
| clsq_random_32_2_a | cblib | psd+int | 33 x 979 | >60s | N/A | N/A | N/A | **0.4501** | 12.9919 | 7.14855 |
| expdesign_A_8_4 | cblib | psd+int | 29 x 134 | 0.7762 | N/A | N/A | N/A | 0.0746 | **0.0376** | 5.93682 |
| expdesign_E_8_4 | cblib | psd+int | 26 x 50 | 0.2820 | N/A | N/A | N/A | **0.0174** | 0.0387 | -0.344045 |
| syn05m | cblib | exp+int | 24 x 65 | 1.1054 | N/A | N/A | N/A | 0.0860 | **0.0791** | -837.732 |
| syn10m | cblib | exp+int | 42 x 120 | 14.3409 | N/A | N/A | N/A | **0.0315** | 0.0436 | -1267.35 |
| rsyn0805m | cblib | exp+int | 174 x 537 | >60s | N/A | N/A | N/A | **1.1153** | 2.0208 | -1296.12 |
| batchdes | cblib | exp+int | 25 x 73 | 1.0885 | N/A | N/A | N/A | 0.0843 | **0.0272** | 167428 |
| ex1223a | cblib | exp+int | 17 x 60 | 0.0354 | N/A | N/A | N/A | 0.0717 | **0.0188** | 4.57958 |
| synthes2 | cblib | exp+int | 16 x 47 | 0.7448 | N/A | N/A | N/A | 0.0435 | **0.0342** | 73.0353 |
| beck751 | cblib | exp | 80 x 59 | 0.2706 | N/A | **0.0005** | 0.0015 | 0.0425 | 0.0040 | 7.50095 |
| demb761 | cblib | exp | 131 x 93 | 3.7235 | N/A | **0.0008** | 0.4400 | 0.2099 | 0.0050 | 22.3109 |
| rijc781 | cblib | exp | 24 x 17 | 0.0213 | N/A | **0.0003** | 0.0011 | 0.1371 | 0.0020 | -4.41429 |
| gp_dave_1 | cblib | exp | 705 x 988 | >60s | N/A | **0.0117** | 0.1266 | 0.3689 | 0.1399 | 5.50653 |
| LogExpCR-n20-m400 | cblib | exp | 2022 x 1603 | >60s | N/A | **0.0528** | 0.0558 | 0.8083 | 1.5266 | 0.0164814 |
| HMCR-n20-m400 | cblib | pow | 2022 x 1603 | >60s | N/A | **0.0444** | 0.1136 | 0.0569 | 1.2311 | 0.0345743 |
| HMCR-n20-m800 | cblib | pow | 4022 x 3203 | >60s | N/A | fail | 0.1951 | **0.1550** | 12.2301 | 0.0390837 |
| infeas_clean_10_10_1 | cblib | sdp-infeas | 55 x 10 | fail (rc=1007) | N/A | **0.0045** | >60s | 0.0157 | fail (SLOW_PROGRESS) | infeasible |
| infeas_clean_10_10_3 | cblib | sdp-infeas | 55 x 10 | fail (rc=1007) | N/A | fail | 6.4562 | **0.0104** | fail (SLOW_PROGRESS) | infeasible |
| weak_clean_10_10_3 | cblib | sdp-infeas | 55 x 10 | fail (rc=1007) | N/A | fail | >60s | **0.0070** | fail (SLOW_PROGRESS) | infeasible |
| control1 | sdplib | sdp | 21 x 70 | fail (rc=1007) | N/A | **0.0045** | >60s | 0.0243 | 0.0063 | 17.7846 |
| hinf1 | sdplib | sdp | 13 x 41 | fail (rc=1007) | N/A | fail | 0.1673 ✗ | **0.0122** | 0.0050 ✗ | 2.0326 |
| theta1 | sdplib | sdp | 104 x 1275 | >60s | N/A | 1.7628 | **0.0682** | 0.1183 | 0.0770 | 23 |
| truss1 | sdplib | sdp | 6 x 19 | 0.0373 | N/A | **0.0009** | 0.0024 | 0.0091 | 0.0035 | -9 |
| qap5 | sdplib | sdp | 136 x 351 | >60s | N/A | 0.0845 | **0.0422** | 0.1412 | fail (ALMOST_OPTIMAL) | -436 |
| mcp100 | sdplib | sdp | 100 x 5050 | >60s | N/A | 16.0713 | 0.6803 | **0.1803** | 0.1881 | 226.157 |
| HS21 | maros-meszaros | qp | 2 x 1 | **0.0001** | 0.0004 | **0.0001** | 0.0003 | 0.0012 | 0.0023 | 0.0400007 |
| HS118 | maros-meszaros | qp | 15 x 17 | 0.0024 | 0.0007 | **0.0004** | 0.0012 | 0.0090 | 0.0026 | 664.82 |
| QAFIRO | maros-meszaros | qp | 32 x 27 | 0.0033 | 0.0012 | **0.0007** | 0.0011 | 0.0109 | 0.0031 | -1.59078 |
| DUAL1 | maros-meszaros | qp | 85 x 1 | 0.0165 | **0.0028** | 0.0056 | 0.0037 | 0.1576 | 0.0092 | 0.035013 |
| PRIMAL1 | maros-meszaros | qp | 325 x 85 | fail (rc=1007) | 0.0252 | 0.0065 | **0.0036** | 1.4130 | 0.0516 | -0.035013 |
| CVXQP1_S | maros-meszaros | qp | 100 x 50 | 0.0244 | 0.0023 | **0.0019** | 0.0045 | 0.1822 | N/A | 11590.7 |
| afiro | netlib | lp | 32 x 27 | **0.0003** | 0.0007 | 0.0006 | 0.0015 | 0.0017 | 0.0021 | -464.753 |
| adlittle | netlib | lp | 97 x 56 | 0.0036 | 0.0012 | **0.0007** | 0.0220 | 0.0096 | 0.0050 | 225495 |
| blend | netlib | lp | 83 x 74 | 0.0137 | 0.0046 | **0.0030** | 0.0094 | 0.0088 | 0.0032 | -30.8121 |
| sc50a | netlib | lp | 48 x 50 | 0.0024 | 0.0020 | **0.0015** | 0.0098 | 0.0037 | 0.0277 | -64.5751 |
| share2b | netlib | lp | 79 x 96 | 0.0170 | 0.0050 | **0.0024** | 0.0129 | 0.0086 | 0.0043 | -415.732 |
| stocfor1 | netlib | lp | 111 x 117 | 0.0080 | 0.0025 | **0.0020** | 0.0108 | 0.0046 | 0.0048 | -41132 |
| flugpl | miplib | milp | 18 x 18 | 1.5784 | 0.1278 | N/A | N/A | **0.0147** | 0.0960 | 1.2015e+06 |
| p0201 | miplib | milp | 201 x 133 | >60s | 1.1003 | N/A | N/A | **0.4851** | 0.8088 | 7615 |
| gt2 | miplib | milp | 188 x 29 | >60s | **0.0574** | N/A | N/A | 0.0630 | 0.0658 | 21166 |
| pk1 | miplib | milp | 86 x 45 | >60s | >60s | N/A | N/A | >60s | >60s | 11 |

</div>

## Status

Last full validation at all four levels: **4905 checks, 0 failures** at `-O0`,
`-O1`, `-O2`, `-O3`. The suite has since grown to **5033 checks, 0 failures**
(default `-O2`, 2026-10-02). **171/171** examples
exit 0. Warning-free. ASan + UBSan clean on the samples (the sanitized suite
exits 1 on the two `T197` checks, issue #11).

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
`./out/c_examples/logistic_large 2>&1 | grep -E 'route|cones|mip'`

## License

Apache-2.0. All code is original; see [LICENSE](LICENSE). The 48 example ports
re-implement models published in MOSEK's public documentation. PrimalSolver is
not affiliated with, or endorsed by, MOSEK.
