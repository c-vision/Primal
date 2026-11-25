# PrimalSolver

A convex optimization solver written from scratch in portable C99. No external
dependencies — `libc` and `libm`, nothing else.

It solves **LP, MILP, QP, QCQP, SOCP, SDP, exponential and power cones, and
mixed-integer conic problems**, and it hands back the primal point, the dual
point and a status you can check rather than trust.

[![Sponsor](https://img.shields.io/badge/Sponsor-❤-ea4aaa)](https://github.com/sponsors/c-vision)
[![PyPI](https://img.shields.io/pypi/v/primalsolver)](https://pypi.org/project/primalsolver/)
[![build](https://github.com/c-vision/Primal/actions/workflows/ci.yml/badge.svg)](https://github.com/c-vision/Primal/actions/workflows/ci.yml)
[![julia](https://github.com/c-vision/Primal/actions/workflows/julia.yml/badge.svg)](https://github.com/c-vision/Primal/actions/workflows/julia.yml)

## What it solves

- **LP, QP** — presolve, then simplex or a Mehrotra interior point.
- **MILP, MIQP** — branch & bound with cutting planes, strong branching,
  probing and primal heuristics.
- **SOCP** — primal-dual conic interior point with Nesterov–Todd scaling.
  Primal can eliminate rotated epigraphs with one fixed positive member
  per block, including cases with up to four equalities on the tails.
- **SDP** — Nesterov–Todd interior point on PSD blocks. For a single block
  with fixed diagonal and compatible objective signs, Primal constructs
  the primal and dual solution directly. It can also dualize an affine
  matrix inequality to remove entrywise linking rows and detect
  infeasibility from exact zero diagonal constraints in presolve.
  Specialized log-det barriers solve free-diagonal and small affine SDPs
  in the original scalar coordinates.
- **Exponential and power cones** — native interior point with closed-form
  barriers. Mixed SOC and primal exponential models, and larger models with
  only primal exponential cones, can use a higher-order corrector and a
  sparse KKT solve. For exponential epigraphs with fixed positive scales and
  affine arguments in up to 64 free variables, Primal can use Newton's
  method on the reduced problem, including a single rotated quadratic
  penalty. Positive weighted sums of fixed-scale epigraphs can also be
  reduced to log-sum-exp constraints on the free variables. Primal
  verifies the reconstructed conic solution. For certain weighted
  power-cone loss models, a penalty bound permits a reduction to LP.
- **Mixed-integer conic** — branch and bound with conic relaxations.
  For cardinality-constrained portfolios with separate SOC, EXP and PSD
  risk blocks, Primal can certify a global bound from the individual blocks.
  [Portfolio algorithm](algorithms/mixed_integer_optimization.md#cardinality-constrained-portfolios-with-separate-risk-blocks).
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
[JuMP](https://jump.dev) solver. Install it next to JuMP with Julia 1.10 or
later — no C compiler and no checkout, the prebuilt library comes from
`PrimalSolver_jll`:

```julia
using Pkg
Pkg.add(["JuMP", "PrimalSolver"])
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
[`examples/julia/`](examples/julia/). To work on the package against a local C
build, see [`julia/README.md`](julia/README.md).

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

Measured on Apple Silicon (arm64), macOS 27.2.1, with Apple clang 21.0.0 and
C99 `-O2`, on **2026-11-19**. Times are in seconds. **Bold** marks the lowest
median in each row. Every cell comes from the same session: the median of 15
runs of `bench/bench_all.py` (`tools/bench_repeat.py --runs 15`), and the
aggregator verified that every objective is identical across the 15. Primal
residuals and relative gap were within `1e-8`. The reference columns drift with
the machine by about 10% between sessions — HiGHS and SCIP do not share the
Primal code, so a shift there is the host, not us — and a few rows are won by
less than 10%: read the order within a row, not the absolute value across
tables.

<div>

| instance | class | vars x cons | nnz | Primal (s) | HiGHS (s) | Clarabel (s) | SCS (s) | SCIP (s) | obj |
|---|---|---|---|---|---|---|---|---|---|
| lp_50x25 | lp | 50 x 25 | 171 | **0.000138** | 0.0014452 | 0.0001687 | N/A | N/A | 64.1886 |
| lp_100x50 | lp | 100 x 50 | 576 | **0.000568** | 0.0015196 | 0.0006134 | N/A | N/A | 121.946 |
| lp_200x100 | lp | 200 x 100 | 2120 | **0.001139** | 0.0032946 | 0.0020054 | N/A | N/A | 284.367 |
| lp_400x200 | lp | 400 x 200 | 8026 | **0.005075** | 0.0114544 | 0.0106674 | N/A | N/A | 546.481 |
| qp_50x25 | qp | 50 x 25 | 1445 | **0.00034** | N/A | 0.0003636 | N/A | N/A | 12.8014 |
| qp_100x50 | qp | 100 x 50 | 5631 | **0.001299** | N/A | 0.0020746 | N/A | N/A | 26.4052 |
| qp_200x100 | qp | 200 x 100 | 22212 | **0.005674** | N/A | 0.00668 | N/A | N/A | 55.9929 |
| milp_40x20 | milp | 40 x 20 | 139 | **0.043884** | N/A | N/A | N/A | 0.1355436 | 143.955 |
| milp_60x30 | milp | 60 x 30 | 260 | **0.090254** | N/A | N/A | N/A | 0.2418019 | 200.846 |
| milp_80x40 | milp | 80 x 40 | 460 | **0.196321** | N/A | N/A | N/A | 0.9413622 | 277.019 |
| socp_40 | socp | 60 x 1 | - | **0.000069** | N/A | 0.0000912 | 0.000265 | N/A | 0.707107 |
| socp_120 | socp | 180 x 1 | - | **0.000172** | N/A | 0.000208 | 0.0006825 | N/A | 0.707107 |
| socp_200 | socp | 300 x 1 | - | **0.000266** | N/A | 0.0003956 | 0.0010498 | N/A | 0.707107 |
| sdp_4 | sdp | 4 x 4 | - | **0.000007** | N/A | 0.0001345 | 0.0001691 | N/A | -9.83233 |
| sdp_5 | sdp | 5 x 5 | - | **0.000008** | N/A | 0.0001226 | 0.0001803 | N/A | -4.79948 |
| sdp_6 | sdp | 6 x 6 | - | **0.00001** | N/A | 0.000152 | 0.0002013 | N/A | -10.2438 |
| sdp_7 | sdp | 7 x 7 | - | **0.000013** | N/A | 0.0002342 | 0.0003712 | N/A | -8.18613 |
| sdp_8 | sdp | 8 x 8 | - | **0.000012** | N/A | 0.0004059 | 0.0004301 | N/A | -22.165 |
| sdp_9 | sdp | 9 x 9 | - | **0.000014** | N/A | 0.0005474 | 0.0003368 | N/A | -17.0048 |
| sdp_10 | sdp | 10 x 10 | - | **0.000015** | N/A | 0.0007234 | 0.0006403 | N/A | -31.7195 |
| sdp_11 | sdp | 11 x 11 | - | **0.000019** | N/A | 0.0010424 | 0.0014954 | N/A | -20.5753 |
| sdp_12 | sdp | 12 x 12 | - | **0.000018** | N/A | 0.001674 | 0.0008633 | N/A | -44.5174 |
| sdp_13 | sdp | 13 x 13 | - | **0.000024** | N/A | 0.0018503 | 0.0009397 | N/A | -38.7212 |
| sdp_14 | sdp | 14 x 14 | - | **0.000026** | N/A | 0.0019185 | 0.0009463 | N/A | -74.2121 |
| sdp_15 | sdp | 15 x 15 | - | **0.000028** | N/A | 0.0022253 | 0.0013447 | N/A | -51.0706 |
| sdp_16 | sdp | 16 x 16 | - | **0.000031** | N/A | 0.0030726 | 0.0014801 | N/A | -69.7722 |
| sdp_17 | sdp | 17 x 17 | - | **0.000037** | N/A | 0.0038465 | 0.0018549 | N/A | -38.8839 |
| sdp_18 | sdp | 18 x 18 | - | **0.000043** | N/A | 0.0039952 | 0.0022638 | N/A | -78.2696 |
| sdp_19 | sdp | 19 x 19 | - | **0.000045** | N/A | 0.0039709 | 0.0020052 | N/A | -123.432 |
| sdp_20 | sdp | 20 x 20 | - | **0.000059** | N/A | 0.0057177 | 0.0025415 | N/A | -140.192 |
| sdp_21 | sdp | 21 x 21 | - | **0.000057** | N/A | 0.0080453 | 0.0040277 | N/A | -93.0904 |
| sdp_22 | sdp | 22 x 22 | - | **0.000068** | N/A | 0.0094083 | 0.0087225 | N/A | -149.613 |

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
| 20×4 | **0.017** | 0.188 | 0.09× |
| 40×8 | **0.025** | 0.457 | 0.05× |
| 80×12 | **0.042** | 2.195 | 0.02× |

Measured on 2026-11-19 with `bench/pexp_bench.py --runs 21` (one interleaved
series; each cell is the median of 21 solves).
`bench/pexp_bench.c` generates the instances from a fixed seed. The runner
checks the known objectives (`18.718151648415301`, `47.09337729630259`,
`81.84315908327622`), the residuals and the relative gap at `1e-8`.
Primal timings cover `PRIMAL_optimize`. Clarabel 0.11.1 runs at one thread
and includes solver construction. See the
[reduced Newton derivation](algorithms/exponential_cone.md#adding-a-rotated-quadratic-penalty).

### SOC, exponential, and mixed cone matrix

Measured on **2026-11-19** with the same generated instances in both solvers
(`d=8`, link density `0.50`, one replica). The SOC instances consist of
independent rotated cones with a fixed positive member in each block. EXP
instances have fixed positive scales and affine arguments. Mixed instances
have one rotated SOC in addition to the listed exponential blocks.
Times are the medians of **21 runs with debug tracing disabled** — the trace is
written from inside the solve and inflates the measured milliseconds — while
the route and the iteration count come from one traced run
(`bench/pexp_matrix.py --only family --runs 21`). Primal timings cover
`PRIMAL_optimize`. Clarabel 0.11.1 uses one thread, includes solver
construction and runs with `1e-9` feasibility and gap tolerances. Primal uses
`1e-8` tolerances. The largest public relative gap was `2.5e-10`, and the
runner checks every objective against the Clarabel reference at `1e-6`.

| Family | Blocks | Primal (ms) | Clarabel (ms) | Primal / Clarabel |
|---|---:|---:|---:|---:|
| SOC | 10 | **0.023** | 0.259 | 0.09× |
| SOC | 20 | **0.034** | 0.636 | 0.05× |
| SOC | 40 | **0.051** | 1.096 | 0.05× |
| SOC | 80 | **0.080** | 2.211 | 0.04× |
| EXP | 10 | **0.023** | 0.143 | 0.16× |
| EXP | 20 | **0.026** | 0.207 | 0.13× |
| EXP | 40 | **0.026** | 0.328 | 0.08× |
| EXP | 80 | **0.040** | 0.598 | 0.07× |
| Mixed | 10 | **0.022** | 0.109 | 0.20× |
| Mixed | 20 | **0.026** | 0.170 | 0.15× |
| Mixed | 40 | **0.028** | 0.353 | 0.08× |
| Mixed | 80 | **0.040** | 0.826 | 0.05× |

The [matrix benchmark](bench/pexp_matrix.py) generates the instances and
checks objective agreement. [Detailed measurements](BENCHMARKS.md#matrice-soc-exp)
include solver routes and iteration counts.

With 240 exponential blocks, 48 free coordinates, one rotated SOC and link
density `0.25`, Primal takes **0.241 ms** (median of 21 runs) and Clarabel
**5.186 ms** (median of 5; measured 2026-11-19, same protocol). The
[wider cone benchmark](BENCHMARKS.md#wide-cone-benchmarks)
includes sizes up to 64 free coordinates.

### Public instance sets

The single-run column records the **2026-11-19** comparison. Bold marks
the faster follow-up median. The `beck751` and `rijc781` times are
from two interleaved series of 21 runs on **2026-11-24**, with solver
construction excluded from Clarabel's times. The `mcp100` comparison
uses three interleaved runs on the same date. The HMCR results use
five solves per solver. The `p0201` and `expdesign_D_8_4` comparisons
use five interleaved solves on **2026-11-25**, with MIP gap set to zero
and `1e-8`, respectively. The portfolio comparisons use five Primal
and two Pajarito solves per instance on **2026-11-25**, after a warm-up
for each solver. The remaining rows use five solves, with three warm
batches for the Julia references.

| instance | set | class | vars x cons | fastest in single run | Primal median (s) | reference median (s) | obj |
|---|---|---|---:|---|---:|---:|---:|
| ex1223a | CBLIB | exp + int | 17 x 60 | Primal | **0.0110** | 0.0167 (Pajarito) | 4.57958 |
| beck751 | CBLIB | exp | 80 x 59 | Primal | **0.000029** | 0.000331 (Clarabel) | 7.50095 |
| rijc781 | CBLIB | exp | 24 x 17 | Clarabel | **0.000020** | 0.000056 (Clarabel) | -4.41429 |
| mcp100 | SDPLIB | SDP | 100 x 5050 | Pajarito/Hypatia | **0.05636** | 0.16674 (Clarabel) | 226.15735 |
| HMCR-n20-m400 | CBLIB | power | 2022 x 1603 | Clarabel | **0.002946** | 0.015535 (Clarabel) | 0.0345743 |
| HMCR-n20-m800 | CBLIB | power | 4022 x 3203 | SCS | **0.005703** | 0.104136 (SCS) | 0.0390837 |
| p0201 | MIPLIB | MILP | 201 x 133 | SCIP | **0.004656** | 0.297278 (SCIP) | 7615 |
| expdesign_D_8_4 | CBLIB | exp + PSD + int | 61 x 148 | Pajarito | **0.000938** | 0.045817 (Pajarito) | 0.8433069245 |
| port_12_9_3_a_1 | CBLIB | SOC + EXP + PSD + int | 90 x 169 | Pajarito | **0.004568** | 17.246444 (Pajarito) | -0.0332185523 |
| port_12_9_3_b_1 | CBLIB | SOC + EXP + PSD + int | 91 x 170 | Pajarito | **0.003490** | 6.600786 (Pajarito) | -0.0347049507 |
| HS21 | Maros-Meszaros | QP | 2 x 1 | Primal | **0.000031** | 0.000041 (Clarabel) | 0.0400007 |
| afiro | Netlib | LP | 32 x 27 | Primal | 0.000131 | **0.000116 (Clarabel)** | -464.753 |
| sc50a | Netlib | LP | 48 x 50 | Primal, HiGHS, Clarabel (tie) | 0.000347 | **0.000221 (HiGHS)** | -64.5751 |

Primal timings cover `PRIMAL_optimize` on one thread, excluding file reading.
The benchmark report includes the full comparison and the follow-up solves on
`PRIMAL1`, `LogExpCR-n20-m400`, `control1` and `theta1`.
[49 public instances](BENCHMARKS.md#istanze-pubbliche) ·
[Repeated comparison](BENCHMARKS.md#vittorie-confermate-sulle-istanze-pubbliche) ·
[November 24 results](BENCHMARKS.md#verifica-mirata-del-2026-11-24) ·
[November 22 results](BENCHMARKS.md#verifica-mirata-del-2026-11-22) ·
[November 23 results](BENCHMARKS.md#verifica-mirata-del-2026-11-23) ·
[SDPLIB results for mcp100, qap5 and hinf1](BENCHMARKS.md#verifica-sdplib-del-2026-11-24) ·
[CBLIB power results](BENCHMARKS.md#verifica-power-cblib-del-2026-11-24) ·
[MIPLIB p0201 results](BENCHMARKS.md#verifica-miplib-p0201-del-2026-11-25) ·
[CBLIB experimental design results](BENCHMARKS.md#verifica-cblib-expdesign-del-2026-11-25) ·
[CBLIB portfolio results](BENCHMARKS.md#verifica-cblib-portafogli-del-2026-11-25)

In a separate infeasibility comparison on **2026-11-23**, both Primal and
Clarabel identified `infeas_clean_10_10_1` (CBLIB). Median time to verdict
was **0.021 ms** for Primal and **3.718 ms** for Clarabel. Primal reports
the problem status from a sequence of exact face reductions. This route
leaves the dual-ray getter unavailable.
[Infeasibility results and timing protocol](BENCHMARKS.md#inammissibilità-sdp-cblib)

## Documentation

The [algorithm guide](algorithms/README.md) follows the numerical methods
from model formulation through solution verification. See the
[API guide](docs/api.md) and [I/O formats](docs/io-formats.md) for reference
material.

## License

Apache-2.0. All code is original; see [LICENSE](LICENSE). The examples that port
models from MOSEK's public documentation re-implement them. PrimalSolver is not
affiliated with, or endorsed by, MOSEK.
