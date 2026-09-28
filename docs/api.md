# API guide

**Source:** `primal.h`

547 public functions, all prefixed `PRIMAL_`. This page is the map: what the
groups are, how to call them, and where the exact signature lives. The
authoritative list of parameters, ranges and error codes is the header
`primal.h` itself, plus `PRIMAL_getcodedesc` / `PRIMAL_getlasterror` at
runtime.

## The shape of a call

```c
#include "primal.h"

PRIMALenv_t  env;
PRIMALtask_t task;
PRIMALrescodee rc;

PRIMAL_makeenv(&env, NULL);
PRIMAL_maketask(env, maxcon, maxvar, &task);   /* pre-allocates maxvar/maxcon */
```

`PRIMAL_maketask` **pre-allocates** `maxvar` and `maxcon`. Do not also call
`PRIMAL_appendvars` / `PRIMAL_appendcons` for the same variables and
constraints — that doubles the dimensions and the solution buffers no longer
match the model.

A task is created with or without dimensions:

```c
PRIMAL_maketask(env, 0, 0, &task);                 /* empty, grow with append* */
PRIMAL_appendvars(task, nvar);
PRIMAL_appendcons(task, ncon);
```

Everything that **adds a visible variable** extends `numvar`, and the new count
must be read back before the solution getters are used: `appendacc`,
`appendafes` and the disjunctive encoder all grow the model.

## Model construction

| Group | Calls |
|---|---|
| Bounds | `putvarbound`, `putconbound`, `putvarboundlist`, `putconboundlist`, `chgvarbound`, `chgconbound`, `put*boundslice*` |
| Objective | `putcj`, `putclist`, `putcfix`, `putobjsense`, `putqobj`, `putqobjij` |
| Matrix | `putarow`, `putacol`, `putaij`, `putarowlist`, `putacollist`, `inputdata` |
| Quadratic rows | `putqcon`, `putqconk`, `putqconij` |
| Cones | `appendcone`, `appendconeseq`, `appendprimalgeomeanconedomain` |
| Bar variables | `appendbarvars`, `putbarablockij`, `putbarcblockij`, `putbaraij`, `putbarcj` |
| Integrality | `putvartype`, `putvartypelist` |
| Names | `putvarname`, `putconname`, `putbarname`, `putconename`, `putobjname` |
| I/O | `readdata`, `writedata`, `readtask`, `writetask`, `linkfiletotask` |
| Constraints | `appendafes`, `appendacc`, `appenddjcs`, `appendcones`, `appendsos` |

A `puta*` call is a **replace** of that row or column, not an append: the row is
cleared and rewritten. Writing the same `(i, j)` twice inside one call stores
two entries and the operator `getaij` reports their sum; the storage readers
report both entries. See [certificates](certificates.md) for why those two
readings are deliberately different.

## Solving and reading the solution

```c
rc = PRIMAL_optimize(task);

if (rc == PRIMAL_RES_OK) {
    double x[2], obj;
    PRIMALsolstae sta;
    PRIMAL_getsolsta(task, PRIMAL_SOL_BAS, &sta);
    PRIMAL_getxx(task, PRIMAL_SOL_BAS, x);
    PRIMAL_getprimalobj(task, PRIMAL_SOL_BAS, &obj);
}
```

| Call | Returns |
|---|---|
| `getxx`, `gety`, `getxc` | point, row multipliers, row activities |
| `getslc`, `getsuc`, `getslx`, `getsux` | slacks and reduced costs, by sign |
| `getprimalobj`, `getdualobj` | `pobj`, `dobj` |
| `getprimalinfeas`, `getdualinfeas` | violation of the model by the published point |
| `getpviolcon`, `getpviolvar`, `getpviolcones`, `getpviolbarvar` | per-index violation |
| `getdviol*`, `getpvioldjc`, `getpviolacc` | dual-side and higher-order violations |
| `getbarxj`, `getbarsj` | PSD blocks, primal and dual |
| `getdualray`, `getprimalray` | Farkas vectors |
| `getsolsta`, `getprosta` | solution and problem status |

Every solution getter answers `PRIMAL_RES_ERR_ARG` when there is no point to
deliver. See [certificates](certificates.md) — this is a contract, not a
detail.

### Information items

`getdouinf`, `getintinf` and `getlintinf` (and the `getna*inf` forms, by
name) accept every information-item id of the reference. The measured ones:

| Item | Value |
|---|---|
| `DINF_OPTIMIZER_TIME`, `DINF_SIM_TIME`, `DINF_MIO_TIME` | CPU seconds of the last `optimize` |
| `DINF_*_OBJ`, `DINF_MIO_OBJ_*`, `DINF_SOL_ITR_*`, `DINF_SOL_ITG_*` | objectives, gaps, violations and norms of the published point |
| `IINF_INTPNT_ITER` | iterations of the linear/quadratic interior point |
| `IINF_SIM_PRIMAL_ITER`, `IINF_SIM_DUAL_ITER` | pivots of the primal and of the dual simplex |
| `IINF_MIO_NUM_SOLVED_NODES`, `IINF_MIO_NUM_RELAX`, `IINF_MIO_NUM_BRANCH` | branch-and-bound nodes solved, relaxations solved, branchings |
| `IINF_ANA_PRO_*`, `IINF_OPT_NUM*`, `IINF_RD_*`, `LIINF_RD_*` | model shape |
| `IINF_OPTIMIZE_RESPONSE`, `IINF_SOL_*_PROSTA`, `IINF_SOL_*_SOLSTA` | last response code and statuses |

Every other id is accepted and answers 0. The iteration and node counters
describe the last `optimize` call only and count every run of that optimizer
made on the task's behalf: the route itself, the node relaxations of
branch-and-bound (probing and strong branching included), the LP masters of
the cut loops and each engine of the concurrent optimizer. The conic interior
point (SOCP, SDP, exponential and power cones) is not counted.

## Reading the model back

The model is fully readable: coefficients, quadratic parts, bar terms, bounds,
names, and six independent name tables (variables, constraints, bars, cones,
disjunctions, ACCs) plus the objective's own slot.

The rule that governs every one of those readers: **a refusal writes nothing.**
A call that cannot fit its result into the buffer you provided answers
`ERR_ARG` without having touched your buffer or your `*numret`. A truncated list
that returns `OK` is a different model, and the caller cannot tell the two
apart.

Passing `NULL` for both the buffer and the count is the documented way to ask
"how much do I need".


## Parameters

31 parameters are declared in one declarative table; each row carries its type,
field, default, accepted range and name, and the defaults, validation, getters
and shadow-task copies are all derived from that one row. About sixteen of them
are wired to behaviour — presolve, scaling, iteration limits, the MIP
tolerances, the near-optimality factor, thread count. The rest are accepted,
range-checked and readable, and do not yet change the solve; that is a declared
gap, and adding a row to the table is the whole cost of wiring one.

```c
PRIMAL_putintparam(task, PRIMAL_IPAR_MIP_MAX_NODES, 10000);
PRIMAL_putdouparam(task, PRIMAL_DPAR_MIP_TOL_REL_GAP, 1e-5);
PRIMAL_getparaminfo(task, PRIMAL_IPAR_PRESOLVE, &name, &len, &def, &lo, &hi);
```

Reading by name is available: `putnaintparam`, `putnadouparam`,
`getparamname`, `getnaintparam`, `getnadouparam`.

## Callbacks

| Call | Purpose |
|---|---|
| `putcallbackfunc` | progress: every iteration, every round of the cut loop |
| `putresponsefunc` | BEGIN/END of optimize, read and write |
| `putexitfunc` | cleanup hook on the environment |
| `linkfunctotaskstream` / `linkfiletotask` | log routing |

`GMB_DBG=1` prints the diagnostics described in the README — the `[route]`,
`[cones]`, `round=` and `[mip]` lines — to **stderr**, so a sample's own output
on stdout stays clean.

## Memory

`PRIMAL_deletetask` and `PRIMAL_deleteenv` release everything. A task holds no
internal pool: the debug allocators are plain `calloc`/`free` with counting, so
their reports match the plain ones exactly.

## Limits

- **No per-function reference page.** This is the map; the signature, the
  parameter meaning and the accepted range are in `primal.h` and at runtime via
  `getcodedesc`.
- **Concurrency is documented for one solve only.** `PRIMAL_IPAR_NUM_THREADS`
  parallelizes *inside* a solve ([branch and bound](branch-and-bound.md)).
  Driving a single task from two threads at once is not supported and carries
  no lock; separate tasks and separate environments are independent.
- **About half the declared parameters do not change the solve yet** — accepted,
  range-checked and readable.
- No string-valued parameters, so `getstrparam` answers `ERR_ARG`.

## Where to look next

- `samples/mosek_comparison/*.c` — 48 small ports, each with its expected
  optimum checked
- `samples/*.c`, `samples/finance/*.c`, `samples/books/*.c` — the larger models
- [certificates](certificates.md) — what a status guarantees
- [I/O formats](io-formats.md) — MPS, LP, OPF, CBF
