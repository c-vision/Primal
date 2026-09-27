# Certificates

**Source:** `primal_solution.c`, `primal_verdict.c`, `simplex.c`, `primal_solio.c`

The design rule of this solver in one sentence: **a status that names a
certificate is not published unless the vector that proves it is published
too.** Everything below follows from that.

## Optimality: the KKT certificate

For continuous problems with a published dual result, the caller can check
the following conditions through public getters. Integer incumbents have no
continuous dual objective; use the MIP search bound instead.

```
c + Qx + A'y + z = 0        stationarity
Ax + s = b,  x, s ≥ 0       primal feasibility
A'y + z ∈ K*                 dual feasibility
x_j · z_j = 0                complementarity
|pobj − dobj| ≈ 0            strong duality
```

Dual sign convention, for the min-normalized problem:

| Sign | Meaning |
|---|---|
| `y_i > 0` | row `i` is active at its **upper** bound |
| `y_i < 0` | row `i` is active at its **lower** bound |
| `z_j > 0` | variable `j` is at its **upper** bound |
| `z_j < 0` | variable `j` is at its **lower** bound |

The dual objective is `dobj = cfix − Σ Y_i b_i^act − Σ Z_j xb_j^act − ½ x'Qx`.
`cfix` is a term of the objective **as written**, so it is added outside the
`−1` factor that normalizes a maximization: multiplied by it, the dual
objective misses by exactly `2·cfix` on a max problem.

## Infeasibility: the dual ray

`PRIMAL_getdualray(y)` returns `numcon` entries with

```
A'y ≤ 0  on every nonnegative variable,   Σ_i y_i·b_i^act > 0
```

— a nonnegative combination of the constraints that no feasible point can
satisfy.

## Unboundedness: the primal ray

`PRIMAL_getprimalray(ρ)` returns `numvar` entries with

```
ρ ≥ 0,   A·ρ = 0,   c'ρ < 0   (for a minimization)
```

— a recession direction along which the objective improves without limit.
For a convex QP, the direction must also satisfy `Qρ = 0`.

Both rays are normalized to `max |·| = 1`, so the tolerances sit on the scale
of the model rather than on the scale of whatever internal representation
produced them.

## The measurement gate

A ray is written out only after it has been **measured** in the form the solver
actually solved. An interior-point route has no witness, so its last iterate is
offered to the same measurement and the measurement decides — the certificate
is a property of the result, not of the route
([simplex](simplex.md)).

For a conic model the measurement is against the **model's own cones**: `ρ` must
lie in every cone of the task, and the dual ray `d = A'y` must lie in `K*` on
every block, with the row multipliers signed and the right-hand-side
combination negative. A candidate that does not measure proves nothing.

## What is not published, and why

| Case | Verdict | Vector |
|---|---|---|
| LP/QP infeasible or unbounded | `PRIM_INFEAS_CER` / `DUAL_INFEAS_CER` | **published** |
| Conic infeasible / unbounded, **no PSD block** | same | **published** |
| Any route, model **with** a PSD block | in `prosta` | none — the compressed block has no image in `numvar` scalars |
| Tangent-cut route | in `prosta` | none — it solves in cut space, whose rows are tangents; lifting a witness is not sound |
| MIP unbounded relaxation | `UNKNOWN` unless the integer model is resolved | none — a relaxation ray does not establish integer unboundedness |
| Node cap reached, no incumbent | nothing | none |

In every "no vector" case the status is `UNKNOWN` and the verdict is readable
from `PRIMAL_getprosta`. A solution getter answers `PRIMAL_RES_ERR_ARG` — the
same code it uses for a model that simply has no solution. There is no case
where a status says "infeasible" and a getter hands over a zero vector that
turns out to violate the model.

## Solution getters and an absent solution

Eleven getters — `getxx`, `gety`, `getslc`, `getsuc`, `getslx`, `getsux`,
`getprimalobj`, `getdualobj`, `getprimalinfeas`, `getdualinfeas`,
`getsolutionslice` — answer `ERR_ARG` when there is no point to deliver. This
is separate from the certificate rule: a verdict about the model lives in
`prosta`/`solsta`, and a point lives in the solution buffers. One flag answers
"is there a point", never "was a verdict reached".

## Violation getters

`getprimalinfeas` / `getdualinfeas` measure the published point **against the
model as it is currently written**, and they count every way a term can enter a
row: scalar coefficients, quadratic row parts, and bar terms (off-diagonal
counted twice). Perturb a bound after the solve and the violation moves by the
amount you perturbed — that is the intended behaviour, not a leak.

Cone membership is deliberately **not** in `getprimalinfeas`: it is a fourth,
separate number, printed by the `[cones]` diagnostic line when
`GMB_DBG=1`, because a cone violation has to be measured in the cone's own
units.

## Limits

- No ray for MIP, and none for a model with a PSD block.
- No `getdualray` for the tangent-cut route in either direction.
- The `[cones]` line is a diagnostic under `GMB_DBG`, not a public getter.
