# Semidefinite programming

**Source:** `sdp.c` / `sdp.h`

Primal-dual interior-point method for semidefinite programs, and the home of
the **unified conic path**: PSD blocks, SOC/RQUAD blocks and
exponential/power blocks in *one* augmented KKT system.

## The unified path

The interesting decision in this file is that the cones are not solved by
separate methods. A model can carry a PSD block, a quadratic cone and an
exponential cone at the same time, and all of them go into a single augmented
system:

| Block | Row of the KKT |
|---|---|
| SOC / RQUAD | arrow `A(z)` / `A(s)` |
| PSD | Nesterov–Todd Schur complement |
| exp / power | Hessian-NT row `Thin`, μ-free and σ-free |
| scalar | direct entry |

Because the block types are interleaved in the same matrix, the whole system is
equilibrated by a **Jacobi row/column pass** before the dense LU, and the
solution is **iteratively refined** with a Wilkinson stopping test during the
polish. The refinement only runs in the polish phase.

## The Nesterov–Todd row for exp/power

The Hessian-NT scaling is `μ`- and `σ`-free: `Thin = M(z, s/scale)` is a
function of the iterate only, and `σ` appears solely in the residual. That is
the Mehrotra structure — one matrix, two residuals — and it is why the metric
does not need a damping parameter.

## Acceptance is measured, not assumed

The three relative figures (`rel_pri`, `rel_dual`, `rel_gap`) count equations
and complementarity. They do **not** see whether a point is inside the cones.
So the exp/power iterate is accepted only if its own measured relative triple
meets the tolerances declared on the task; otherwise the route reports "not
solved" and the dispatcher falls back to tangent cuts
([tangent cuts](tangent-cuts.md)). The fallback is a decision, not an
accident.

A related rule: a **cone has no rows to leave unsatisfied**, so the vertex of a
block satisfies every equation the model has. That means the triple alone can
close on a point that is not the optimum — the fourth face (the block's reduced
cost) is what decides, and it is checked in `PRIMAL_optimize`, the one place
that answers for the user's model rather than for an iterate.

## Where the cap lives

A cap on the entries of a PSD block keeps the *first* LP of an outer
approximation finite. A point that sits **on** that cap is the answer to the
capped problem, not to the user's, so it publishes nothing: a recession
direction is measured on the model as written, and if none measures, the verdict
is "no verdict" rather than a value.

## Measured

SDP `min ⟨C,X⟩  s.t.  X_ii = 1, X ⪰ 0`, `C = −vvᵀ`, closed-form optimum
`−(Σ|v_i|)²`:

| d | PrimalSolver | SCS (first-order) | objective agrees to |
|---|---|---|---|
| 4 | 7 µs | 0.2 ms | ~1e-8 |
| 6 | 11 µs | 0.2 ms | ~1e-8 |
| 8 | 13 µs | 0.5 ms | ~1e-8 |

This family is answered by the fixed-diagonal certificate
(`sdp_fixed_diagonal`): when the rows are exactly `X_ii = 1` on a single PSD
block the optimum is `−(Σ|v_i|)²` in closed form, so the cells above time the
certificate, not an interior-point iteration. They read 2.0 ms, 9.1 ms and
599 ms before that path existed, and the jump at `d = 8` was the dense LU of the
KKT — which no longer applies either.

On a model that does iterate — `maxcut_sdp` (`p=4`, `q=4`, `n=8`, 16 edges: one
4×4 block plus 16 scalar rows) — the representation is what moves the time:

| KKT form | time |
|---|---:|
| sparse (the default above the work estimate) | 13.3 ms |
| dense (`GMB_SDP_DENSE=1`) | 38.4 ms |

Medians of seven runs each, the example's own timer: **2.9× for the sparse
form**, and both reach the same `−4` objective.

## Limits

- The augmented system is **sparse** whenever the dense work estimate
  (`Nsys³` per assembly plus the LU) exceeds `SDP_SPARSE_WORK` (4e8): the pattern
  is built once, the values are updated per iteration, the ordering is computed
  once and reused, and the factorization is sparse LU with partial pivoting
  (`splu_factor_ord`). Mixed SOC + exp/power models additionally trial the sparse
  form against a density estimate, with no size floor. `GMB_SDP_DENSE` and
  `GMB_SDP_SPARSE` force either route. The validated 2×2-pivot LDLᵀ
  ([linear algebra](linear-algebra.md)) is **not** used here: this path factors
  with LU.
- No ray is published for a model **with** a PSD block: the compressed block has
  no image in `numvar` scalars ([certificates](certificates.md)).
- An extreme not attained inside a bar has no recession direction and is
  reported as "no verdict" rather than as a value.
