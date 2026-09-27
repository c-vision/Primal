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
| 4 | 2.0 ms | 0.2 ms | ~1e-5 |
| 6 | 9.1 ms | 0.2 ms | ~1e-5 |
| 8 | **599 ms** | 0.5 ms | ~1e-5 |

Two things to read here. PrimalSolver is far slower than SCS, and SCS is
first-order so it only agrees to `1e-5` while the IPM reaches `1e-8`. The jump
at `d = 8` is the dense LU of the KKT and is a real scaling limit.

## Limits

- Experimental HSD convergence on degenerate SDPs can depend on build settings;
  an iteration limit does not provide a solution.
- The augmented system is assembled **densely** (`Sys[Nsys²]`), which caps
  problem size. The sparse 2×2-pivot LDLᵀ exists
  ([linear algebra](linear-algebra.md)) and is validated, but is not wired here
  yet.
- No ray is published for a model **with** a PSD block: the compressed block has
  no image in `numvar` scalars ([certificates](certificates.md)).
- An extreme not attained inside a bar has no recession direction and is
  reported as "no verdict" rather than as a value.
