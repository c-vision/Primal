# Interior point — LP and QP

**Source:** `ipm.c` / `ipm.h`

Mehrotra predictor–corrector interior-point method for LP and convex QP. Two
linear-algebra backends, chosen by problem size and sparsity.

## The method

The standard formulation: at each iterate,

1. solve an **augmented system** (or its equivalent normal equations) for the
   search direction,
2. choose a step that keeps `x, s > 0` — primal and dual steps are taken
   **separately** and fractionally to the boundary,
3. update the barrier parameter `μ` following Mehrotra's adaptive rule.

The predictor–corrector uses the affine step to estimate the centering
direction, then solves a second system with the second-order correction. Both
steps end at `α_max(1 − θ)` with a floor on `θ`, so an iterate never touches
the boundary of the positive orthant.

Stopping is on the **relative** triple:

```
rel_pri = pfeas / (1 + ‖b‖∞)
rel_dual = dfeas / (1 + ‖c‖∞)
rel_gap  = |pobj − dobj| / (1 + |pobj| + |dobj|)
```

against the task's own `TOL_PFEAS` / `TOL_DFEAS` / `TOL_REL_GAP` (default
`1e-8`). A `near`-relative acceptance factor (`TOL_NEAR_REL`, default 1000)
multiplies them; it is a tolerance multiplier, not a separate status.

## Two backends

| Backend | System | Chosen when |
|---|---|---|
| Dense augmented | the full KKT, dense LU | small/medium problems, any QP |
| Sparse normal equations | `K = A·M⁻¹·Aᵀ + δI`, `M = Q + Z/X`, sparse Cholesky | large sparse LP, large sparse QP |

Normal equations are cheaper in flops but square the condition number. The
`δI` shift is what keeps it positive definite; the trade is deliberate, and it
is why the dense augmented system stays available.

For a QP the same Cholesky runs on `M = Q + Z/X` — the primal-dual scaling of
the quadratic term.

## Measured

- 90 000-variable transport LP: ~0.6 s.
- Sparse QP, `n = 20000`: ~0.06 s.
- 400×200 LP, sparse route forced: 0.57 s (HiGHS: 0.016 s on the same
  instance). PrimalSolver is **slower** here; see
  [how it compares](../README.md#how-it-compares) and the honest note in it.

## Adaptive regularization

Degenerate systems are regularized adaptively: when the normal-equations
Cholesky detects a breakdown, the shift `δ` is raised and the direction
recomputed, rather than the solve being abandoned. This is what keeps
degenerate LPs and QPs converging instead of returning a plausible wrong point.

## Limits

- No crossover to a basis.
- No homogeneous self-dual embedding: phase-1 is done by a separate
  primal-dual phase, so an infeasible problem costs an extra factorization
  rather than being detected by the central path.
- The normal-equations path is the condition-number-limited one; on very badly
  scaled data the dense augmented path is the safer choice and is not
  automatically selected.
