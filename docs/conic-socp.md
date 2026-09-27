# Conic interior point — SOCP

**Source:** `socp.c` / `socp.h`

Primal-dual interior-point method over second-order cones, handling `R₊`,
quadratic (`QUAD`) and rotated quadratic (`RQUAD`) blocks.

## The method

The standard conic IPM: at each iterate solve

```
[ A   0 ] [ dx ]   [ rx ]
[ 0   W ] [ ds ] = [ rs ]
```

for the search direction, where `W` is the scaling of the cone at `(z, s)`.
The step is limited so that `(z + α dz, s + α ds)` stays in `K`, and the
barrier parameter decreases as in the LP/QP method.

## Nesterov–Todd scaling

For the quadratic cones the scaling point is the Nesterov–Todd point, found by
Newton iterations on the analytic center of `{Z ≻ 0, Z S = z sᵀ}`. It is the
scaling that makes the normal equations symmetric, which is what allows a
**Cholesky** instead of a general factorization.

So there are two backends:

| Backend | When |
|---|---|
| Dense augmented LU | small/medium |
| NT normal equations + sparse Cholesky | large sparse |
| Augmented sparse LU | **fallback**, when the scaling point degenerates near the cone boundary |

That last row is the interesting one. Near the boundary the NT scaling becomes
nearly singular, the Cholesky loses accuracy, and the honest response is to
fall back to a factorization that does not need the scaling to be well
conditioned. The fallback is a decision made per solve, not a configuration.

## Kept best iterate

The method tracks the best *feasible* iterate by duality gap and returns it
when the gap diverges. This is a stack buffer, allocated on the stack on
purpose: a fresh heap allocation perturbed the floating-point arithmetic enough
to change the verdict on degenerate cases. The condition for using it is
narrow on purpose — a diverging gap, a previously good feasible point, and not
an infeasible problem (which must be allowed to be infeasible).

## Measured

SOCP `min Σ t_k  s.t.  (t_k, x_{2k}, x_{2k+1}) ∈ QUAD, Σ x_i = 1`, optimum
`1/√2` closed form:

| n | blocks | PrimalSolver | Clarabel |
|---|---|---|---|
| 40 | 20 | 0.5 ms | 0.1 ms |
| 120 | 60 | 4.6 ms | 0.2 ms |
| 200 | 100 | 14.6 ms | 0.5 ms |

**PrimalSolver is 10–30× slower than Clarabel here, at the same accuracy.**
The gap is the price of dependency-free C: Clarabel is Rust with tuned sparse
kernels.

## Limits

- Only the quadratic cones: no PSD block (that is [SDP](sdp.md)), no
  exponential or power cone (that is [exp/power](exp-power-cones.md)).
- The assembled KKT is **unsymmetric** — the cone block appears transposed
  relative to the primal block — so the sparse symmetric `LDLᵀ` cannot replace
  the sparse LU here. Making it symmetric is a prerequisite, and a known piece
  of open work.
- No homogeneous embedding: infeasibility is detected by an explicit phase-1
  cone.
