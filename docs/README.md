# Documentation

One file per algorithm or subsystem. Each one says what the algorithm is, when
the solver picks it, what it was measured at, and where it stops.

Start with the [README](../README.md) for the shape of the solver; come here
for a specific route.

## The solve path, in order

| File | What it covers |
|---|---|
| [standard-form.md](standard-form.md) | the canonical form every route consumes |
| [scaling.md](scaling.md) | power-of-two equilibration, and the Jacobi pass on the augmented KKT |
| [presolve.md](presolve.md) | row/column reduction, exact postsolve, the homogeneous lift of a ray |
| [linear-algebra.md](linear-algebra.md) | dense LU, sparse LU, Cholesky, sparse LDLᵀ with 2×2 pivots, Jacobi eigenvalues |

## The routes

| File | Class |
|---|---|
| [simplex.md](simplex.md) | LP / QP — two-phase primal simplex, source of both Farkas rays |
| [interior-point.md](interior-point.md) | LP / QP — Mehrotra, dense augmented or sparse normal equations |
| [conic-socp.md](conic-socp.md) | SOCP — NT scaling, sparse Cholesky, dense-LU fallback |
| [sdp.md](sdp.md) | SDP and the unified conic path (PSD + SOC + exp in one KKT) |
| [exp-power-cones.md](exp-power-cones.md) | exponential and power cones, natively |
| [tangent-cuts.md](tangent-cuts.md) | the outer-approximation fallback |
| [qcqp-encoding.md](qcqp-encoding.md) | quadratic rows encoded exactly into rotated cones |
| [branch-and-bound.md](branch-and-bound.md) | MILP / MIQP / mixed-integer conic |
| [disjunctive-constraints.md](disjunctive-constraints.md) | AFE, ACC, DJC, SOS1/SOS2, semi-continuous |

## The contract

| File | What it covers |
|---|---|
| [certificates.md](certificates.md) | KKT, Farkas rays, what a status does and does not guarantee |
| [api.md](api.md) | the map of the 547 public functions, parameters, callbacks |
| [io-formats.md](io-formats.md) | MPS, LP, OPF, CBF — what is read, written, and refused |

## Building

| File | What it covers |
|---|---|
| [windows.md](windows.md) | building on Windows with MinGW-w64 (and why not MSVC yet) |

## Conventions

Unless a page says otherwise:

- Sense is normalized: a maximization is multiplied by `−1` internally and the
  result is reported in the problem's own sense.
- Tolerances are **relative** where a triple is published
  (`pfeas/(1+‖b‖∞)`, `dfeas/(1+‖c‖∞)`, `|pobj−dobj|/(1+|pobj|+|dobj|)`), and a
  tolerance the task declares is always the one the decision uses.
- "Measured" means a number this repository produced and a test asserts, not a
  number copied from somewhere else. Where a comparison number comes from
  another solver it is named as that solver's number.
- Limits sections are part of the documentation, not an apology: they say where
  the algorithm stops, which is the useful part.
