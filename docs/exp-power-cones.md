# Exponential and power cones

**Source:** `expcone.c` / `expcone.h`

Barrier, gradient and Hessian in closed form for the **non-symmetric** cones —
the exponential cone and the power cones. These are the cones where a symmetric
conic method does not apply, and this file is what makes them native rather than
approximated.

## The cones

| Type | Membership | Note |
|---|---|---|
| `PEXP` | `t ≥ u·e^(v/u)`, `u > 0`; or `u = 0, t ≥ 0, v ≤ 0` | closed exponential cone |
| `DEXP` | `t ≥ −v·e^(u/v−1)`, `v < 0`; or `v = 0, t ≥ 0, u ≥ 0` | mathematical dual of `PEXP` |
| `PPOW(a)` | `t^a·u^(1−a) ≥ \|v\|` | 0 < a < 1, power cone |
| `RPOW(a)` | `√2·t^a·u^(1−a) ≥ \|v\|` | rotated power cone |

`RPOW(1/2)` coincides with `RQUAD`, which is why both exist.

`DEXP` corrects the former mirrored-cone interpretation; models using that
interpretation must be reformulated. The native and cut routes use this same
coordinate convention.

## What the file provides

| Function | Purpose |
|---|---|
| `expcone_barrier` / `expcone_grad` / `expcone_hess` | closed forms for the primal cone |
| `expcone_dual_in` | membership in the **dual** cone `K*` |
| `expcone_dual_point` | the scaling point `W` solving `−∇f(W) = s` |
| `expcone_scaling` | primal-dual scaling for the native IPM |
| `expcone_dual_maxstep` | maximum step keeping a point in `K*` |

`expcone_dual_in` is **scale-invariant**: it normalizes by the 1-norm first.
That is not cosmetic — a dual variable lives at `μ ≈ 1e-8`, and a membership
test that is not scale-invariant gives a different answer on a problem of a
different scale, i.e. the same model behaves differently depending on how it is
parameterized.

Similarly `expcone_dual_maxstep` exists as a separate function from the primal
one: a dual step has to be tested against `K*`, and testing it against `K` —
which is what a single shared function would do — collapses the step to
approximately zero and freezes the iterate.

## Accuracy against hand values

Four oracles with closed-form optima, solved natively:

| Problem | Optimum | Native | Tangent cuts |
|---|---|---|---|
| `PEXP`, `u = v = 1` | `e` | 1.6e-13 | 1.3e-12 |
| `PEXP`, `u = 2, v = 1` | `2√e` | 6.9e-14 | 1.9e-8 |
| `PPOW(0.3)`, `u = 1, v = 2` | `2^(1/0.3)` | 1.9e-13 | 5.1e-8 |
| `RPOW(0.4)`, `u = v = 1` | `2^(−1.25)` | 9.6e-15 | 1.5e-9 |

Relative errors. The native route is **4 to 5 orders of magnitude** more
accurate on three of the four, because a closed-form barrier and a tangent cut
are not comparable instruments.

A degenerate CBF case (2×2 PSD plus a power cone, optimum on the boundary)
answers natively at 7.1e-6 radial KKT residual, against 5.4e-5 for the cuts.

## When the native route is skipped

On a 40-block model (`logistic_large`, mixed PEXP and RQUAD) no iterate of the
native path reaches the declared relative triple, so the tangent cuts answer
instead — correctly, and by a decision made on measured quality
([tangent cuts](tangent-cuts.md)).

The path can also be forced off with `GMB_NO_EXP_IPM=1`, which selects the
cuts. Note that for the native path the variable must be *unset*, not set to
`"0"` — the switch counts **presence**, not value.

## Limits

- No ray is published for a model with a PSD block
  ([certificates](certificates.md)).
- The degenerate case where the conic route freezes on one particular
  trajectory toward a boundary optimum is documented as open work; the result
  is the same on every route, only the route differs.
- Accuracy on very badly scaled blocks degrades; the multi-block floor is a
  known open item.
