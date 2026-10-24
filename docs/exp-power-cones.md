# Exponential and power cones

**Source:** `expcone.c` / `expcone.h`

Barrier, gradient and Hessian in closed form for the **non-symmetric** cones —
the exponential cone and the power cones. These are the cones where a symmetric
conic method does not apply, and this file is what makes them native rather than
approximated.

## The cones

| Type | Membership | Note |
|---|---|---|
| `PEXP` | `t ≥ u·e^(v/u)`, `u, v ≥ 0` | the exponential cone |
| `DEXP` | as `PEXP` with `u ≤ 0` | its mirror |
| `PPOW(a)` | `t^a·u^(1−a) ≥ \|v\|` | 0 < a < 1, power cone |
| `RPOW(a)` | `√2·t^a·u^(1−a) ≥ \|v\|` | rotated power cone |

`RPOW(1/2)` coincides with `RQUAD`, which is why both exist.

## The barriers

Every block uses a logarithmically homogeneous **self-concordant** barrier with
parameter ν = 3 (`<x, ∇f(x)> = −3`):

| Type | Barrier |
|---|---|
| `PEXP` | `f = −log(u·log(t/u) − v) − log t − log u` |
| `PPOW(a)` | `f = −log(t^(2a)·u^(2(1−a)) − v²) − (1−a)·log t − a·log u` |
| `RPOW(a)` | as `PPOW` with `2·t^(2a)·u^(2(1−a))` |

The `PEXP` barrier is the one of Nesterov and Chares, also used by MOSEK, ECOS,
SCS and Clarabel. Until 2026-10-24 the code used `−log(t − u·e^(v/u)) − log u`
(ν = 2). That function is logarithmically homogeneous but **not**
self-concordant. Along random lines, `|f'''| / (2·f''^1.5)` reaches **2.54**,
measured with exact Taylor jets (the bound is 1). It cannot be fixed by
rescaling either: the faces `t = 0` and `u = 0` of the cone meet at `(0, 0, −1)`,
so no self-concordant barrier of this cone has ν < 3. Without
self-concordance the Newton steps lose centrality. On the max-entropy family
(`bench/kkt_scale exp N`) a symmetric problem broke its symmetry through an
ill-conditioned KKT, the dual step went to zero, and N = 30, 50, 120 and 300
fell to the tangent cuts. N = 120 took 20 s and 8.8 GB, with a relative error
of 3e-7. With the ν = 3 barrier every size from 10 to 4000 answers natively,
to about 1e-12. N = 120 takes 0.02 s and 3 MB, N = 4000 takes 5.8 s and
146 MB (`T298` checks both the self-concordance bound and the family).

The scaling point `W` (`−∇f(W) = s`) of the `PEXP` barrier comes from one scalar
equation. `s₂ = −1/ψ` fixes ψ, `t = (1 − u·s₂)/s₀`, and with `a = 1/u` the
remaining row is `a − s₂·log((a − s₂)/s₀) + s₂ − s₁ = 0`. That function is
increasing and concave, and it is negative at `a = 0⁺` exactly when `s` is
inside `K*`. Newton started on the left of the root therefore converges to it
monotonically.

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
are not comparable instruments. (The `PEXP` rows were measured with the earlier
ν = 2 barrier, see above. They were not re-measured.)

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
