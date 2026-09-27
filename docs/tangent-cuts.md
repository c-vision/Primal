# Tangent cuts (outer approximation)

**Source:** `sdp.c`, `expcone.c`

The fallback route for a model with a nonlinear or PSD block: linearize the
nonlinear part, solve the resulting LP, cut, repeat.

## The method

At the current point `x̄`, build a **tangent** of each nonlinear block, add the
tangent as a linear constraint, re-solve. The sequence of LP optima converges
from below (for a minimization) to the true optimum, and the accepted point
gives the matching upper bound. When the two coincide, optimality is proven by
the sandwich.

The cut is a genuine tangent of the block at the point, in the block's own
units. A power cone is homogeneous of degree 1, and reading its residual in
`t`-space shrinks a geometric violation by `1/(a·t^(a−1))` — a factor 6.4 on
one of the finance samples, and 3333 at `a = 1/3, t = 1e-6`. The stop test
therefore reads the **relative cone slack**, faces `t ≥ 0, u ≥ 0` included;
outside the domain `pow()` returns NaN, and `NaN > violation` is false, which
would read as "no violation".

## What it publishes

`pobj = dobj`, and a dual block `Z_j = C_j − Σ y_i A^i_j` — without the
multipliers of the cuts themselves. Those multipliers are an artifact of the
approximation, not of the model, so they are excluded from the public reduced
costs.

That leaves a real question: is the published `Z` inside `K*`? Measured, yes on
the AM-GM case forced onto this route — spectrum `{2, 1.3e-16}`, `⟨Z, B⟩ = 0`
at 2e-9, `dobj = pobj = 2`. It is a measurement, not a theorem, and a model
where the cuts leave a hole in the dual cone is possible in principle.

## How the route is chosen

By **measured quality**, never by an internal barrier parameter. μ is a path
parameter; `⟨x, s⟩ = μ·ν` measures complementarity; neither is an accuracy
certificate. The comparison is the relative triple against the tolerances the
task declares — and the acceptance factor `TOL_NEAR_REL` does **not** move that
choice, precisely so a fallback can never deliver a point that the native route
would beat.

## Measured

- Forcing this route with `GMB_NO_EXP_IPM=1` on `regression_regularized`:
  converges in 8 rounds (`rel_viol` 1.78e-9), exit 0.
- A risk-budgeting model: 0.98 s native, 0.07 s by cuts — the cuts are 14×
  faster on that one, and both reach `pobj = 2 − log 2`.

## Normalization

Every cut row is normalized by its own `max |·|`. The half-space is identical
and only the scale changes; without it the accumulated LP mixes coefficients
from `1e-4` to `1e4` and the LP's own IPM diverges. With it, this route
converges on the cases it previously failed, including the tightest tolerance
case in the suite, which now returns `OPTIMAL` with a measured cone slack of
2.1e-16.

## Limits

- Convergence is first-order in general: the number of rounds grows with the
  accuracy demanded.
- No ray is published from this route in either direction: it solves in **cut
  space**, whose rows are tangents rather than the model's constraints, so a
  witness of it is not a witness of the model, and lifting one is not sound.
  The verdict stays in the problem status with no vector
  ([certificates](certificates.md)).
- The dual block `Z` is admissible in the measured cases, not by construction.
