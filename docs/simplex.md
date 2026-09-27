# Simplex

**Source:** `simplex.c` / `simplex.h`

A two-phase primal simplex on the standard form ([standard form](standard-form.md)),
with dense partial-pivoting LU ([linear algebra](linear-algebra.md)) on the
basis. It is the route that produces both Farkas rays, because a tableau
carries the evidence a certificate needs.

## The method

Standard primal simplex, two phases. Phase 1 minimizes the sum of artificials;
phase 2 minimizes the real objective. Price: **Dantzig** (most negative reduced
cost) with a **Bland** fallback after a stalling counter trips, which is what
makes anti-cycling a guarantee rather than a hope.

Refactorization is on a numeric-growth trigger, not on a fixed iteration count.

## Why the simplex is still here

The interior-point methods are faster on large sparse problems, but the simplex
is not dead for two reasons:

1. **It writes the certificates.** See below.
2. **A basis is reusable.** `PRIMAL_solvebasis` and
   `PRIMAL_solvewithbasis` exist because a solved basis is a piece of state
   worth keeping; an interior-point method has none.

## Farkas rays

Both rays come out of the tableau, and each is written by a specific event.

**Dual ray (infeasibility).** Phase 1 ends with a positive artificial
objective while the basis is feasible for phase 1. The reduced-cost row of that
basis *is* a certificate: `b'y > 0` together with `A'y ≤ 0` is a combination of
the user's constraints that no nonnegative point can satisfy.

**Primal ray (unboundedness).** The ratio test finds no leaving row. The
direction of unbounded growth satisfies `A·ρ = 0`, `ρ ≥ 0`, `c'ρ < 0` for a
minimization — a recession direction along which the objective improves without
limit.

An interior-point method has neither: it has a last iterate, not a witness. So
its last iterate is offered to the same measurement, and **the measurement
decides** whether a certificate is declared. That is why the ray is a property
of the result and not of the route.

## Where a ray is refused

Two rejections, both because the vector would not mean what its name says:

- Support on a variable upper-bound cap row. A ranged variable becomes two
  rows; a multiplier on the cap has no one-entry-per-constraint image in the
  user's model, so it is not published even though the reduced-space argument
  behind it is sound.
- Support on **both** sides of a ranged row, for the same reason.

The conic routes publish both rays for a **bar-free** model; with a PSD block
the compressed block has no image in `numvar` scalars, so no ray is published
there ([certificates](certificates.md)).

## Limits

- Dense basis; the sparse LU exists but the simplex does not use it.
- No crossover from an interior-point point (the routes are alternatives, not
  stages — see [the routing table in the README](../README.md)).
- No bound flipping / crossover heuristics of the classical kind; cramming is
  not implemented.
