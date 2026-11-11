# QCQP encoding

**Source:** `primal_quad.c`

A convex QCQP constraint is not a new constraint type: it is an exact rewriting
into rotated quadratic cones, done by eigendecomposition. This file is that
rewriting, and the place where a **non-convex** model is refused.

## The rewriting

For a row with a quadratic part `Q`, find the eigendecomposition `Q = V Λ Vᵀ`.
An indefinite `Q` on a convex model splits into a positive and a negative
part, and the positive part is absorbed into a rotated cone, giving the form

```
(Q⁺) + rotated-cone constraint  ≡  original row
```

The mapping is **exact** — it is an eigendecomposition, not an approximation
or a relaxation. A diagonal `Q` is trivially handled, since its eigenvectors
are the axes in any reading.

## Non-convex domains are refused

The encoder measures the scale of each spectrum and refuses the whole model
beyond a **relative** threshold `1e-9 · max(1, max|λ|)`, with the reason printed
on stderr.

Two properties of that threshold are deliberate:

- It is **relative**, not absolute. An eigenvalue of `−2e-11` against a
  spectrum of 2 is noise and the model solves; an eigenvalue of `−1e-4` against
  a spectrum of `2e6` is the same relative distance and also solves. An
  absolute `1e-9` would have rejected the second.
- It refuses the **whole model** with an argument error rather than silently
  dropping the offending row. A truncated eigenvalue on the wrong side is a
  different model that still returns `OK`, which is worse than a refusal.

## What it buys

Once encoded, a QCQP row is an ordinary conic row and the model is solved by
the unified conic path ([SDP](sdp.md)) — a QCQP with both bars and quadratic
rows in the same task is one solve, not two.

## Limits

- Convexity only. There is no global branch-and-bound for non-convex quadratic
  programs; a non-convex domain is an error, not a slower answer.
- The dense symmetric block per quadratic row is `numvar × numvar`
  ([standard form](standard-form.md)).
- Only convex `Q` is representable on a `<=` row; equality and ranged quadratic
  rows are not encoded.
