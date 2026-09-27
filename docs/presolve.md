# Presolve

**Source:** `presolve.c` / `presolve.h`

Presolve removes structure the user did not ask for, and puts the answer back
afterwards. Two properties make it usable: the reduction is **exact**, and the
reconstruction is **exact too** — including for a *direction*, which is the
part that is easy to get wrong.

## What is removed

| Reduction | What it catches |
|---|---|
| Empty rows | a row with no coefficients (0 = b) |
| Empty columns | a variable that appears in no row |
| Singleton rows | a row with one nonzero, folded into that variable's bound |
| Duplicate rows | the same constraint written twice |

## Postsolve

Every reduction is logged, and the log is replayed in reverse to rebuild the
user's `x` and `y`. This is the ordinary half.

## The homogeneous lift

A Farkas direction cannot go through the ordinary postsolve, and the reason is
worth stating because it is easy to get subtly wrong. A direction has no
objective term, so the reconstruction must be **homogeneous**:

```
y_i = -(Σ_k A_kj · y_k) / A_ij
```

with the objective coefficient `c_j` deliberately excluded. The reduced
right-hand side is `b_k − A_kj·x_j^fix`, so this returns exactly `y_i·b_i` and
`b'y` is invariant under the reduction.

Why it matters: a witness found on the reduced system has to **name the rows
that were removed**, otherwise the certificate does not describe the user's
model. Replaying the log homogeneously is what gives the rows back their
multipliers.

## Conic presolve

On a conic model the dual-safe reduction above (dropping rows implied by the
bounds, then restoring them) runs before `optimize_conic` and is undone
afterwards. See [scaling](scaling.md) for why this one is *not* the same
operation as bound tightening.

## Limits

- Row and column removal only. No substitution of a variable out of several
  rows, no clique or dominated-variable detection.
- No probing, no objective-bound propagation.
- Convexity detection is not part of presolve: a non-convex quadratic domain is
  refused at encoding time ([QCQP encoding](qcqp-encoding.md)), not fixed here.
