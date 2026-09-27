# Scaling

**Source:** `scaling.c` / `scaling.h`

Scaling is a row/column equilibration step that runs before the solve, and it
is deliberately the simplest thing that works: exact **powers of two**.

## Why powers of two

A factor of `2^k` is exact in binary floating point. Multiplying and dividing
by it introduces no rounding at all, so the equilibration cannot itself perturb
the numbers the solver is about to work with. Most equilibration schemes
multiply by an arbitrary `1/sqrt(row_norm)`, which is harmless in exact
arithmetic and a small perturbation in floating point.

The route (`PRIMAL_IPAR_SCALING`, default on) applies it to rows and columns of
the standard form.

## Why it matters here

The conic and semidefinite paths assemble an **augmented KKT system** whose
multiplier block grows like `1/μ` — about 11 orders of magnitude of dynamic
range by the time `μ` reaches `1e-8`. Without equilibration that block
dominates the pivoting and the factorization loses most of its significant
digits. A **Jacobi row/column equilibration** of the augmented matrix is applied
immediately before the dense LU for exactly this reason.

## Scaling versus presolve

They are not the same thing and are not alternatives:

- **Scaling** multiplies a row or a column by a number. It is reversible
  exactly, costs one pass, and changes no answer.
- **Presolve** ([presolve](presolve.md)) removes rows and columns, and has to
  reconstruct the answer afterwards.

The conic presolve is also *dual-safe*: a row implied by the variable bounds is
dropped before `optimize_conic` and restored after, because the dual of an
implied row is zero and the KKT of the user's model therefore still holds. That
is not true of bound tightening, which is why bound tightening runs on the MIP
path only ([branch and bound](branch-and-bound.md)).

## Environment switches

`GMB_BENCH_NOSCALING` and `GMB_BENCH_NOPRESOLVE` exist so the benchmark harness
can measure each contribution separately. Both are development-only and are not
part of the public API.

## Limits

- Row and column equilibration only: no second-order (B=I-diag(1/|Aᵀr|)) or
  geometric-mean scaling, no scaling of the cone or PSD blocks beyond the
  Jacobi pass on the assembled KKT.
- Not idempotent by construction — a row is scaled once, and the reverse pass
  restores it exactly because the factors are powers of two.

