# Scaling

**Source:** `scaling.c` / `scaling.h`

Scaling is a row/column equilibration step that runs before the solve, and it
is deliberately the simplest thing that works: exact **powers of two**.

## Cost: one sweep per pass, not one per row

Each pass equilibrates the rows and then the columns, four times. Both directions
are `O(nnz + n)`:

- **row pass** — one sweep of the column-wise store accumulates `Σⱼ a_ij²` and the
  entry count for every row at once, one sweep applies each entry's row factor.
- **column pass** — one sweep per column over its own entries.

The row pass used to loop over rows on the outside and scan every column inside,
filtering `sub[k] == i`, which made a single pass `2·O(n·nnz)` and the four passes
`8·O(n·nnz)`. On the sizes that matter this is the difference between half a
second and a millisecond:

| `nvar × ncon` (`nnz`) | before | after |
|---|---|---|
| 400 × 200 (4 827) | 0.0020 s | 0.00003 s |
| 1600 × 800 (76 814) | 0.112 s | 0.0004 s |
| 2022 × 1603 (194 663) | 0.515 s | 0.0009 s |
| 100 × 5050 (30 281) | 0.216 s | 0.0002 s |

The arithmetic per entry is unchanged, so the result is **bit-identical**: verified
by comparing `A`, `r`, `d`, `lc`, `uc` and `c` byte for byte against the previous
loop at five sizes, and by the benchmark objectives, which did not move on any of
the 49 instances of the README table. The factor applied to an entry is the one
computed for that row **in that pass**, not the accumulated `r[i]` — multiplying
once per pass by the accumulated factor applies `f₁f₂f₃f₄` to every entry, which is
a different (and silently wrong) scaling.

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

