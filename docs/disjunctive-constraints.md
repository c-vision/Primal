# Disjunctive and affine conic constraints

**Source:** `primal_afe.c`, `primal_djc.c`

Constraints that are not "a row of `A`": disjunctions (either/or), SOS1/SOS2
sets, semi-continuous and semi-integer variables, and general affine conic
constraints. All of them reach the same place: a set of rows in the standard
form, plus a structure the MIP search can branch on.

## Affine expressions (AFE)

A building block used by everything below: an expression `F·x + g` over a named
set of columns, stored sparsely per row.

| Call | Purpose |
|---|---|
| `appendafes` / `getnumafe` | create expressions, count them |
| `putafefentry` / `putafefrow` / `putafeg` | write `F` and `g` |
| `getafeftrip` / `getafefnumnz` | read back |
| `putafebarfentry` | add a `⟨F̄_ij, X_j⟩` bar term to an expression |

The bar term is a real term, not metadata: it enters the row the ACC produces,
on both the linear and the conic branch.

## ACC — affine conic constraint

`appendacc` + `putacc` declares that a list of expressions must lie in a
conic domain. The sign convention is the reference one: the constraint reads
`F x + g − b ∈ domain`, so `b` is what the expression must equal for the zero
vector to be inside the domain.

The conic domains supported are the quadratic cones, the exponential and power
cones, and the linear ones (`ℝ`, `{0}`, `ℝ₊`, `ℝ₋`). The ACC is encoded by
adding the rows that put the expressions in the cone, which is why
`getpviolacc` and `getdviolacc` can measure it: the ACC keeps the first row it
produced, and the dual multipliers of those rows are its `doty`.

Generated rows are refreshed before solving after AFE coefficient, constant,
bar-term or bound edits. Cloning preserves the canonical ACC/DJC definitions.

## DJC — disjunctive constraint

`appenddjcs` + `putdjc` declares a **disjunction of clauses**: an OR, where each
clause is an AND of domains over affine expressions.

```
(domains of clause 0)  OR  (domains of clause 1)  OR  …
```

`termsizelist[i]` is the number of domains in clause `i`; `domidxlist` runs to
`sum termsizelist`; `afeidxlist` runs to the sum of the clauses' domain
dimensions. `b` has the same sign convention as an ACC: `F x + g − b`.

Encoding is big-M: one binary selector per clause, and the rows that select
which clause is active. The consequence is the same as for the MIP: the rows
are **emitted and not withdrawn**, so an ACC/DJC slot that has been written
cannot be rewritten — it would need different rows.

For multiple clauses, every required big-M must follow from finite variable
bounds or external singleton linear rows. An unsupported unbounded relaxation
returns `ERR_ARG`; there is no arbitrary `1e6` fallback. Bounds and AFEs are
re-read before each solve.

**Only linear domains are representable in a DJC.** A conic domain is an
argument error, and that is a declared limitation rather than a gap: the
backend is a big-M MIP, and there is no MIP big-M encoding for a nonlinear cone
here.

## SOS1 / SOS2

`sos1` permits at most one nonzero member; `sos2` permits at most two, adjacent
in weight order. Negative values count as nonzero. The MIP search branches on
these support restrictions without a 64-member size exemption.

## Semi-continuous / semi-integer

`{0} ∪ [l,u]` — the variable may be off, or inside `[l,u]`. Encoded with the
same selector mechanism. The consequences for the search and for
`PRIMAL_getprimalinfeas` are in
[branch and bound](branch-and-bound.md), and they are not optional details: the
root box has to be relaxed for the lower bound to remain valid.

## Limits

- Big-M everywhere: no ideal formulation, no big-M free encoding.
- No rewriting of an existing disjunctive slot.
- A clause made only of unconstrained (`ℝ`) domains is always true and emits no
  rows.
- No ray or certificate is derived from a disjunction.
