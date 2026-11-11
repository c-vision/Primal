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

## DJC — disjunctive constraint

`appenddjcs` + `putdjc` declares a **disjunction of clauses**: an OR, where each
clause is an AND of domains over affine expressions.

```
(domains of clause 0)  OR  (domains of clause 1)  OR  …
```

`termsizelist[i]` is the number of domains in clause `i`; `domidxlist` runs to
`sum termsizelist`; `afeidxlist` runs to the sum of the clauses' domain
dimensions. `b` has the same sign convention as an ACC: `F x + g − b`.

### Row-wise big-M formulation

Let $z_d\in\{0,1\}$ select clause $d$. For each inequality
$a^Tx\le\beta$ of that clause, PRIMAL emits

$$
a^Tx+M_{d,i}z_d\le\beta+M_{d,i},
\qquad \sum_d z_d\ge1.
$$

When $z_d=1$, the original inequality applies. When $z_d=0$, the
row is relaxed by $M_{d,i}$. On a box $l\le x\le u$, the largest
left-hand side is obtained by choosing $u_j$ for $a_j\ge0$ and
$l_j$ for $a_j<0$. The encoder uses

$$
M_{d,i}=\max\!\left(0,\sum_{a_j\ge0}a_j u_j+
                              \sum_{a_j<0}a_j l_j-\beta\right).
$$

In [primal_djc.c](../primal_djc.c), `djc_encode` and
`djc_rewrite_row` use this bound when the selected terms are finite.
If a selected term is nonfinite, the implementation uses `DJC_BIGM = 1e6`.
This fallback can restrict an inactive clause when its true violation
exceeds the constant. A loose finite $M$ can weaken the relaxation
and worsen scaling.

For $0\le x\le10$ and $x\le2$ or $x\ge6$, the two upper-form
rows are $x\le2$ and $-x\le-6$. Their constants are $8$ and $6$.
The emitted rows are $x+8z_1\le10$ and $-x+6z_2\le0$, with
$z_1+z_2\ge1$. The relaxed point $x=4$, $z_1=z_2=1/2$ satisfies
the encoded LP and illustrates the role of integrality.

### Supported domains and updates

The domain registry used by ACCs includes nonlinear cones. DJC validation
accepts exactly `R`, `RZERO`, `RPLUS` and `RMINUS`.
`R` contributes no row, `RZERO` contributes two opposite inequalities,
and the other two domains contribute one inequality with the appropriate
sign. A nonlinear cone in a DJC yields `PRIMAL_RES_ERR_ARG`.

A filled DJC slot keeps its selector variables and row layout. Accepted
edits to referenced AFEs can refresh the coefficients and right-hand sides
of those rows through `primal_djc_sync`.

## SOS1 / SOS2

`sos1` is a set of columns of which at most one may be non-zero; `sos2` at
most two. Both are handled as a special form of the same structure: the
selector variables and the rows that link them, with SOS1 branching available
to the MIP search ([branch and bound](branch-and-bound.md)).

## Semi-continuous / semi-integer

`{0} ∪ [l,u]` — the variable may be off, or inside `[l,u]`. Encoded with the
same selector mechanism. The consequences for the search and for
`PRIMAL_getprimalinfeas` are in
[branch and bound](branch-and-bound.md), and they are not optional details: the
root box has to be relaxed for the lower bound to remain valid.

## Limits

- The DJC formulation uses big-M and its relaxation strength depends on the row bounds.
- No rewriting of an existing disjunctive slot.
- A clause made only of unconstrained (`ℝ`) domains is always true and emits no
  rows.
- No ray or certificate is derived from a disjunction.

## Reference

- J. P. Vielma, [Mixed Integer Linear Programming Formulation Techniques](https://doi.org/10.1137/130915303), 2015.
