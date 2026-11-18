# Standard form

**Source:** `stdform.c` / `stdform.h`

Simplex and the LP/QP interior-point method work with equalities and
nonnegative variables. Standard-form conversion provides those data
and a map back to the original variables. Conic backends build their
own representations to preserve the cone geometry. Start with the
[modeling chapter](problem_formulations.md) for the full LP/QP conversion.

## What the form is

For an LP or QP, the conversion turns row inequalities into equalities
by adding slacks, shifts or splits variables, and changes maximization
to minimization. The linear standard form is $Ax=b$, $x\ge0$, where
$x$ now includes the slacks. A QP also carries its transformed Hessian.

- **A and Q sparse, in CSC.** Dense materialization happens only on demand,
  for the small dense kernels.
- **Free variables split.** `x ∈ ℝ` becomes `x⁺ − x⁻` with both halves ≥ 0.
- **Ranged rows split** into a `<=` row and a `>=` row, each with its own slack.
- **Fixed variables substituted out**, the objective constant folded into
  `cfix`.

## Worked example

Minimize $2x$ with $1\le x\le4$ and $x\ge3$. Substitute $x=1+u$.
The upper bound gives $u+s=3$ and the lower row gives $u-r=2$, where
$u,s,r\ge0$. The objective becomes $2+2u$. The minimum occurs at
$(u,s,r)=(2,1,0)$, and mapping back gives $x=3$ and objective 6.
The constant 2 affects the reported objective even though it does
not affect the minimizing point.

## Mappings back to the user's model

| Function | Direction |
|---|---|
| `stdform_map_x` | standard-form `x, s` → user `x` |
| `stdform_map_y` | row multipliers → user `y` |
| `stdform_map_dir` | a **direction** → a direction |

`stdform_map_dir` is separate on purpose, and it is not a detail: a ray of
Farkas or a recession direction lives on *directions*, not on values, so it
must not pick up the shift that `map_x` applies to fixed variables. Mapping a
direction through a shift would produce a vector that is not a direction.

The same care governs `stdform_map_y` for a ranged row: it **sums** the
multipliers of the two halves, so a user constraint never ends up with two
unrelated numbers.

## Conic quantities

The unified conic backend uses a similar shift and cap row for bounded
scalar variables where applicable. Cone coordinates are represented
with their block and its affine links. The unified converter can
absorb eligible members directly into the block. The SOCP converter
builds explicit equalities and affine cone slacks. See the
[solver architecture](architecture.md#the-dispatch) and
[fixed-coordinate reduction](socp_fixed_reduction.md).

## Limits

- The quadratic form is stored as a symmetric dense block per quadratic row
  (`putqconk`), not as a sparse Cholesky factor: fine for the model sizes
  QCQP rows appear in, not for a wide sparse `Q`.
- The mapping in this note describes `stdform.c`. For the mappings used
  by individual conic engines, follow the linked chapter for that solver.

## Exercise

If the example finds the direction $\Delta u=1$, what is $\Delta x$?
It is 1 because $x=1+u$ gives $\Delta x=\Delta u$. The shift is part
of the point map only.
