# Standard form

**Source:** `stdform.c` / `stdform.h`

Every route — simplex, interior point, conic, MIP relaxation — consumes the
same canonical form. This is the file that turns a user's model into it, and
back.

## What the form is

A problem arrives as `min c'x` (or max) with row bounds, variable bounds,
integrality, cones and quadratic terms. It leaves as a single
`A x + s = b`, `x, s ≥ 0`, with:

- **A and Q sparse, in CSC.** Dense materialization happens only on demand,
  for the small dense kernels.
- **Free variables split.** `x ∈ ℝ` becomes `x⁺ − x⁻` with both halves ≥ 0.
- **Ranged rows split** into a `<=` row and a `>=` row, each with its own slack.
- **Fixed variables substituted out**, the objective constant folded into
  `cfix`.

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

Ranged scalar variables and the members of a cone block need the same two-sided
treatment as ranged rows, and they get it here: `x = lo + u` with `u ≥ 0`,
closed by a cap row `u + s = up − lo`, `s ≥ 0`. An infinite bound degrades to
a one-sided row. This is what lets a conic model contain a variable bounded on
both sides without a second code path.

## Limits

- The quadratic form is stored as a symmetric dense block per quadratic row
  (`putqconk`), not as a sparse Cholesky factor: fine for the model sizes
  QCQP rows appear in, not for a wide sparse `Q`.
- Cone members are not split; the conic routes consume them whole.
