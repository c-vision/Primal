# Branch and bound

**Source:** `primal_mip.c`, `primal_mip_opt.c`

Depth-first branch and bound over conic relaxations, with cuts, bound
tightening, probing, strong branching and a primal heuristic. Relaxations are
solved by the same routes as the continuous problems, including the conic and
SDP paths — a MIP with a PSD block is not a special case.

## Node selection and branching

- **Best bound**, not deepest: the node with the lowest lower bound is
  expanded, ties going to the most recent (which keeps the previous DFS order).
- **Branch on `floor(x*)`** raw. Branching on a value that is already integral
  within the integrality tolerance makes the child equal to the parent, and the
  tree regenerates to the node cap — a measured failure, not a theoretical one.
- A child that does not tighten the node's box is not created.

## Two tolerances, not one

| Parameter | Default | Question it answers |
|---|---|---|
| `MIP_TOL_INTHER` | 1e-5 | may an integer constraint count as satisfied? |
| `MIP_TOL_FEAS` | 1e-6 | does this candidate really satisfy the model? |

Conflating them is not a cosmetic choice. With a single 1e-5 threshold and a
big-M disjunctive encoding, rounding a selector to `1 − 5e-6` moves its row by
5 — and the solver then publishes a point that is **not feasible**, with status
`INTEGER_OPTIMAL`. The feasibility threshold is what measures a candidate
before it is allowed to become an incumbent.

A candidate that fails is not discarded: the integers are pinned at their
rounded values and the node's relaxation is re-solved
(**fix-and-optimize**). That is still a relaxation of the node, so no integer
point is lost.

## Semi-continuous and semi-integer variables

The domain is `{0} ∪ [l, u]`. Two consequences, both learned the hard way:

- The root box is relaxed to `[0, u]`. A valid superset of `{0} ∪ [l,u]` — if it
  is not relaxed, the relaxation ignores the `x = 0` branch and the node's
  lower bound stops being a lower bound, so pruning becomes unsound.
- A **deactivated** variable sits legitimately below its own bound, so
  `PRIMAL_getprimalinfeas` measures the distance to the *domain* `{0} ∪ [l,u]`,
  not to the bound.

## Bound tightening

Every scalar row `l ≤ a'x ≤ u` yields implicit bounds on each variable
(`a_ij x_j ≤ u − smin` and `≥ l − smax`). Applied to the node's local arrays;
the published model is untouched.

Two rules that are not negotiable:

- Rows with **bar or quadratic** terms are skipped. The sparse store of scalar
  coefficients does not carry them, and a bound derived from a row that
  contains more is simply false — measured: the SDP MIP becomes infeasible.
- It does **not** run on the LP/QP path, because there the multiplier of the
  implicit bound belongs to the row that implied it. Publishing it as a
  variable-bound multiplier breaks the KKT of the original model. The LP/QP
  path gets the same tightening plus a dual repair, and the result is
  dual-safe.

## Cuts

| Cut | Source | When |
|---|---|---|
| Chvátal–Gomory | `Σ floor(a_j) x_j ≤ floor(u)` | non-negative integer columns |
| Cover | knapsack rows, greedy by coefficient | all-binary knapsack row |
| Gomory from the tableau | fractional basic integer variable | integer data only |

A cut is added to a **clone** of the task used only by the relaxations, so the
model the user reads back (`numcon`, `A`, bounds) is unchanged. A row that is
already integral produces itself as a cut and is skipped.

The tableau Gomory cut is valid for one round only: after a Gomory cut the
slacks have fractional coefficients, and the pure cut is no longer valid. The
mixed GMI variant is not implemented.

## Probing and heuristics

- **Probing**: for each binary variable (first 20), fix it to 0 and to 1 and
  re-solve the relaxation; if one side is infeasible the variable is fixed.
  Dual-safe, because the MIP publishes no duals.
- **Strong branching**: on the first 3 fractional candidates, by decreasing
  fraction, solve both children and branch on the variable with the best
  worst-case bound; a closed child scores `INF`. Beyond the first 200 nodes
  the most fractional variable wins, because the alternative costs 2 LP solves
  per node.
- **Primal heuristic**: at the root with no incumbent, round the node
  solution, pin the integers, repair with the relaxation. If the relaxation is
  an SDP, the `(x, X)` pair it returns is admissible for the node, hence for
  the model.

## Concurrency

`PRIMAL_IPAR_NUM_THREADS > 1` enables parallel probing, parallel strong
branching and a two-thread root decomposition. Each child thread solves a
**clone** with its own bounds and one thread, and the two children partition
the root region — so the best of the two is the optimum. The default is one
thread; the corpus is identical either way.

## Cap and status

A node cap (`MIP_MAX_NODES`, default 100000) is not a proof of anything. When
it trips:

- with an incumbent → `TRM_MAX_ITER` with an integer-feasible, not-proven-optimal status;
- **without** an incumbent → nothing is published at all; the status is
  `UNKNOWN`.

Publishing an incumbent as optimal at the cap is exactly the claim the status
cannot support.

## Measured

`sched` (the largest sample) does not close: 100 000 nodes, 27 left open, and
an incumbent of 97.328509 which is the true optimum but is **not proven**. The
sample asserts that status rather than pretending otherwise.

## Limits

- No Farkas rays for MIP ([certificates](certificates.md)).
- No conflict analysis, no pseudo-cost propagation, no Gomory mixed-integer
  cuts.
- No MIP with a quadratic objective on a conic relaxation beyond MIQP.
- The node cap is a hard stop, not a time limit; there is a `MAX_TIME`
  parameter but no user-visible guarantee attached to it.
