# QCQP encoding

Read the [convex QP chapter](quadratic_programming.md) for the Hessian
test and [the cone chapter](socp.md) for rotated cone geometry.
Here the goal is to replace a convex quadratic inequality by an
equivalent cone constraint. [Encoder source](../primal_quad.c)

## The rewriting

For an upper-bounded row $a^\top x+\tfrac12x^\top Qx\le b$, convexity
requires $Q\succeq0$. Write $Q=V\Lambda V^\top$ and
$z=\Lambda^{1/2}V^\top x$ using the positive eigenvalues. Introduce an
epigraph variable $w$ and impose

$$
(w,1,z)\in Q_{\mathrm{rot}},\qquad a^\top x+w\le b .
$$

The cone gives $2w\ge\|z\|_2^2=x^\top Qx$. An original feasible point
can choose $w=\tfrac12x^\top Qx$. Conversely, any encoded feasible point
satisfies the original row. The same argument encodes a lower-bounded
row by negating it, so its original $Q$ must be negative semidefinite.
An encoded quadratic objective uses the positive-semidefinite case
for minimization and the negative-semidefinite case for maximization.

## Worked example

Maximize $x$ subject to $x^2+x\le6$. The encoding introduces
$(w,1,\sqrt2x)\in Q_{\mathrm{rot}}$ and $x+w\le6$.
At $x=2$, choose $w=4$. Both constraints are tight. Conversely the
original inequality is $(x+3)(x-2)\le0$, so $x=2$ attains the
maximum. In the shadow task the coordinate $\sqrt2x$ is represented
by a new variable and an equality.

## Non-convex domains are refused

The encoder measures the scale of each spectrum and refuses the whole model
beyond a **relative** threshold `tol · max(1, max|λ|)` with
`tol = PRIMAL_DPAR_SEMIDEFINITE_TOL_APPROX` (default `1e-10`).
Debug tracing reports the offending entity and eigenvalue.

Two properties of that threshold are deliberate:

- It is **relative**, not absolute. An eigenvalue of `−2e-11` against a
  spectrum of 2 is noise and the model solves; an eigenvalue of `−1e-4` against
  a spectrum of `2e6` is the same relative distance and also solves. An
  absolute `1e-9` would have rejected the second.
- The encoder treats wrong-sign eigenvalues within this threshold as
  numerically flat. It also retains only eigenvalues above the
  implementation's magnitude cutoff for the expected sign. The
  equivalence above is exact for the retained form. `T87` checks
  rejection and near-flat cases.

## What it buys

Once encoded, a QCQP row is an ordinary conic row and the model is solved by
an eligible conic route. A QCQP with both PSD bars and quadratic rows
uses one unified conic model. [Unified SDP solver](semidefinite_programming.md)

## Limits

- Convexity only. There is no global branch-and-bound for non-convex quadratic
  programs; a non-convex domain is an error, not a slower answer.
- The dense symmetric block per quadratic row is `numvar × numvar`
  ([standard form](standard_form.md)).
- Only convex `Q` is representable on a `<=` row; equality and ranged quadratic
  rows are not encoded.

## Exercise

Encode $-x^2+x\ge-2$. Negating gives $x^2-x\le2$. Use the same
rotated cone as above, with $w-x\le2$. The feasible interval is
$[-1,2]$ and both endpoints can choose $w=x^2$.

## Reference

- S. Boyd and L. Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/), §§4.2–4.4.
