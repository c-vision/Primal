# Certificates of infeasibility, unboundedness and optimality

**Verified against** revision `1f2ddee` plus the presolve correction `2c48c87`, with the measurements taken at `24b408a` and the user-space ray check of `T306`: `primal_std.c`, `primal_optimize.c`, `primal_verdict.c`,
`primal_sdptask.c`, `presolve.c`, `sdp.c`, `primal.h`. The vectors quoted below were returned by `PRIMAL_optimize`
and the ray getters.

## Learning objectives and prerequisites

After this chapter you can:

- state Farkas' alternatives for the standard form and for a cone, and check a given vector against them by substitution;
- say what PRIMAL's ray getters promise, how a candidate ray is normalized and measured, and in which cases a verdict is
  returned **without** a vector;
- follow a candidate ray from the engine through the presolve lift and the measurement of `ray_publish`;
- read the five-case table of what the public API returns for infeasible and unbounded LP, SOC and SDP models;
- explain the fixed-diagonal optimality certificate of the SDP route as a measured certificate.

Prerequisites: [duality and dual cones](mathematical_foundations.md), [presolve](presolve.md) (the homogeneous lift),
[simplex](simplex.md).

## Farkas alternatives

For the standard form $Ax=b,\ x\ge0$ exactly one of the two systems has a solution:

$$
\text{(P)}\ \ Ax=b,\ x\ge0\qquad\text{or}\qquad
\text{(D)}\ \ A^\top y\le0,\ \ b^\top y>0 .
$$

They cannot both hold: $0\ge(A^\top y)^\top x=y^\top Ax=y^\top b>0$. A vector $y$ satisfying (D) *proves* that (P) is infeasible and
can be checked by two matrix–vector products, which is why the solver reports it instead of just asserting infeasibility. Dually, if
(P) is feasible, a direction $\rho\ge0$ with $A\rho=0$ and $c^\top\rho<0$ is a **recession direction** along which
$\min c^\top x$ is unbounded: $x+\lambda\rho$ stays feasible and $c^\top(x+\lambda\rho)\to-\infty$.

For $\min c^\top x$, $Ax=b$, $x\in\mathcal K$ the same argument gives

$$
\text{infeasible: }\exists y:\ -A^\top y\in\mathcal K^*,\ b^\top y>0;\qquad
\text{unbounded: }\exists\rho\in\mathcal K:\ A\rho=0,\ c^\top\rho<0 .
$$

(If $x\in\mathcal K$ is feasible then $y^\top b=(A^\top y)^\top x\le0$.) For $\mathcal K=\mathbb R^n_+$, $\mathcal K^*=\mathbb R^n_+$ and this is (D).
In the SDP case $\mathcal K=\mathbb S^n_+$ and $A^\top y=\sum_ky_kA_k$ is a matrix.

The two alternatives are not exhaustive for conic problems: a conic system can be infeasible without a strict certificate (a limit of
certificates) and the solver may fail to find any vector. Not having a certificate is therefore *not* a statement about the model.

## What the ray getters promise

[`primal.h`](../primal.h#L1962) states the contract: `PRIMAL_getdualray` fills one entry per constraint with a vector measured as
$\sum_iy_ib_i^{\rm active}>0$ and $A^\top y\le0$ on every nonnegative variable; `PRIMAL_getprimalray` fills one entry per
variable with a recession direction ($(A\rho)_i\le0$ on a row with an upper bound, $\ge0$ with a lower bound, $=0$ on a ranged or
fixed row; $\rho_j\ge0$ if $x_j$ has a finite lower bound, $\le0$ with a finite upper bound; and $s\,c^\top\rho<0$ with
$s=+1$ for minimization). Both are scaled to $\max|\cdot|=1$. A getter answers `PRIMAL_RES_ERR_ARG` when no vector was published.
The status names the vector: `PRIMAL_SOL_STA_PRIM_INFEAS_CER` / `DUAL_INFEAS_CER` appear **only** with the vector beside them.
Where a verdict exists without a vector, `solsta` stays `UNKNOWN` and the verdict is read with `PRIMAL_getprosta`.

## One candidate, measured and published

```text
RAY-PUBLISH(engine status, candidate y, candidate rho)                  # primal_std.c:619
 1  y   <- dual candidate in the standard form;   rho <- primal candidate
 2  dual_ray_measured(y):   y <- y / max|y|;
        A'y <= 1e-8 (1 + max column sum |A|)  componentwise   and   b'y > 1e-8 (1 + sum|b|)
 3  ray_representable(y):   reject if y puts weight on a row that came from x_j <= u_j,
        or on BOTH sides of a ranged constraint (no one-number-per-constraint image)
 4  prim_ray_measured(rho): rho <- rho / max|rho|;  rho_j >= -1e-8(1+maxcol);
        |A rho|_inf <= 1e-7 max|A| n   and   c'rho < -1e-8 (1 + sum|c|)
 5  if the dual candidate measured:   map it to the user's rows (stdform_map_y, row scaling and sign), publish, return INFEASIBLE
 6  else if the primal candidate measured:   map to the user's columns (stdform_map_dir, column scaling), publish, return UNBOUNDED
        (infeasibility dominates unboundedness: a ray is a statement about the feasible set)
 7  else return the engine's own status: it stands, only the vector is withheld
```

Where the candidates come from: the **tableau simplex** writes its own witness (an artificial objective $>0$ at the end of phase I gives
$y$; a ratio test with no leaving row gives $\rho$). The **interior-point** routes write none and offer their last iterate to the same
measurement. When presolve removed singleton rows or columns the candidate is lifted by the linear, constant-free pass
`presolve_postsolve_dir` ([presolve](presolve.md)) before measurement. A status of INFEASIBLE or UNBOUNDED from the engine is kept
("a positive phase-1 optimum and a ratio test with no leaving row are proofs about the system that was solved"): what the measurement
decides is whether a vector goes with it.

## Worked examples, measured

**Infeasible LP.** $x=2,\ x+y=5,\ y+z=1,\ x,y,z\ge0$. Then $y=3$ and $z=-2$: infeasible. By substitution,
$y^\star=(-1,1,-1)$ gives $A^\top y^\star=(-1+1,\ 1-1,\ -1)=(0,0,-1)\le0$ and $b^\top y^\star=-2+5-1=2>0$. `PRIMAL_optimize` returns
`ERR_INFEASIBLE` (1002), `solsta=PRIM_INFEAS_CER` and `PRIMAL_getdualray` returns exactly $(-1,1,-1)$ at presolve levels 0, 1 and 2
(test `T305`, which is also the regression test of [the presolve correction](presolve.md#right-hand-sides-that-become-negative)).

**Unbounded LP.** $\min-x_1-2x_2$ s.t. $-x_1+x_2\le1$, $x\ge0$. The direction $\rho=(1,1)$: $A\rho=-1+1=0\le0$,
$\rho\ge0$, $c^\top\rho=-3<0$. `PRIMAL_optimize` returns `ERR_UNBOUNDED` (1003), `solsta=DUAL_INFEAS_CER` and `PRIMAL_getprimalray`
returns $(1,1)$ (so does $(2,1)$ scaled to $(1,0.5)$, also a valid ray: the certificate is a *cone* of directions).

**Unbounded SOC.** $\min-x_0$ with $(x_0,x_1,x_2)\in Q_3$. The direction $\rho=(1,0,1)$ is in $Q_3$ (boundary: $1=\|(0,1)\|$),
$c^\top\rho=-1<0$. The solver returns it as the primal ray.

**Infeasible conic, with a dual ray.** $x_0=-1$ with $x_0$ in the first coordinate of $Q_3$. The vector $y=-1$ gives
$-A^\top y=(1,0,0)\in Q_3=\mathcal K^*$ and $b^\top y=(-1)(-1)=1>0$: the conic Farkas condition of the formulas above. `PRIMAL_getdualray` returns
$y=-1$ here and also on the LP route for $x_0=-1,\ x_0\ge0$: **one sign on both routes**, the one of `primal.h` ($y>0$ on a lower side, $y<0$ on an upper side,
$-A^\top y\in\mathcal K^*$, $\sum y_ib_i^{\rm active}>0$). The conic route measures the textbook form ($y\ge0$ on an upper side, $d=A^\top y\in\mathcal K^*$, signed right-hand side
negative) and publishes its negative; before that was done (test `T307`) it returned $y=+1$, the opposite of the LP route.

**Infeasible SDP.** $X_{11}=1,\ X_{22}=1,\ X_{12}=2,\ X\succeq0$: $\det X=1-4<0$. By hand, $y=(-1,-1,\tfrac32)$ gives
$-\sum_ky_kA_k=\begin{pmatrix}1&-\tfrac34\\-\tfrac34&1\end{pmatrix}\succeq0$ (with $A_3=\tfrac12(E_{12}+E_{21})$) and
$b^\top y=-1-1+2\cdot\tfrac32=1>0$. `PRIMAL_optimize` returns `ERR_INFEASIBLE` and `PRIMAL_getprosta` reports `PRIM_INFEAS`, but **no vector**:
`solsta=UNKNOWN`, both ray getters answer `ERR_ARG` (the compressed bar block has no image as one number per constraint).

**A conic model with no verdict.** $x_0=x_1=x_2=1$, $(x_0,x_1,x_2)\in Q_3$ is infeasible ($1<\sqrt2$) but the route ends with
`TRM_MAX_ITER` (1007), `solsta=UNKNOWN`, `prosta=UNKNOWN`: the line search could find no descent step and neither relaxation (see next
section) proved anything. The model is infeasible; the solver says nothing, which is the honest answer when it cannot measure a witness.

| model | return | `solsta` | `prosta` | dual ray | primal ray |
|---|---|---|---|---|---|
| infeasible LP ($x+y\le1,\ x+y\ge2$, presolve off) | 1002 | 5 `PRIM_INFEAS_CER` | 4 | $(-1,1)$ | no |
| unbounded LP | 1003 | 6 `DUAL_INFEAS_CER` | 5 | no | $(1,1)$ |
| infeasible SOC ($x_0=-1$) | 1002 | 5 | 4 | $y=-1$ | no |
| unbounded SOC | 1003 | 6 | 5 | no | $(1,0,1)$ |
| infeasible SDP | 1002 | 0 `UNKNOWN` | 4 | no | no |
| SOC infeasible but not proved | 1007 | 0 | 0 | no | no |

*Table: measured with `PRIMAL_optimize`, `PRIMAL_getsolsta`, `PRIMAL_getprosta` and the two getters at `24b408a`.*

> **Ray re-measured on the user's model.** The first row is only certified with presolve off. With presolve on (levels 1 and 2),
> `bound_tighten` derives $x_1\ge1$ and $x_2\le1$ from the two rows and fixes both variables, and the engine proves infeasibility
> against those *derived* bounds: the vector $(-1,0)$ it returns measures in the standard form but is not a certificate of the rows the user wrote
> (the first row alone is feasible). Before the correction tested by `T306` that vector was published; now `ray_publish`
> re-measures every dual ray on the model's own rows and bounds ($L=\sum_{y_i>0}y_i\,l_i+\sum_{y_i<0}y_i\,u_i$ against
> $U=\sum_{d_j>0}d_jx^{\max}_j+\sum_{d_j<0}d_jx^{\min}_j$ with $d=A^\top y$, and $L>U$) and withholds a ray that fails. The verdict stays readable
> (`rc=1002`, `prosta=PRIM_INFEAS`, `solsta=UNKNOWN`, `PRIMAL_getdualray` answers `ERR_ARG`); the row of the table above is therefore a *with-presolve-off* row, and with
> presolve on this model has a verdict and no vector. The ray is not recovered by a re-solve without bound tightening.

## Conic and SDP routes: verdicts measured on relaxations

A conic route that ends without a point answers two questions about the *model* on linear relaxations, each chosen so that its answer
is sound ([`conic_no_answer_verdict`](../primal_verdict.c#L359)): infeasibility is tested on an LP that **contains** the cones (an LP infeasible on a
subset proves nothing), unboundedness on one **inside** the cones (a direction of a wider set is not a direction of the model). A dual
vector is published only if $d\in\mathcal K^*$ holds on every cone block with the signed right-hand side negative
([`conic_dual_ray_measures`](../primal_verdict.c#L196)); a recession direction is published as the user's primal ray only for a model with no
bar variables ([`conic_publish_pray`](../primal_verdict.c#L259)). Otherwise the verdict is `prosta` alone. In the tangent-cut SDP route a point sitting on the artificial box cap
$\pm$`SDP_BIGM` is not an answer: [`bar_cap_verdict`](../primal_sdptask.c#L1096) publishes an unbounded verdict only when a
recession direction is measured on the model's own rows and cones, and otherwise no verdict ([SDP](semidefinite_programming.md)).
Mixed-integer infeasibility and the tangent-cut route likewise leave the verdict in `prosta` (relaxation rays do not certify integer infeasibility).

## An optimality certificate that is measured: the fixed-diagonal SDP

Certificates are not only for failure. For $\min\langle C,X\rangle$ with fixed positive diagonals, [`sdp_fixed_diagonal`](../sdp.c#L2995) builds
$X=\hat u\hat u^\top$ and a dual slack $S$ with $S\hat u=0$ and a Laplacian structure, and **measures** the primal residual, the dual residual
and the gap against the same tolerances as the iterative route; only if all pass is the point returned
([SDP chapter](semidefinite_programming.md#the-fixed-diagonal-certificate-no-iteration)). On the example $C=\begin{pmatrix}0&-1\\-1&0\end{pmatrix}$ it
returns $X^\star=\begin{pmatrix}1&1\\1&1\end{pmatrix}$, $S^\star=\begin{pmatrix}1&-1\\-1&1\end{pmatrix}$ with primal and dual objective $-2$: the pair satisfies
$\mathcal A(X)=b$, $S=C-\mathcal A^*(y)\succeq0$, $\langle X,S\rangle=0$, which is optimality by weak duality.

## Costs and limits

- A measurement costs two sparse products; it is negligible next to the solve.
- A tolerance family $10^{-8}$ (relative to the column and right-hand side scales) separates a certificate from a near miss. A ray that misses
  is not "almost a proof": the status falls back to the engine's claim without a vector, or to `UNKNOWN`.
- Variable-bound rows and two-sided ranged rows have no one-number-per-constraint dual image, so a dual ray that needs them is withheld
  (`ray_representable`).
- For conic, SDP, MIP and tangent-cut routes a verdict can exist without a vector, by design.

## Exercise

For the infeasible LP $x_1+x_2\le1,\ x_1+x_2\ge2,\ x\ge0$ write it in the standard form with a slack and a surplus, find a vector satisfying
(D) and compare it with what the solver returns with presolve off.

*Answer.* $x_1+x_2+s_1=1$, $x_1+x_2-s_2=2$, so $A=\begin{pmatrix}1&1&1&0\\1&1&0&-1\end{pmatrix}$, $b=(1,2)$. Conditions: $A^\top y=(y_1+y_2,\ y_1+y_2,\ y_1,\ -y_2)\le0$, i.e.
$y_1\le0$, $y_2\ge0$, $y_1+y_2\le0$; and $b^\top y=y_1+2y_2>0$. Take $y=(-1,1)$: $A^\top y=(0,0,-1,-1)\le0$ and $b^\top y=1>0$. With presolve off
the solver returns $(-1,1)$ for the two user rows (upper row $x_1+x_2\le1$ first), the same vector.

## References

- J. Farkas, [Theorie der einfachen Ungleichungen](https://doi.org/10.1515/crll.1902.124.1), *Journal für die reine und angewandte Mathematik* 124:1–27, 1902.
- A. Schrijver, *Theory of Linear and Integer Programming*, Wiley, 1986 (reprinted 1998).
- S. Boyd and L. Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/), Cambridge University Press, 2004, section 5.8.
