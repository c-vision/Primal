# From models to solver equations

**Verified against** revision `1f2ddee` (`stdform.c`, `stdform.h`,
`primal_optimize.c`, `primal_quad.c`, `primal.h`).

## Learning objectives and prerequisites

After this chapter you can:

- classify a model as LP, QP, QCQP, SOCP, SDP, EXP, POW or MIP from its data,
  and name the `primal.h` objects that carry each form;
- turn bounds and inequalities into the equality-and-orthant form
  $Ax=b$, $x\ge0$ by hand, with every sign, shift and slack explicit, and check the
  result against `stdform_build`;
- map a solution of the standard form back to the variables and duals of the
  original model, and measure it there;
- say when a quadratic constraint is rewritten into a cone.

Prerequisites: [mathematical foundations](mathematical_foundations.md) (convexity,
cones, duality).

## Model classes

| Class | Mathematical form | Where it lives in `primal.h` |
|---|---|---|
| LP | $\min c^\top x+c_f$ s.t. $l_c\le Ax\le u_c$, $l_x\le x\le u_x$ | `putcj`, `putcfix`, `putarow`, `putconbound`, `putvarbound` (keys `LO`, `UP`, `FX`, `FR`, `RA`) |
| QP | LP with $+\tfrac12x^\top Qx$, $Q\succeq0$ for a minimization ($Q\preceq0$ for a maximization) | `putqobj`: triplets of the lower triangle, the off-diagonal value is the full coefficient of $x_ix_j$ |
| Convex QCQP | rows $a_i^\top x+\tfrac12x^\top Q_ix\in[l_i,u_i]$; $Q_i\succeq0$ on an upper-bounded row, $Q_i\preceq0$ on a lower-bounded row | `putqconk`; fixed, ranged and free rows with $Q_i\ne0$ are rejected |
| SOCP | variables in a Lorentz cone $t\ge\|u\|_2$, or in a rotated cone $2x_1x_2\ge\sum_{i\ge3}x_i^2$ with $x_1,x_2\ge0$ | `appendcone` with `PRIMAL_CT_QUAD`, `PRIMAL_CT_RQUAD`; or affine expressions in a domain |
| SDP | $\min c^\top x+\sum_j\langle\bar C_j,X_j\rangle$ s.t. $l\le Ax+\sum_j\langle\bar A_{ij},X_j\rangle\le u$, $X_j\succeq0$ | `appendbarvars`, `putbarcj`, `putbaraij` |
| EXP | $(x_1,x_2,x_3)$ with $x_1\ge x_2e^{x_3/x_2}$, $x_2\ge0$ | `PRIMAL_CT_PEXP` ([exponential cone](exponential_cone.md)) |
| POW | $(x_1,x_2,x_3)$ with $x_1^\alpha x_2^{1-\alpha}\ge|x_3|$, $0<\alpha<1$; rotated: $2x_1^{2\alpha}x_2^{2(1-\alpha)}\ge x_3^2$ | `PRIMAL_CT_PPOW`, `PRIMAL_CT_RPOW` ([power cone](power_cone.md)) |
| MIP | any of the above plus $x_j\in\mathbb{Z}$, binary, $x_j\in\{0\}\cup[l,u]$ (semi-continuous), SOS1/SOS2 sets, disjunctions | `putvartype` (`INT`, `INT_BIN`, `SEMI_CONT`, `SEMI_INT`), `appendsos`, `appenddjcs` ([MIP](mixed_integer_optimization.md)) |

`RPOW` with $\alpha=\tfrac12$ is the rotated Lorentz cone. A model can mix
classes: a quadratic row next to a cone, or integers on top of an SDP.

## The standard form

The continuous LP/QP engines consume

$$
\min\ \tfrac12\tilde x^\top\tilde Q\tilde x+\tilde c^\top\tilde x\quad\text{s.t.}\quad
\tilde A\tilde x=\tilde b,\ \ \tilde x\ge0,\ \ \tilde b\ge0 .
$$

`stdform_build` (`stdform.c`) produces it from the original data. The rules,
applied in this order:

1. **Fixed variables** ($l_x=u_x$) are substituted out: their contribution moves
   into the constant, the objective and the right-hand sides.
2. **Finite lower bound:** $x=l_x+\tilde x$ (a *shift*). **Only an upper bound:**
   $x=u_x-\tilde x$ (a *negated column*). **Free:** $x=\tilde x^+-\tilde x^-$ (two
   columns).
3. **A finite upper bound on a shifted variable** becomes an extra row
   $\tilde x+s=u_x-l_x$ with a private slack $s$ (kind `VARUB`).
4. **Constraint rows.** A free row is dropped. A fixed row $a^\top x=l_c$ stays an
   equality. A lower-bounded row $a^\top x\ge l_c$ becomes $a^\top x-s=l_c$ (kind
   `LO`); an upper-bounded row $a^\top x\le u_c$ becomes $a^\top x+s=u_c$ (`UP`); a
   ranged row gives one of each.
5. **Normalization.** A row with negative right-hand side is multiplied by $-1$
   ($\sigma=-1$) so that $\tilde b\ge0$.

For an affine change of variables $x=x_0+T\tilde x$ and a symmetric original
Hessian $Q$, the new Hessian is $\tilde Q=T^\top QT$, the new linear term is
$\tilde c=T^\top(c+Qx_0)$ and the new constant is $c^\top x_0+\tfrac12x_0^\top Qx_0$. The
right-hand sides are shifted by $Ax_0$. A *direction* transforms by $T$ alone,
without the shift. `Q` is stored as a lower-triangular CSC matrix with an explicit
diagonal in every column, because the sparse QP engine adds $D$ to that diagonal
directly.

```text
STDFORM(model)                            # stdform_build
 1  for each variable j:  classify it (fixed / shifted / negated / split)
 2  for each constraint i: emit its LO and/or UP rows with one slack each
 3  for each shifted variable with a finite upper bound: emit a VARUB row
 4  for each row with b < 0: negate it, record sigma = -1
 5  build  A~ (CSC)  and  Q~ = T'QT  (lower CSC)  and  c~
 6  record, for every row and column, which constraint or variable it came from
```

Step 6 is what makes the return trip possible.

### Before the standard form

`opt_routes` (`primal_optimize.c`) does not build the standard form from the raw
model. For an LP or QP it first removes redundant rows (`row_redundant`), tightens
the variable bounds from the rows (`bound_tighten`, remembering which row implied
each tightened bound so that the duals can be repaired), equilibrates rows and columns
by powers of two (`scale_equilibrate`), and only then calls `stdform_build`; an LP is
then presolved ([presolve](presolve.md)). The tightened bounds are part of the data
that `stdform_build` sees, which is why the polygon LP of the [simplex
chapter](simplex.md) ends with 4 rows and 6 columns instead of 2 and 4.

## Worked example: from the inequalities to the standard form and back

Minimize $\tfrac12v^2$ subject to $1\le v\le4$ (a variable range) and the row
$v\ge2$. The example calls `stdform_build` directly, so that every transformation is
visible. In `PRIMAL_optimize` `bound_tighten` first tightens one bound of this model
(the log reads `LP: 1 tightened bounds`), so the standard form PRIMAL really solves
is smaller than the one below.

**Forward.** $v$ has a finite lower bound, so $v=1+\tilde x$ with $T=1$, $x_0=1$.
Then $\tilde Q=1$, $\tilde c=T(c+Qx_0)=1\cdot(0+1)=1$ and the constant is
$\tfrac12\cdot1\cdot1^2=\tfrac12$: the objective is $\tfrac12\tilde x^2+\tilde x+\tfrac12$.
The row $v\ge2$ is a `LO` row: $v-s_1=2$, i.e. $\tilde x-s_1=1$. The upper bound $v\le4$
is a `VARUB` row: $\tilde x+s_2=3$. All right-hand sides are positive, so no row is
negated. `stdform_build` returns exactly this, with $m=2$ rows, $n=3$ columns and
`cfix` $=0.5$:

| row | kind | $\tilde x$ | $s_1$ | $s_2$ | rhs |
|---|---|---:|---:|---:|---:|
| 0 | `LO` of constraint 0 | 1 | −1 | 0 | 1 |
| 1 | `VARUB` of variable 0 | 1 | 0 | 1 | 3 |

**Solve.** The standard-form objective is increasing for $\tilde x\ge0$, so the
lower row binds: $\tilde x=1$, hence $s_1=0$ and $s_2=2$.

**Back.** `stdform_map_x` gives $v=\text{shift}+\tau\tilde x=1+1=2$. The objective in the
original model is $\tfrac12\cdot2^2=2$, which equals
$\tfrac12\tilde x^2+\tilde x+\tfrac12=\tfrac12+1+\tfrac12$.

**Duals back.** Stationarity of the standard form,
$\tilde Q\tilde x+\tilde c-\tilde A^\top\tilde y-\tilde z=0$, with $\tilde x=1>0$
(so $\tilde z_1=0$) and $s_2=2>0$ (so $\tilde z_3=0$), gives $\tilde y=(2,0)$ and
$\tilde z_2=\tilde y_1=2\ge0$. `stdform_map_y` maps row duals back with
$y_i=-\sum_r\sigma_r\tilde y_r$ over the rows that came from constraint $i$ (the
`VARUB` rows are skipped), so the original row gets $-2$. `PRIMAL_optimize` on the
original model reports $v=2$, objective 2, and $y=\texttt{slc}=-2$ for the row, with
primal and dual infeasibility 0. The sign follows the `primal.h` convention
($y<0$ for a row active at its lower bound; see
[mathematical foundations](mathematical_foundations.md)).

```mermaid
flowchart LR
    M["Original model"] --> R["row_redundant, bound_tighten,<br/>scaling"]
    R --> S["Standard form<br/>stdform_build"]
    S --> E["Engine<br/>simplex or interior point"]
    E --> P["Map back<br/>stdform_map_x, stdform_map_y"]
    P --> V["Measure in the original model<br/>(infeasibilities, objectives)"]
```

*Figure: the round trip. The standard form is an intermediate representation; the
published point, the duals and the infeasibility measures are all expressed in the
original model.*

## Conic forms and quadratic rows

A *conic* model keeps its cones: a Lorentz or rotated cone is a constraint on
variables (`appendcone`) or on affine expressions of the variables
(`appendacc`, [`primal_afe.c`](../primal_afe.c)). An SDP carries symmetric
matrix variables with the trace inner product. Such a model is not sent to
`stdform_build`; it follows the routes of [the architecture chapter](architecture.md).

A convex quadratic *row* is rewritten exactly into a rotated cone by
[`primal_quad.c`](../primal_quad.c). For $Q=R^\top R\succeq0$,

$$
\tfrac12v^\top Qv+a^\top v\le b\iff\bigl(b-a^\top v,\ 1,\ Rv\bigr)\in\mathcal Q_r ,
$$

where $\mathcal Q_r=\{(u,w,y):2uw\ge\|y\|^2,\ u,w\ge0\}$ (the left side with $w=1$ reads
$2(b-a^\top v)\ge\|Rv\|^2=v^\top Qv$). PRIMAL's encoder uses the eigendecomposition
$Q=V\Lambda V^\top$ for $R$, adds one variable per retained eigenvalue and a variable fixed
at 1; the details, the convexity test and the refusal of non-convex rows are in
[quadratic programming](quadratic_programming.md#convex-qcqp-rewriting-into-rotated-cones).
The map is exact; its cost is extra variables, extra equality rows and fill, even though
nothing is approximated.

**A small SOCP, checked by hand.** Minimize $t$ subject to $x_1+x_2=10$ and
$(t,x_1,x_2)$ in the Lorentz cone, i.e. $t\ge\sqrt{x_1^2+x_2^2}$. The distance from the
origin to the line is attained at $x_1=x_2=5$, so $t=5\sqrt2=10/\sqrt2\approx7.071$. This is
test `T22`.

## Where each class goes

| Model | Route (`opt_routes`) |
|---|---|
| any integer, semi-continuous or SOS member | branch and bound ([MIP](mixed_integer_optimization.md)) |
| PSD bar variables | native SDP interior point, then tangent cuts ([SDP](semidefinite_programming.md)) |
| quadratic rows, or a quadratic objective with cones | rotated-cone encoding, then the conic route |
| exponential or power cones | native barrier in `sdp.c`, then tangent cuts ([exponential cone](exponential_cone.md)) |
| Lorentz and rotated cones | conic route ([SOCP](conic-socp.md)) |
| LP, QP | the standard form of this chapter ([simplex](simplex.md), [interior point](interior-point.md)) |

Free-variable splitting, slack insertion and the `VARUB` rows increase the
dimension even when a model has few original rows: $n$ grows by one column per
inequality row, per finite two-sided bound and per free variable.

## Exercise

Write the standard form of: minimize $x_1+x_2$ subject to $x_1+2x_2\ge4$, $0\le x_1\le3$,
$x_2$ free. Count rows and columns, then map the solution $\tilde x$ back.

*Answer.* $x_1=\tilde x_1$ (lower bound 0), $x_2=\tilde x_2^+-\tilde x_2^-$. Rows:
`LO`: $\tilde x_1+2\tilde x_2^+-2\tilde x_2^--s_1=4$; `VARUB`: $\tilde x_1+s_2=3$. That is
$m=2$ rows and $n=5$ columns, with cost $(1,1,-1,0,0)$ (verified with
`stdform_build`). The solution is $\tilde x=(0,2,0,0,3)$, mapping back to
$x_1=0$, $x_2=2-0=2$ with objective 2; `PRIMAL_optimize` reports $x=(0,2)$, objective 2
and the row multiplier $-\tfrac12$.

## References

- S. Boyd and L. Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/), Cambridge University Press, 2004, chapters 3–4 and appendix B.
- D. Bertsimas and J. N. Tsitsiklis, *Introduction to Linear Optimization*, Athena Scientific, 1997, section 1.1 (standard form).
