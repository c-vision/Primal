# The exponential cone

The barrier examples and figures were measured at revision `1f2ddee`. The
reduced Newton section describes the implementation verified by test `T313`
and the [current cone matrix](../README.md#soc-exponential-and-mixed-cone-matrix).

## Learning objectives and prerequisites

After this chapter you can:

- state the exponential cone and its dual and test a point for membership;
- write the 3-self-concordant barrier, its gradient and Hessian, and recover the
  *scaling point* $W$ of a dual vector $s$ by a one-dimensional Newton iteration;
- explain why this cone cannot use the Nesterov–Todd scaling of the symmetric cones,
  what PRIMAL uses instead, and what the higher-order correction adds;
- eliminate fixed-scale exponential epigraphs and reconstruct a conic dual
  from the reduced smooth problem;
- follow a model with an exponential cone through the routes of PRIMAL, including the
  retry in secant mode and the tangent-cut fallback.

Prerequisites: [conic duality](mathematical_foundations.md), [SOCP](socp.md) and
[SDP](semidefinite_programming.md) (the same predictor–corrector and the same KKT engine).

## The cone and its dual

PRIMAL's cone `PEXP` acts on three coordinates $(t,u,v)$. The inequality
$t\ge u\cdot e^{v/u}$ says that $t$ lies on or above the exponential curve
at a fixed positive scale $u$. Here $\mathrm{cl}$ denotes closure
and $\mathrm{int}$ denotes interior.

$$
\mathcal K_{\exp}=\mathrm{cl}\left(\lbrace(t,u,v):\ t\ge u\cdot e^{v/u},\ u>0\rbrace\right).
$$

$$
\mathrm{int}(\mathcal K_{\exp})
=\lbrace(t,u,v):\ u>0,\ t>0,\ \psi=u\ln(t/u)-v>0\rbrace .
$$

For $u>0$, $v$ can have either sign and $t$ must be positive. The closure
adds the face $\lbrace(t,0,v):t\ge0,\ v\le0\rbrace$. As $u$ tends to zero from
above, $u e^{v/u}$ tends to zero for fixed $v\le0$ and diverges for fixed $v>0$.
The cone is convex because $(u,v)\mapsto u\,e^{v/u}$ is the perspective of $e^{v}$.
The header lists these coordinates in the same order. [`primal.h`](../primal.h#L188)

`DEXP` is *minus* this
cone, $x\in\text{DEXP}\iff-x\in\text{PEXP}$ (`primal_sdptask.c` encodes it with $z_a=-x_{\text{mem}[a]}$);
it is what the header of `primal.h` calls "dual", but it is a mirror image, not $\mathcal K^*_{\exp}$.

![Sections of the exponential cone at u = 1 and u = 2](figures/exp_cone_section.svg)

*Figure: the sections of $\mathcal K_{\exp}$ at heights $u=1$ (shaded above the curve $t=e^v$) and
$u=2$ (curve $t=2e^{v/2}$) in the $(v,t)$ plane. Each section is the epigraph of a convex curve, and the section at
height $u$ is the one at $u=1$ stretched by $u$ in both axes. The black point is the optimum
$(t,u,v)=(2,2,0)$ of the example below, which lies on the boundary. Analytic figure by
[`figures/make_figures.c`](figures/make_figures.c) at `1f2ddee`; the generator tests 40 boundary points on
both sides with `expcone_barrier` and checks the optimum against `PRIMAL_optimize`.*

![Animated sections of the exponential cone as u increases](figures/exponential_sections.gif)

*Animation: hold $u$ fixed in each frame and raise it between frames. The point
at $v=0$ follows $t=u$. The curve moves up and broadens because both axes are
scaled by $u$. The shading marks the feasible side $t\ge u e^{v/u}$.*

The cone is not self-dual. Its dual is

$$
\mathcal K_{\exp}^*=\lbrace s:\ s_0>0,\ s_2<0,\ s_1-s_2+s_2\ln(-s_2/s_0)\ge0\rbrace \ \cup\ \lbrace (s_0,s_1,0):s_0,s_1\ge0\rbrace ,
$$

and `expcone_dual_in` ([`expcone.c:238`](../expcone.c#L238)) tests its interior after dividing $s$ by
$\|s\|_1$, so the test does not depend on the scale of the pair. A pair $(z,s)\in\mathcal K\times\mathcal K^*$
has $\langle z,s\rangle\ge0$. Equality means the two vectors are orthogonal.
If both are nonzero, equality forces each to the boundary of its cone.

## Barrier, gradient and Hessian

The barrier is the 3-self-concordant barrier of Nesterov and Chares,

$$
f(t,u,v)=-\ln\psi-\ln t-\ln u,\qquad\psi=u\ln(t/u)-v,
$$

implemented in [`pexp_core`](../expcone.c#L36). With $\psi_t=u/t$, $\psi_u=\ln(t/u)-1$, $\psi_v=-1$:

$$
g=\nabla f=(-\tfrac{\psi_t}{\psi}-\tfrac1t,\ -\tfrac{\psi_u}{\psi}-\tfrac1u,\ \tfrac1\psi),\qquad
H=\frac{\nabla\psi\nabla\psi^\top}{\psi^2}-\frac{\nabla^2\psi}{\psi}+\mathrm{diag}\!(\tfrac1{t^2},\tfrac1{u^2},0),
$$

with $\nabla^2\psi$ having the nonzero entries $\psi_{tt}=-u/t^2$, $\psi_{tu}=1/t$, $\psi_{uu}=-1/u$.
Because $f(\lambda z)=f(z)-3\ln\lambda$ it satisfies $\langle g,z\rangle=-3$ and $Hz=-g$.

**Numbers.** At $z=(3,1,\tfrac12)$: $\psi=\ln3-\tfrac12=0.5986123$, $f=-0.5854711$,
$g=(-0.8901768,\,-1.1647348,\,1.6705304)$ and

$$
H=\begin{pmatrix}0.60680&-0.46511&-0.93022\\-0.46511&2.69767&-0.27519\\-0.93022&-0.27519&2.79067\end{pmatrix}.
$$

Evaluated with `expcone_barrier`, `expcone_grad`, `expcone_hess`: $\langle g,z\rangle=-3.0000000000$ and
$|Hz+g|\le2.2\cdot10^{-16}$, the two identities above.

**Why $\nu=3$.** The comment in [`expcone.c`](../expcone.c#L25) records that an earlier barrier
$-\ln(t-ue^{v/u})-\ln u$ has parameter 2 but is *not* self-concordant (along random lines
$|f'''|/(2f''^{3/2})$ reaches 2.54 near the boundary, instead of staying $\le1$), and gives the geometric argument that no
barrier of this cone can have $\nu<3$ (the faces $t=0$ and $u=0$ meet at $(0,0,-1)$). Without
self-concordance the Newton steps lost centrality: the max-entropy family fell to the tangent cuts at
$N=30,\dots,300$ before the change. These statements are the code's own record; this chapter does not
reprove them.

## Scaling point of a dual vector

For $s\in\mathrm{int}\mathcal K^*$ there is a unique $W\in\mathrm{int}\mathcal K$ with $-\nabla f(W)=s$
([`expcone_dual_point`](../expcone.c#L274)); it replaces the NT point, which does not exist for a
non-symmetric cone. Solve the three equations of $-g=s$ in order:

$$
s_2=-\tfrac1{\psi_w}\ \Rightarrow\ \psi_w=-\tfrac1{s_2};\qquad
s_0=\tfrac{1-u_ws_2}{t_w}\ \Rightarrow\ t_w=\tfrac{1-u_ws_2}{s_0}.
$$

Writing $a=1/u_w$, the second row becomes a scalar equation

$$
\gamma(a)=a-s_2\ln\frac{a-s_2}{s_0}+s_2-s_1=0,\qquad
\gamma'(a)=1-\frac{s_2}{a-s_2}>0,\quad\gamma''(a)=\frac{s_2}{(a-s_2)^2}<0 .
$$

$\gamma$ is increasing and concave, and $\gamma(0)=-(s_1-s_2+s_2\ln(-s_2/s_0))<0$ exactly when $s$ is in the
interior of the dual cone. Newton's method from $a=0$ therefore increases monotonically to the unique root, after which
$u_w=1/a$, $t_w=(1-u_ws_2)/s_0$ and $v_w=u_w\ln(t_w/u_w)+1/s_2$.

**Check.** Take $s=-g(z)=(0.8901768,\,1.1647348,\,-1.6705304)$ for the $z$ above. Newton on $\gamma$ from $a=0$ gives
$a=0,\ 0.8918529,\ 0.9991239,\ 0.9999999447,\ 1.0000000000$ with $\gamma=-1.78,\,-0.177,\,-1.4\cdot10^{-3},\,-9\cdot10^{-8},\,-4\cdot10^{-16}$:
monotone, and $u_w=1$. `expcone_dual_point` returns $W=(3,1,0.5)=z$ to ten digits, as it must on the central path where
$s=-\mu g(z)$ up to the factor $\mu$.

## The Newton row of a cone block

On the central path $s=-\mu\nabla f(z)$ and $\langle z,s\rangle=\nu\mu$. A non-symmetric cone has no Jordan product, so PRIMAL
keeps the cone equation in the KKT matrix and builds a row (header of [`expcone.h`](../expcone.h)):

$$
\Theta_{\rm in}\,\Delta z+\Delta s/\text{scale}=r_t,\qquad
\text{scale}=\tfrac{\langle z,s\rangle}{\nu},\quad s_n=\tfrac{s}{\text{scale}},\quad
r_t=\frac{-s-\sigma\mu\,\nabla f(z)}{\text{scale}},
$$

where $\Theta_{\rm in}=M(z,s_n)$ is either the barrier Hessian $H_z$ or the Hessian-NT chain
$H_s^{-1/2}(H_s^{1/2}H_zH_s^{1/2})^{1/2}H_s^{-1/2}$ with $H_s=\nabla^2f_*(s_n)=(\nabla^2f(W))^{-1}$. The chain is used when
$s_n$ is in the interior of the dual cone ([`expcone_scaling`](../expcone.c#L530) returns which branch). Two devices keep the
block of order 1: $M$ is evaluated at the $\mu$-free direction $s_n$, and the whole row is divided by scale. The result is that
$\Theta_{\rm in}$ depends only on the iterate, never on $\sigma$ or $\mu$: predictor and corrector share the matrix and
differ in $r_t$. At a paired point the chain equals $\nabla^2f(z)$ (tests `T78`, `T80`).

The step lengths come from membership tests: `expcone_maxstep` bisects (50 halvings) on `expcone_barrier` for the primal,
and `expcone_dual_maxstep` bisects on `expcone_dual_in` for the dual, since testing the dual in $\mathcal K$ would clamp it
to zero ([`sdp.c:146`](../sdp.c#L146)). Because the cones are not self-dual, primal and dual step lengths differ.

## One iteration, with the cone row and the corrector

```text
EXP-BLOCK-ITERATION(z, s in the interior; shared KKT with the other blocks)
 1  gap <- <z, s>;  scale <- gap / 3;  s_n <- s / scale
 2  if s_n in int K*:  Theta_in <- Hessian-NT chain(H_z, H_s)   else:  Theta_in <- H_z
 3  predictor (sigma = 0):  r_t <- (-s) / scale;  solve the KKT system -> dz_a, ds_a
 4  eta <- -1/2 D^3 f(z)[u, dz_a]  with  H u = ds_a                 # higher-order term (below)
 5  sigma <- from the predicted complementarity, as in SDP step 9-10
 6  corrector:  r_t <- (-s - sigma mu grad f(z) - eta) / scale;  same matrix, new right side
 7  steps: ap <- expcone_maxstep, ad <- expcone_dual_maxstep, then the fraction to the boundary
 8  accept only if the merit  |rp| + |rd| + |mu_trial - sigma mu|  falls (4-iterate window)
```

**Higher-order term.** [`expcone_pexp_correction`](../expcone_corrector.c#L55) contracts the third derivative of the barrier
with the affine directions analytically: it solves $Hu=\Delta s_a$ by a $3\times3$ Cholesky, then evaluates
$\eta=-\tfrac12D^3f(z)[u,\Delta z_a]$ from the closed forms of $\psi$ and its derivatives; no finite differences are used. For the
orthant analogue $f=-\ln z$ this gives $\eta=\Delta s_a\Delta z_a/z$, the familiar Mehrotra cross product.
Test `T299` checks it against an independent Taylor-jet computation of $f$ at 30 points and checks the homogeneity
identity $\eta(z,v)=Hv$ for $\Delta s=Hz$. The scaling data are cached between predictor and corrector
(`expcone_scale_cached`) and invalidated each outer iteration.

The dispatcher tries `sdp_ipm_csr_corrected` for PEXP models with SOC or
RQUAD blocks and for pure PEXP models with at least ten blocks, provided
there are no PSD blocks. A pure PEXP model can retry with the original
engine if the corrected engine fails to certify its point.
[Dispatch implementation](../primal_sdptask.c#L1060)

## Reduced Newton method for fixed-scale epigraphs

Suppose each exponential block has a fixed scale $u_i>0$, an argument
$v_i=b_i+a_i^\top x$, and an epigraph variable $t_i$ with cost $w_i>0$
in minimization form. With no other constraints on $x$ or on the
epigraph variables, the minimum for any fixed $x$ has
$t_i=u_i\exp(v_i/u_i)$. The objective, its gradient and its Hessian are

$$
\begin{aligned}
F(x)&=\sum_i w_i u_i \exp((b_i+a_i^\top x)/u_i),\\
\lambda_i&=w_i\exp((b_i+a_i^\top x)/u_i),\\
\nabla F(x)&=\sum_i\lambda_i a_i,\qquad
\nabla^2 F(x)=\sum_i\frac{\lambda_i}{u_i}a_i a_i^\top.
\end{aligned}
$$

PRIMAL starts at $x=0$, solves $H\Delta x=-\nabla F(x)$ by dense Cholesky
and uses backtracking to accept an objective-decreasing step. For $m$
epigraphs and $d$ free coordinates, assembling and factoring the Newton
matrix costs $O(md^2+d^3)$ per iteration, and solving with the factor
costs $O(d^2)$.

To reconstruct the dual, write the original argument equality as
$r_i^\top x+\ell_i v_i=\beta_i$. In the public sign convention for
minimization, its multiplier is $y_i=-\lambda_i/\ell_i$. The cone dual is

$$
s_i=(w_i,\ \lambda_i(\ln(\lambda_i/w_i)-1),\ -\lambda_i).
$$

This vector lies on the dual boundary and is complementary to
$(t_i,u_i,v_i)$. The dual objective contribution is
$-\beta_i y_i-u_i(s_i)_1$, where $(s_i)_1$ denotes the middle coordinate.
Stationarity in $x$ is $\sum_i y_i r_i=0$. When the epigraph objective
approaches zero, $y_i=0$ and $s_i=(w_i,0,0)$ give the dual bound consisting
of the fixed objective terms.

For example, unit scales and weights with arguments $v_1=x$ and $v_2=-x$
give $F(x)=e^x+e^{-x}$. At $x=0$, $F'(0)=0$ and $F''(0)=2$. Both
epigraph variables are 1 and both cone duals are $(1,-1,-1)$. Each block
contributes 1 to the dual objective, giving a gap of zero.

PRIMAL attempts this route for up to 64 free coordinates and 24 iterations.
Above 16 coordinates it sizes the coefficient buffer to the problem and
skips zero coefficients when assembling the lower triangle of the Hessian.
Near a minimizer, the objective change is computed term by term using
`expm1` when subtracting the two objective values would lose the
decrement to rounding. Given a step $\Delta x$, the change in the $i$th
term is $w_i t_i\,\mathrm{expm1}(a_i^\top\Delta x/u_i)$.
It checks the original row residuals, dual cone membership and objective gap
using the task tolerances. The stationarity check also scales to the
contributions in each free column, so changing the units of a variable
cannot make a visibly nonstationary point pass. An unsupported model,
failed factorization or unsuccessful check passes control to the conic
interior-point method. [Reduced Newton implementation](../primal_sdptask.c#L255)

### Adding a rotated quadratic penalty

A single RQUAD block $(w,u_Q,x)$ with a fixed head $u_Q>0$ and positive
epigraph cost $c_Q$ adds the constraint $2wu_Q\geq\|x\|_2^2$. Eliminating
$w$ adds $\gamma\|x\|_2^2/2$ to the reduced objective, with
$\gamma=c_Q/u_Q$. The gradient gains $\gamma x$ and the Hessian gains
$\gamma I$. The same Newton solve applies when the rotated tail contains
every free coordinate exactly once.

Let $g=\sum_i r_i y_i$ be the gradient contribution from the exponential
terms. The rotated cone is self-dual. Its dual certificate has free
head $c_Q$, tail $g$, and remaining head $\|g\|_2^2/(2c_Q)$. The fixed
primal head contributes $-\|g\|_2^2/(2\gamma)$ to the dual objective.
The quadratic part of the gap, including the pairing with $g$, is
$\|g+\gamma x\|_2^2/(2\gamma)$. PRIMAL verifies both rotated points
along with the exponential blocks and the original equalities.

For example, $F(x)=e^{1+x}+x^2/2$ has $F'(-1)=1-1=0$ and a positive
second derivative. Its minimum is $3/2$. With $u_Q=2$ and $c_Q=2$,
the rotated primal point is $(1/4,2,-1)$ and the dual is $(2,1/4,1)$.
They are both on the cone boundary and their inner product is zero.
Test `T314` checks both head positions and the general conic route when
a bound on $x$ changes the optimum. [Regression test](../test_primal.c#L22002)

### Reducing grouped exponential epigraphs

Consider a linear objective in free coordinates $x$ and inequalities that
cap positive weighted sums of exponential epigraph variables. A group $g$
has disjoint cone heads $t_i$ with fixed positive scales $u_i$ and affine
arguments $v_i(x)$. If $B_g>0$ denotes the cap, a feasible choice of heads
exists exactly when

$$
F_g(x)=\log\left(\sum_{i\in g}\exp(z_i(x))\right)\le 0,
\qquad z_i(x)=v_i(x)/u_i+\log(w_i u_i/B_g).
$$

Every feasible head contributes at least $w_i u_i\exp(v_i/u_i)$ to the
sum. Choosing $t_i=u_i\exp(v_i/u_i)$ attains that lower bound. A free
alias tied to the head by an equality can be eliminated along with it.
Thus these bounds become smooth convex inequalities in $x$, while the
objective stays linear.

Write $a_i=\nabla z_i$ and $p_i=\exp(z_i)/\sum_{k\in g}\exp(z_k)$ within a
group. The gradient of $F_g$ is the weighted mean $m_g=\sum_{i\in g}p_i a_i$.
Its Hessian is the covariance $\sum_{i\in g}p_i(a_i-m_g)(a_i-m_g)^\top$,
which is positive semidefinite. PRIMAL subtracts the largest argument
before computing the exponential weights.

Let $J$ have rows $m_g^\top$, and let $s_g$ and $\lambda_g$ be the positive
slack and multiplier of the reduced inequality. The Newton equations for
stationarity, feasibility and complementarity use
$H=\sum_g\lambda_g\nabla^2F_g$. Eliminating the slack and multiplier
directions gives the matrix

$$
H+\sum_g (\lambda_g/s_g)m_g m_g^\top .
$$

PRIMAL factors this matrix once per iteration, computes an affine predictor
and a centered corrector, and backtracks on positive slacks, multipliers
and the residual merit. The matrix order is the number of free coordinates.
For example, `beck751` retains eight free coordinates and five inequalities
from 80 scalar columns and 59 rows. [Measurements](../BENCHMARKS.md#verifica-mirata-del-2026-11-24)

For the lift, set $S_g=\sum_{i\in g}w_i t_i$. The original upper-row
multiplier is $y_g=\lambda_g/S_g$. For a term $i$ put $r=v_i/u_i$ and
$d_0=w_i y_g$. Its dual cone triple in PRIMAL's order is
$(d_0,\ d_0 e^r(r-1),\ -d_0 e^r)$, a supporting plane at the recovered
primal triple. The equality coefficients convert this triple into the
original scale, argument and alias multipliers. The implementation checks the
original rows, free-variable stationarity, primal and dual cone membership
and the relative gap. The native conic route handles a failed check or
factorization. The reduced path covers minimization models with up to 16
free coordinates, 16 positive upper sums and 48 disjoint PEXP blocks.
The scale of each block is fixed by a singleton equality.

For example, minimize $x$ subject to $2e^{-x}+e^x/2\le3$. Setting $y=e^x$
gives $y^2-6y+4\le0$, so the minimum is $x=\log(3-\sqrt5)$. The
multiplier of the original upper row is $1/\sqrt5$ because the derivative
of the sum there is $-\sqrt5$. Test `T322` checks this value and the
lifted multipliers for both a direct head and a free alias.
[Implementation](../primal_gp.c)

### Reducing a log-sum-exp of hinged losses

A risk model can have many scenarios with a small common decision
vector. For scenario $i$, a private loss $z_i$ satisfies
$z_i\ge 0$ and $z_i\ge b_i-a_i^Tx-\eta$. A fixed-scale exponential
cone gives $v_i\ge\exp(z_i-s)$, and one row imposes
$\sum_i w_i v_i\le B$ with $w_i>0$ and $B>0$.
For fixed $x,\eta,z$, the smallest feasible $s$ is

$$
s=\log\left(\sum_i w_i \exp(z_i)/B\right).
$$

When the objective is $\alpha\eta+\beta s$ with $\alpha,\beta>0$,
eliminating $s$ gives a convex problem in the common variables
$(x,\eta)$ and the private losses. Each optimal loss is
$[b_i-a_i^Tx-\eta]_+$, but retaining $z_i$ allows Newton steps across
the kink. The logarithmic barrier keeps both $z_i$ and
$\ell_i=z_i+a_i^Tx+\eta-b_i$ positive.

Let $p_i=w_i e^{z_i}/\sum_j w_j e^{z_j}$. The Hessian with respect
to the private losses is $D-uu^T$, where

$$
D_{ii}=\beta p_i+\mu/z_i^2+\mu/\ell_i^2,
\qquad u_i=\sqrt{\beta}\,p_i.
$$

One application of the inverse needs diagonal division and a rank-one
correction. If $F$ is the common-variable block and $C$ the cross
block, the Schur system has matrix $F-C(D-uu^T)^{-1}C^T$.
PRIMAL accumulates this matrix from individual scenario slopes.
With $m$ losses and $d$ common coordinates, one Newton step costs
$O(md^2+d^3)$ and the factorization has dimension $d$.
The step size preserves the original inequality slacks.

On reconstruction, the original exponential cones are tight.
A supporting plane at each cone gives the dual price of its
exponential epigraph. A scenario's loss price lies between 0 and
$\beta p_i$. PRIMAL uses the two barrier slacks to interpolate
between those endpoints and adjusts the prices within their bounds
so that their sum is $\alpha$. It then checks the original rows,
scalar bounds, cone membership, stationarity and primal-dual gap.

In [`primal_exp_hinge.c`](../primal_exp_hinge.c) this path recognizes
private loss columns, unit exponential scales, one positive weighted
sum and a small set of common linear inequalities. T335 checks a
solution at the hinge, a strictly positive loss and rejection when a
loss column occurs in an extra constraint.

## Routes in PRIMAL

```text
PRIMAL_optimize
  -> optimize_sdp_ipm               unless GMB_NO_EXP_IPM
       try reduced Newton for eligible PEXP epigraphs, with an optional RQUAD penalty
       try the reduced log-sum-exp inequalities for eligible grouped PEXP epigraphs
       try the reduced Newton system for a weighted sum of hinged losses
       otherwise pass exp/power cones as native blocks to the interior-point method
       first attempt: Hessian-NT row
       if not solved and nb = 0:    retry in SECANT mode (Dahl-Andersen primal-dual secant scaling),
                                    iterate restored if that fails too                   (unless GMB_NO_EXP_RETRY)
  -> optimize_conic                 tangent-cut outer approximation, the fallback
```

The tangent-cut route replaces the cone by supporting hyperplanes ([tangent cuts](tangent_cuts.md)); a near-optimal
point saved by the native route is published only if the cuts do not answer.

In secant mode on a model whose only nonscalar cones are primal exponential
blocks, the
corrector can target zero complementarity once both relative residuals are
below one tenth of their tolerances and the gap is within 5% above its
tolerance. The cone line search and the final accuracy checks still apply.
[`sdp_ipm_run`](../sdp.c)

## Worked example, solved

$$
\min t\quad\text{s.t.}\quad u+v=2,\quad(t,u,v)\in\mathcal K_{\exp}.
$$

On the constraint $v=2-u$, so $t\ge g(u)=u\,e^{(2-u)/u}=u\,e^{2/u-1}$. Then $g'(u)=e^{2/u-1}(1-2/u)$ vanishes at $u=2$, where $g(2)=2$: the optimum is
$(t,u,v)=(2,2,0)$ with value 2, on the boundary ($\psi=2\ln1-0=0$).

*Optimality by substitution.* The dual is $\max2y$ s.t. $s=c-A^\top y=(1,-y,-y)\in\mathcal K^*$. Complementarity
$\langle z,s\rangle=2\cdot1+2\cdot(-y)=0$ forces $y=1$ and $s=(1,-1,-1)$. This is in the closed dual cone: $s_0>0$, $s_2<0$,
and $s_1-s_2+s_2\ln(-s_2/s_0)=-1+1-\ln1=0$, on its boundary. Dual value $2y=2=$ primal value.

*PRIMAL.* `PRIMAL_optimize` takes the native route and returns $x=(2.00000000,\ 2.00000000,\ 0.00000000)$, objective $2.0000000000$ for
primal and dual and $y=-1$ (the public sign is the opposite of the $y=1$ above, see [SDP](semidefinite_programming.md)). The
debug trace shows 44 iterations (it=0..43) where a problem of three variables would take a handful on the interior:
the optimum is on the boundary, so $\mathrm{cond}(\nabla^2f(z))$ grows to $1.4\cdot10^{10}$ (it=3), the dual step lengths in the first iterations are
0.38–0.92 and $\mu$ is not monotone (it=20..23 rises from $3.8\cdot10^{-7}$ to $1.5\cdot10^{-6}$). The final measures are
rel_pri $3.0\cdot10^{-13}$, rel_dual 0, rel_gap $2.1\cdot10^{-15}$, and the cone check prints the solution as
$(2,\ 1.99999999997,\ 3.3\cdot10^{-11})$.

## Costs and failure modes

- **Per iteration**: one $3\times3$ evaluation (barrier, Hessian, chain, $3\times3$ Cholesky) per exponential block,
  plus the KKT factorization, which dominates ([SDP](semidefinite_programming.md) for the dense and sparse forms).
- **Boundary optima** make $H_z$ and $H_s$ ill-conditioned (condition numbers of $10^{6}$–$10^{11}$ appear in the trace) and
  slow the method, as in the example. That is the intrinsic cost of a barrier method at a boundary optimum.
- **Not solved is not infeasible.** A failed native run triggers the secant retry, then the cuts; no infeasibility is claimed
  without a measured certificate ([certificates](certificates.md)).
- **Dual vectors** must be in the dual cone, not in $\mathcal K$: a dual step tested against the primal cone would be clamped to 0.

## Exercise

For $z=(3,1,\tfrac12)$ verify that $\langle g(z),z\rangle=-3$ (by homogeneity of $f$, or from the formula for $g$), and then check that
$s=-g(z)$ satisfies the interior condition of the dual cone.

*Answer.* By homogeneity, $f(\lambda z)=f(z)-3\ln\lambda$, so
$\tfrac{d}{d\lambda}f(\lambda z)|_{\lambda=1}=\langle g,z\rangle=-3$; the formula for $g$ gives the same number (the
program behind the numbers above evaluated $-3.0000000000$). For the dual condition,
$s=(0.8901768,\,1.1647348,\,-1.6705304)$ has $s_0>0$, $s_2<0$ and
$s_1-s_2+s_2\ln(-s_2/s_0)=1.7837058>0$ (this is $-\gamma(0)$), so $s\in\mathrm{int}\mathcal K^*$.

## References

- Yu. Nesterov, [Towards non-symmetric conic optimization](https://doi.org/10.1080/10556788.2011.567270), *Optimization Methods and Software* 27(4–5):893–917, 2012.
- R. Chares, *Cones and interior-point algorithms for structured convex optimization involving powers and exponentials*, Ph.D. thesis, Université catholique de Louvain, 2009.
- A. Skajaa and Y. Ye, [A homogeneous interior-point algorithm for nonsymmetric convex conic optimization](https://doi.org/10.1007/s10107-014-0773-1), *Mathematical Programming* 150:391–422, 2015.
- J. Dahl and E. D. Andersen, [A primal-dual interior-point algorithm for nonsymmetric exponential-cone optimization](https://doi.org/10.1007/s10107-021-01631-4), *Mathematical Programming* 194:341–370, 2022.
