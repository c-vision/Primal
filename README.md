# PrimalSolver

**A Dependency-Free Mathematical Optimization Engine**

PrimalSolver is a dependency-free mathematical optimization solver written from scratch in C99, supporting LP, MILP, QP, QCQP, SOCP, SDP and mixed-integer conic optimization.

- **48 C ports of MOSEK examples (47 distinct) reproduced with matching optimal values**
- **4242 reliability checks**
- **Full primal/dual solution certificates**
- **Hybrid engine**: presolve + sparse normal-equations IPM for large LPs (90000-var transport in ~0.6 s) and large sparse QPs (n=20000 in ~0.06 s)
- **Zero external dependencies** (libc + libm only)

## What it is

PrimalSolver is an independent mathematical optimization engine, implemented from scratch in C99, with extensive compatibility validation against the public MOSEK 11.0 examples. It is not a wrapper, and it is not a teaching implementation of the simplex method: it is a general-purpose solver built around a single internal conic representation, with optimality certificates exposed through a public API.

## Independently verifiable certificates

Most solvers return a solution and ask you to trust it. PrimalSolver returns a **certificate**:

```
solver
   │
   ▼
primal solution  +  dual solution (y, slc/suc, slx/sux)
   │
   ├── stationarity         c + Qx + A'y + z = 0
   ├── primal feasibility
   ├── dual feasibility
   └── complementarity
   │
   ▼
independently verifiable certificate (via public getters)
```

Every solution comes with duals that satisfy stationarity, primal/dual feasibility and complementarity, and strong duality is reported as `|pobj − dobj|`. All of it is retrievable through public getters, so the caller can verify the result deterministically instead of trusting the solver.

Infeasibility and unboundedness are certificates as well. `PRIMAL_getdualray`
returns `y` (`numcon` entries) with `Σ_i y_i·b_i^act > 0` and `A'y ≤ 0` on every
nonnegative variable — a linear combination of the constraints that no
nonnegative point can satisfy, so the model has no feasible solution — and
`PRIMAL_getprimalray` returns `ρ` (`numvar` entries), a recession direction of the
feasible set along which the objective improves without bound (`c'ρ < 0` for a
minimization, `> 0` for a maximization). Both are scaled to `max |·| = 1`. A
vector is handed out only after it has been **measured** in the standard form the
solver actually solved, and the `PRIM_INFEAS_CER` / `DUAL_INFEAS_CER` statuses are
reported only together with the matching ray: a status names a certificate, so it
does not outlive one. Where there is no vector to give, the getters answer
`PRIMAL_RES_ERR_ARG` — the same code a solution getter uses for a problem that has
no solution.

That rule now holds on every route, and it did not before `T98`. Three places
declared a certificate they never produced — the tangent-cut route in both
directions, and a mixed-integer search whose relaxation is unbounded — and a
fourth declared a primal-infeasibility certificate from a *basis* the caller had
handed in, which is a statement about the basis. None of them can carry a ray:
the cut route solves in **cut space**, whose rows are tangents and not the
model's constraints, so a witness of it is not one of it (lifting it is not
sound), and branch-and-bound never lifts the relaxation's direction into the
integer model. All four now publish `UNKNOWN` and carry the model verdict in
`prosta`, which answers the question about the model. The basis case was also a
crash: it raised `has_sol` without allocating a solution — `opt_prepare` is the
only place `t->x` is allocated — so the next `PRIMAL_getxx` copied from a null
pointer. `T98` measures both halves, including a positive control: a genuinely
infeasible LP still answers `PRIM_INFEAS_CER` *with* a ray that measures.

That fix exposed a second conflation, closed by `T99`: `has_sol` was answering
two questions at once — *is there a point to hand out*, and *was a verdict
reached*. Because `prosta` and `solsta` were read through it, every route that
concluded without a point had to raise it, and the solution getters then
answered `OK` on the all-zero buffer `opt_prepare` left behind. On an infeasible
LP that is self-contradictory output: `PRIMAL_getxx` returns `x = 0` while
`PRIMAL_getprimalinfeas`, reading that same buffer, returns by how much `x = 0`
violates the model. `has_sol` now means only the first question, and a verdict
travels in `rc` plus `prosta` (`solsta` too where a certificate has its ray
beside it), so the four routes that had to raise `has_sol` to be readable — the
LP/QP non-optimal branch, the cut route in both directions, and a MIP stopped by
the node cap with no incumbent — no longer do, and every solution getter
refuses. Certificates are untouched: the ray getters gate on their own
`has_dray`/`has_pray`, so an infeasible LP still reports `PRIM_INFEAS_CER` and
still hands out the ray that measures, with `PRIMAL_getxx` refusing beside it.
`T99` asserts exactly that pairing on eight cases, two of them positive
controls: an optimal LP and a node-capped MIP that *does* have an incumbent.

A third conflation was between a model and its own **representation**. The
tangent-cut route caps every bar entry at ±`SDP_BIGM` (1e6) so that its very
first LP — which has no cuts yet — has a finite answer. A model unbounded
*through* a bar stopped on that cap and the answer was published as a statement
about the model: `rc = OK`, `solsta = OPTIMAL`, `pobj = dobj = -2e6`, with a dual
bar `Z = C − Σ y_i A^i` outside the dual cone because the cap's multipliers never
enter that sum — the hole `T89` had warned about. Both feasibility getters were
already right (`getprimalinfeas = 0`: the rows *are* satisfied;
`getdualinfeas = 1`: `λmin(Z) = −1`); the verdict was the part that did not
listen. What decides now is a **recession direction measured on the model as
written** — its rows, its bounds, and the cone relaxed to a nonnegative
diagonal, the widest relaxation an LP can write for a PSD matrix. The LP only
supplies the candidate, so widening that relaxation can lose a ray and cannot
invent one. A direction that measures means the model has no finite value:
`rc = ERR_UNBOUNDED`, `prosta = DUAL_INFEAS`, and no point published — not even
`getbarxj`/`getbarsj` — because the direction lives partly inside a bar while
`PRIMAL_getprimalray` has `numvar` scalars as its object, the deviation `T85`
declared for the conic routes. A direction that does not measure means no claim
at all: `TRM_MAX_ITER`, `prosta = UNKNOWN`. That verdict lives in **one**
function, `bar_cap_verdict`, which both capping routes call after their own loop:
the tangent-cut route (`± entry ≤ SDP_BIGM` as `R_+` rows in its LP) and the
conic build (the same rows, same constant, in `optimize_conic`). A bar-plus-cone
task never reaches the cut route — `numcones > 0` skips it — so on that shape the
conic build was the only publisher, and before this it answered `-2e6` for one bar
and `-4e6` for two, with `getdualinfeas` already saying `1`. The ray test is now
**cone-aware**: a candidate direction must lie in every cone of the task, so a
model that escapes through a cone as well as through a bar is still recognized,
and the old `numcones == 0` refusal to conclude on that path is gone. `T100`
asserts the verdict on eleven models: five run on **both** routes (A..E, because
what is claimed is the verdict and not the path), a scalar unboundedness that
still hands out its ray beside the certificate (F), and five on the conic build —
an unbounded bar, which also proves from the public log line that it was the conic
build asking about the cap (G), the same model starved to one iteration (L), a
bounded bar at 2 (H), a bar at `r = 1000` answering 2000 (I), and two
bars with **distinct** off-diagonals whose published blocks come out `[[2,2],[2,2]]`
and `[[3,3],[3,3]]` and swap when the two rows are swapped — the placement control
for the compressed-triangle column blocks, where a shared base would make the
model infeasible rather than merely inaccurate. Two consequences are declared
rather than hidden: a model whose infimum is merely *not attained* inside a bar
(`max −x0` under `[[x0,1],[1,x1]] ⪰ 0`, which needs `x1 → ∞`) parks on the cap
with no recession direction and is answered "no verdict" instead of with a value;
and the placement control above has no broken-build behind it — re-introducing the
aliased offsets was refused by this session's permission level, so that one
assertion is a measured invariant, not a measured negative control. On the same
path the PSD tangent rows are now a row kind of their own (`CR_PSDCUT`): the dual
assembly used to walk them as nonlinear cuts and index `cutcol` slots that were
never written for them, reading uninitialised memory into the reduced-cost vector.
No published figure moved — the whole 68-sample corpus, every `[route]`/`[cones]`
trace line included, is byte-identical before and after.

A **cone** adds a side the triple cannot see, and a stalled conic run is not an
answer either way. A cone has no rows to leave a residual on: the vertex of a QUAD
block satisfies every equation the model has, so `rel_pri`, `rel_dual` and
`rel_gap` — the three figures every other guard of this family reads — can all
close there on a point that is nowhere near the optimum. What says so is the
**reduced cost of the block**, which has to lie in the dual cone `K*`, and it is
judged in `PRIMAL_optimize` rather than in a route because that is the only place
the model the *user* wrote is answered (a shadow task calls a route directly, and
its cones belong to the encoder). Measured before the guard: `min −x0` on one free
3-D QUAD block with no rows at all returned `rc = OK`, `solsta = OPTIMAL`,
`pobj ≈ 5e-9` at `x = 0`, with a cone-dual violation of exactly 1. A conic run that
exhausts its iterations has likewise said something about its own trajectory and
nothing about the model, and `TRM_MAX_ITER` is the answer that leaves the user with;
two questions are settled there too, each on the relaxation that makes **its own**
answer sound — infeasibility on a set **containing** the cones (only linear
consequences of membership: `t ≥ 0`, `t ≥ ±u_i` for a QUAD, `t ≥ 0, u ≥ 0` for the
rest), because an LP infeasible on a subset of the model proves nothing;
unboundedness on a set **inside** them (a polyhedral inner approximation), because a
direction of a wider set is not a direction of the model. The two relaxations are
opposite and neither substitutes for the other. Neither verdict carries a vector:
the conic routes publish no Farkas ray (`T85`), so `prosta` says `PRIM_INFEAS`/
`DUAL_INFEAS`, `solsta` says `UNKNOWN`, and nothing about the point is published —
the family `T98` established. Reading the block back also needed the sense algebra
written down: the route's min form has cost `s·c` and slack `s·c − A'π`, and `y` is
published as `−s·π`, so the user-space dual is `d = s·(c + A'y)`. The reading
`c + s·A'y` that was there first is identical on a MINIMIZE model and differs by
exactly `2c` on a MAXIMIZE one — a sign flip of the cone, not a rounding — and a
MAX model whose cone members carry no cost (where this was first pinned) cannot
discriminate them. `T101` G is the model that can: `max −(x0+x1+x2)` with
`x0 = 5`, `x1 = 4` on a QUAD block, whose correct dual is the boundary point
`(5/3, −4/3, 1)` and whose wrong reading is `(−1/3, −10/3, −1)`. Reverting just
that factor does not move a digit there — it **flips the whole verdict** to
`rc = 1007` with every getter refusing, and `T101` is green on `dobj = pobj = −6`.
Fixing the ray test on that shape exposed a third reading error, of the `NaN >
viol` family of `T90`: the `u = 0` face of a `PEXP` was taken as "no violation",
where the cone in fact closes onto `{u = 0, v = 0, t ≥ 0}` — so a candidate
direction escaping along `v` passed the membership test and a bounded model was
answered unbounded. `T101` asserts nine models (A..I) with the dual block, its
boundary identity `t = |(u,v)|` and its complementarity `d·x = 0` recomputed in the
test from the published `y`, and keeps two declared deviations visible: a cone
member with a **finite bound** leaves its block unmeasured rather than measured
wrong (H), and the bar route publishes `slx/sux = 0` on the cone's own variables
where the true block dual is `(5/3, −4/3, 1)` — `getdualinfeas` therefore reads that
dual from `c` and `y` directly, not from the published slacks, so the zeros do not
become a violation (I).

Two limits are declared rather than hidden. A dual ray whose support needs the
cap row of a ranged variable (`x_j ≤ ux_j`), or **both** sides of a ranged
constraint, has no one-entry-per-constraint image in the user's model, so it is
not published even though the reduced-space argument behind it is sound. And
explicit witnesses are written only by the tableau simplex — an interior-point
route has none, so its last iterate is offered to the same measurement and the
measurement decides. The **conic routes now publish both rays** of a **bar-free**
model. The **primal ray** (a recession direction) comes from
`model_lp_witness`'s inner relaxation and is checked by `sdp_ray_measures` —
`ρ ∈ K` for every cone, `Aρ = 0` on every row, the bound signs, `c'ρ < 0`,
normalized to `max|·| = 1` — with `solsta = DUAL_INFEAS_CER` and `getprimalray`
answering (**T101** A/B). The **dual ray** of an infeasibility comes from the
outer relaxation's Farkas witness and is checked against the **model's own
cones** — `d = A'y ∈ K*` on every block, the row multipliers signed, the RHS
combination negative — with `solsta = PRIM_INFEAS_CER` and `getdualray`
answering (**T101** C/J, the §3.7 shape). A candidate that does not measure
proves nothing and the verdict stays in `prosta`. With bars the compressed block
still has no image as `numvar` scalars, so no ray is published there. The same
pass also refuses a conic point that lies **outside** the cones: the route's
three figures do not see membership, and `min x0` with `x0 = -1` over `QUAD`
answered `OPTIMAL` at `x = (-1,0,0)`, outside by 0.5, before `conic_primal_cone_worst`
put the fourth figure into the verdict. MIP rays remain open, declared.

Feasibility of the *cones* is a fourth, separate number. `rel_pri` counts the
equality rows only, so none of the three published figures says by how much the
point is inside `K`; membership used to be guaranteed by the backtrack test and
never measured. `GMB_DBG=1` now prints one line per solve that published into a
conic, SDP or `putqconk` model:

```
  [cones] task rel_slack=-2.88e-09 pri_slack=-5.77e-09 where=cone 0 PPOW a=0.666666666667{...}
```

`rel_slack` is the worst **signed** slack over bounds, cones, quadratic rows and
PSD blocks, divided by `1 + (size of that object)` — `max |·|` of the members for
a cone, the spectral norm for a bar, the bound itself for a bound — because the
tolerance the task declares is a relative one. Reading a block in its own
homogeneous units matters: the `t`-space form of `PPOW(a)` shrinks a geometric
violation by `1/(a·t^(a-1))`, which is 6.4 on `regression_regularized` and 3333 at
`a=1/3, t=1e-6`, and outside `{t >= 0, u >= 0}` `pow()` returns NaN, which
compares false against any violation and so reads as "no violation". Measured on
the `-O2` build: 100 published solutions in the suite (`94` on `HEAD`) and 31
across the samples, worst `4.5e-09`. Five of the hundred sit outside the `1e-8`
the task declares — `{4.8e-05, 2.7e-04, 1.7e-03, 2.6e-02, 0.67}`, a quadratic row,
a 2×2 bar, two bounds and one `QUAD` cone — and all five print identically on
`HEAD`, so the six publications the new guard added are all inside. The
tangent-cut route now **accepts on
that same quantity** (`T90`): it used to stop as soon as the `t`-space residual
fitted `tol·(1+|x0|)`, which published `t = -4.7e-11` — a point where the cone has
no value at all — with `PRIMAL_RES_OK`.

## A common conic language

Rather than shipping independent solvers per problem family, PrimalSolver maps every supported class onto one shared internal representation:

```
LP  →  QP  →  SOCP  →  exponential/power cones  →  SDP  →  mixed-integer conic
```

QCQP constraints are encoded exactly through an eigendecomposition into rotated quadratic cones; semidefinite programs are solved by a primal-dual interior point on the PSD blocks (with a λ_min tangent-cut fallback); SOC and SDP are handled in the same conic loop; mixed-integer conic problems are solved by branch & bound over conic relaxations. This keeps the solver surface small and the behaviour consistent across families.

## Supported problem classes

| Class | Algorithm |
|---|---|
| **LP** (linear) | Presolve (default on) → two-phase primal simplex (Dantzig→Bland anti-cycling); Mehrotra predictor-corrector interior point — dense augmented system (LU) for small/medium and QP, **sparse normal equations (Cholesky) for large LPs** |
| **MILP** (mixed-integer) | DFS branch & bound over LP relaxations, SOS1/SOS2 branching, semi-continuous/semi-integer variables. **Search and publication use two different tolerances**: `MIP_TOL_INTHER` (1e-5) says when an integer constraint may be treated as satisfied, `MIP_TOL_FEAS` (1e-6) is what a candidate incumbent must actually measure in — bounds, rows, cones, semi domains, SOS — with a fix-and-optimize re-solve of the node's relaxation behind a candidate that misses it. Branching splits on the raw `floor(x*)` and never creates a child that does not tighten the node's box: a child that repeats its parent is an infinite tree (measured: the reference's 1e-5 threshold, applied to the split itself, regenerates nodes to the 100000-node cap). `PRIMAL_getprimalinfeas` measures a semi variable's domain `{0} ∪ [l,u]`, so a deactivated variable does not read as bound-infeasible (T95) |
| **QP** (convex quadratic) | Mehrotra interior point — dense augmented system (LU); **sparse normal equations (Cholesky on M=Q+Z/X, K=A·M⁻¹·Aᵀ+δI) for large QPs with sparse Q** |
| **MIQP** | Branch & bound over QP-IPM relaxations |
| **SOCP** (quadratic cones) | Primal-dual conic IPM: R₊, quadratic (QUAD), rotated (RQUAD) cones — dense augmented LU, or for large problems a **hybrid sparse direction: Nesterov-Todd (NT) symmetric normal equations + sparse Cholesky, with an augmented sparse-LU fallback near the cone boundary** |
| **Exponential/power cones** (PEXP/DEXP/PPOW/RPOW) | **Native non-symmetric primal-dual IPM** (default route): closed-form barrier/gradient/Hessian, **dual cone `K*`** and scaling point `W` (`-∇f(W) = s`) in `expcone.c`, exp blocks assembled into the unified augmented KKT of `sdp.c` with **Jacobi row/column equilibration** (the multiplier block grows like `1/μ`, ~1e11 of dynamic range by μ≈1e-8) and a **μ- and σ-free** Hessian-NT row (`Thin` depends on the iterate only; `σ` enters the residual). Accuracy on the T79 oracles: `1.6e-13`/`6.9e-14`/`1.9e-13`/`9.6e-15` against `1.3e-12`/`1.9e-8`/`5.1e-8`/`1.5e-9` from the cuts, and the degenerate CBF `T47` case is answered natively (7.1e-6 radial KKT vs the cuts' 5.4e-5). The **outer approximation via tangent cuts** remains an explicit fallback, selected by **measured quality in MOSEK's form** — relative primal infeasibility `pfeas/(1+‖b‖∞)`, relative dual infeasibility `dfeas/(1+‖c‖∞)` and relative duality gap `\|pobj-dobj\|/(1+\|pobj\|+\|dobj\|)` compared against the task's own `PRIMAL_DPAR_INTPNT_TOL_{PFEAS,DFEAS,REL_GAP}` (defaults 1e-8) — not against an internal μ threshold: μ is a path parameter, `⟨x,s⟩ = μ·ν` measures complementarity, neither is an accuracy certificate. `sdp_ipm` therefore reports "not solved" when the polished point misses a declared criterion and the dispatcher answers with the cuts — deliberately, not by accident (`logistic_large` at rel_gap=1.0e-7; `gp1` and `market_impact` pass and answer natively at rel_pri=3.6e-9 and 2.4e-9). `GMB_NO_EXP_IPM` forces the cuts. Optimal values locked by **T79** (`t=e`, `2√e`, `2^(1/0.3)`, `2^(-1.25)`), dual/scaling identities and μ/σ-independence by **T80**/**T78**, native-vs-cut parity by **T81**, and the gate-obeys-the-declared-tolerances contract by **T82** |
| **SDP** (semidefinite) | **Primal-dual interior point (Nesterov-Todd) on the PSD blocks** — normal equations on the equality rows, `W_j S_j W_j = X_j` scaling, floored centering + backtracking (handles rank-deficient optima); falls back to λ_min tangent cuts when the IPM conversion does not apply — a fallback whose published dual is **measured**, not inferred (`Z >= 0`, `<Z,X> = 0`, `dobj = pobj`; T89) |
| **QCQP** (quadratic constraints) | Exact encoding: eigendecomposition Q = Σλᵣvᵣvᵣ' + RQUAD cones |
| **MICO** (MIP + cones) | Branch & bound with conic shadow-task relaxations |
| **SOC + SDP combined** | One system, not two loops: SOC/RQUAD arrow-KKT rows and NT Schur rows for the PSD blocks are assembled in the **unified conic IPM** in `sdp.c` (T75/T76); λ_min cuts remain the fallback when the conversion does not apply |
| **ACC/DJC** | Affine conic constraints `appendacc(domidx, numafeidx, afeidxlist, b)` and disjunctive constraints `appenddjcs`/`putdjc` (OR of clauses, each a conjunction of domains on affine expressions; reference style, via big-M MIP; `T140`) |

## Key features

- **Optimality certificates**: primal + duals satisfying stationarity, feasibility and complementarity, fully checkable through public getters
- **Infeasibility certificates**: Farkas rays as vectors (`getdualray`/`getprimalray`), normalized to `max |·| = 1` and published only when they measure in the form that was solved
- **Basis solve**: exact evaluation of a given basis (`solvebasis`), MPS-BAS read/write
- **Sensitivity analysis**: cost ranges for which the current solution stays optimal (`costsensitivity`/`rhssensitivity`)
- **The model reads back as it was written**: `PRIMAL_getarow`/`PRIMAL_getacol` and
  their `*slice` forms hand back the entries of a row or column as subscripts and
  values, answer how many were written, and **refuse without touching the caller's
  buffers** when the space does not fit; `getnumanzs`/`getmaxnumanzs`/`getnumqobjnz`
  count what is stored. The **scalar** read of one entry answers something else — the
  operator, i.e. the sum of the entries stored at that `(i,j)`, which is the
  coefficient every route solved — and that split is the subject of its own deviation
  note below (`T107`). Variables and constraints carry names —
  `putvarname`/`putconname`, `getvarnameidx`/`getconnameidx`, and a lookup by name
  (`getvarname`/`getidxvar`, `getconname`/`getidxcon`) — held in two independent
  tables, so a constraint and a variable may share a name the way an MPS file does
  (`T102`). Each table also reads back whole — `getallvarname`/`getallconname` write
  exactly `numvar` (`numcon`) borrowed strings, and an unnamed slot reads `""` there
  and from `getvarnameidx` alike, because two readings of one table that disagree are
  two rules (`T103`). The two quadratic forms read back whole the same way:
  `PRIMAL_getqobj`/`PRIMAL_getqconk` enumerate **the table their own counter counts**
  and therefore answer in different shapes, because the two stores are different —
  the objective is the user's **triplet list in write order**, stored zeros included
  (a written `0.0` is a write, not a hole) with a cross term stated **once**, while a
  constraint's `Q` is the **upper triangle** of the dense store, ascending, so a row
  whose entries cancel reads *empty* rather than refused (`T104`). Bar variables are
  the **third** name table — `putbarname`/`getbarnameidx`/`getbarname`/
  `getidxbarvar`/`getallbarname` — and they add no rule, only a table: the five rules
  live in the same two functions and the capacity is the same `barcap` as the blocks
  themselves. The one thing the two scalar tables could not decide is decided here: a
  bar's name lives in a namespace of **its own**, because a bar is not one of the
  `numvar` scalars — it has its own index space, its own dimension and its own cone
  block. So one string may name a variable, a constraint and a bar at the same time,
  while two bars may not share it (`T105`). Cone blocks are the **fourth** table —
  `putconename`/`getconenameidx`/`getconename`/`getidxcone`/`getallconename` — sharing
  the one capacity of `cone_type`/`cone_nmem`/`cone_mem`/`cone_param` rather than
  carrying a fifth to keep in step, and its own namespace too. Because
  `getconname` (a constraint) and `getconename` (a cone) differ by one letter, the
  independence is not asserted in prose but measured: one string naming a variable, a
  constraint, a bar and a cone at once answers four lookups with four different
  indices, and a duplicate cone name is refused with the constraint's name still alive
  (`T106` C). The **objective** is the same mechanism over a table of one —
  `putobjname`/`getobjname` — which is why renaming it *replaces* instead of refusing:
  with `n = 1` the uniqueness loop has no second slot to collide with, and that
  difference from a table is asserted rather than inferred (`T106` G). The last two
  getters that still broke that contract were the bar-term list readers:
  `PRIMAL_getbaraidxij`/`PRIMAL_getbarcidxj` stopped at the caller's `maxnum` and
  answered `PRIMAL_RES_OK` with a truncated list, which is a reading that lies — an
  `⟨A,X⟩` built from a prefix is a different model. They now count first and refuse
  without writing, with `NULL`/`NULL` as the door that asks for the number alone
  (`T108`)
- **Solution quality**: max primal/dual violation of the current solution, plus
  the interior-point relative criteria in MOSEK's form — `pfeas/(1+‖b‖∞)`,
  `dfeas/(1+‖c‖∞)`, `|pobj−dobj|/(1+|pobj|+|dobj|)` — declared through
  `PRIMAL_DPAR_INTPNT_TOL_{PFEAS,DFEAS,REL_GAP}` (1e-8 each by default), which
  is what the conic route is accepted or rejected on, and
  `PRIMAL_DPAR_INTPNT_TOL_NEAR_REL` (1000), the reference's near-optimal
  acceptance factor, which scales all three before the verdict where this IPM is
  the only conic route
- **A "not solved" answer publishes no solution**: `solsta` stays `UNKNOWN` and
  every solution getter answers `PRIMAL_RES_ERR_ARG`, rather than the zeroed
  vectors the pre-solve left behind. That covers the answers which *do* reach a
  verdict and still have no point: an infeasible model, an unbounded one, a
  node-capped integer search that never found an incumbent (`T99`), and an
  answer that stopped on the solver's own cap on the bar entries (`T100`)
- **Feasibility repair**: elastic LP — minimum-total-violation point for infeasible problems
- **Progress callback**: per-iteration/per-round notifications observable externally
- **Warm start**: user-provided starting point for the IPM
- **Reported problem and solution status**: `PRIMAL_getprosta` and
  `PRIMAL_getsolsta` publish the reference's two numbers, with the pairing its
  tables fix, so `prosta` is derived from `solsta` rather than stored beside it
- **Declared parameters**: the 27 `PRIMAL_IPAR_*`/`PRIMAL_DPAR_*` ids live in one
  table (`PRIMAL_PARAMS` in `primal.c`) holding kind, target field, default and
  inclusive range. Defaults, the range checks of both setters, the getters and
  the copy into shadow tasks all read that table — there is no second list of
  parameters to keep in sync — and `PRIMAL_getparaminfo` reports a row's default
  and bounds. Integers and doubles are separate id namespaces sharing numbers,
  so `PRIMAL_getparaminfo` takes the kind as an input. Wired: `IPAR_SCALING`,
  `IPAR_MIP_MAX_NODES`, `DPAR_MIP_TOL_{ABS_GAP,REL_GAP,INTHER,FEAS}` (the last
  four used to be compile-time constants; `FEAS` did not exist at all — it is the
  tolerance an incumbent is re-verified against, see below)
- **LP presolve**: empty/singleton-row and empty-column reductions with exact primal+dual postsolve (default on, `PRIMAL_IPAR_PRESOLVE`)
- **I/O**: free-format **MPS** (RHS/RANGES/BOUNDS/MARKER/QSECTION), **CPLEX LP**, **CBF v4** (cones + SDP + CHANGE section).
  The MPS and LP readers **populate the name tables** they read through: what a file
  labels is what the task calls it — `putvarname`/`putconname`/`putobjname` at the index
  the label names (`T112`). A file that gives one name to two rows is refused
  `PRIMAL_RES_ERR_FILE` before anything reaches the task, because every later section
  looks rows up by name and the second row would receive no entries at all. CBF keeps
  its group labels out of those tables: a group names `sz` variables at once, and a name
  table admits one index per name.
- **Bound slices and solution slices** (`T114`, `T115`):
  `PRIMAL_getvarboundslice`/`getconboundslice` and the `put*` counterparts,
  `[first, last)`, reference-shaped; a refused read writes nothing and a refused
  write validates the whole slice first. `PRIMAL_getxxslice`/`getyslice`/
  `getslcslice`/`getsucslice`/`getslxslice`/`getsuxslice`/`getskxslice`/
  `getskcslice`, `getxc`/`getxcslice` (the constraint activity, read by the same
  `row_activity` as `getpviolcon`) and `getreducedcosts`
  (`(s_l^x)_j − (s_u^x)_j = −(slx+sux)`) are a second reading of the solution
  vectors, asserted against the whole vector entry by entry. The counters are
  named as the reference names them (`getnumanz`/`getmaxnumanz`). `T117` adds the
  block-triplet form of A-bar and C-bar (`getbarablocktriplet`/`getbarcblocktriplet`
  and their counts): one row per stored lower-triangle entry of every block.
  `T118` adds the objective vector (`getc`/`getcslice`, `putclist`/`putcslice`) and
  the scalar quadratic entry `putqobjij` (`q_ij = q_ji`, lower triangle, replaces the
  pair and removes it at 0). `T119` adds the A store in triplet form (`getatrip`),
  the row/column nonzero counters and their slices, and `putaij` (the scalar entry
  that replaces and collapses the pair). `T120` adds the list forms
  (`putvarboundlist`/`putconboundlist`, `putvartypelist`/`getvartypelist`) and the
  constant-bound slices (`putvarboundsliceconst`/`putconboundsliceconst`), all of
  which validate the whole input before applying. `T121` adds the task name
  (`puttaskname`/`gettaskname`/`gettasknamelen`) and the name lengths
  (`getvarnamelen`/`getconnamelen`/`getobjnamelen`/`getmaxnamelen`; a length never
  counts the terminator, and `gettaskname` refuses when the buffer cannot hold it).
  `T122` adds `putqcon` (replace all quadratic terms of all constraints from one
  triplet list, lower triangle, duplicates accumulate, empty list clears) and the
  utilities `getversion`/`isinfinity`/`getresponseclass`. `T123` fixes a
  branch-and-bound soundness bug: a semi-continuous variable's **root box** is now
  relaxed to `[0,u]`, so the LP relaxation keeps the deactivation `x=0` and its
  optimum is again a valid lower bound (`min x` over `{0}∪[2,5]` is 0, not 2).
  The MPS writer's round-trip was fixed too: for a file-read model its row 0 is the
  `N` objective row, so the writer emits `N` for it and rows `1..numcon-1` as
  constraints, preserving `numcon` (`T112` G). A hand-built model whose row 0 is a
  real constraint keeps the old path (one extra row on re-read), and names stay
  positional -- both declared. `T124` adds the general callback
  (`putcallbackfunc`/`getcallbackfunc`: BEGIN/END of OPTIMIZER/READ/WRITE with the
  `MSKcallbackcodee` numbers) and the response callback (`putresponsefunc`, invoked
  when a solve ends with a non-OK code); the detail vectors are NULL, declared.
  `T255` completes the phase callbacks: the route that answers also emits
  BEGIN/END of SIMPLEX or INTPNT (LP/QP), MIO (branch & bound) and CONIC
  (conic/SDP), so the callback reports which algorithm ran. `T256` adds the OPF
  format (reference 16.3) -- reader and writer for the linear/quadratic/conic/
  integer core, dispatched by the `.opf` extension, with the data callbacks
  `READ_OPF`/`READ_OPF_SECTION`/`WRITE_OPF` emitted. `T257` adds the "middle"
  solution-update codes (`CONIC`, `PRIMAL_SIMPLEX`, `INTPNT`) at each cut round /
  basis / relaxation. `T258` closes the per-iteration gap: the engine loops
  (`simplex.c` 93/36, `ipm.c` 90, `socp.c`/`sdp.c` 34) emit their code at **every
  internal iteration** through a hook (`primal_cb_iter`, guarded by
  `primal_cb_iter_on` so the common case with no callback is a plain
  load+branch). `T260` adds the sub-step codes (`IM_LU` at every LU
  factorisation, `IM_ORDER` at the AMD ordering). Declared deviation: with
  `IPAR_NUM_THREADS > 1` the concurrent LP path interleaves the events. `T259`
  reads and writes OPF quadratic **constraints** (the writer emits the Q part,
  so they round-trip): a convex `q(x) <= b` uses the UP convention and
  `q(x) >= b` is rewritten as `-q(x) <= -b`; an equality or ranged quadratic is
  refused (declared). OPF
  declared deviations: the `zero` cone, `[solutions]` and `[vendor]` are not
  read; the OPF `dpow` maps to our RPOW.
  `T125` adds `getsolution` (the reference's one-call reader: sta/solsta, basis
  keys, `xc`, `xx`, `y`, `slc`/`suc`/`slx`/`sux`, `snx`) plus `getskn`/`getsnx`;
  `skn` is SK_UNDEF for cones and `snx` is 0, both declared. `T126` adds
  `putaijlist`, `chgvarbound`/`chgconbound` (change one side and recompute the key),
  the matrix store in bulk (`getsparsesymmat`/`appendsparsesymmatlist`) and
  `getprobtype` (LO/QO/QCQO/CONIC/MIXED). `T127` adds the parameter resets
  (`resetintparam`/`resetdouparam`/`resetparameters`, from the declarative table's
  defaults) and the 64-bit counter variants (`getnumanz64`/`getmaxnumanz64`/
  `getnumqobjnz64`/`getnumqconknz64`). `T128` adds the contiguous-member cone
  appends (`appendconeseq`/`appendconesseq`) and the bulk row/column setters
  (`putarowlist`/`putacollist`/`putarowslice`/`putacolslice`, CSR data validated
  before writing). `T129` adds the enum symbolic names (`prostatostr`/`solstatostr`/
  `bktostr`/`conetypetostr`/`probtypetostr`/`sktostr`/`rescodetostr`); the text is
  this solver's own, a declared deviation (the reference's exact strings were not
  read). `T130` adds `checkversion`/`getbuildinfo`/`getcodedesc`/`getlasterror` (the
  last response code and message follow the last solve) and `get`/`putatruncatetol`
  (stored, not applied: this solver does not truncate A). `T131` adds the removals
  `removecones` (compacts the cone list) and `removebarvars` (drops bar variables
  and remaps the A-bar/C-bar terms). `T132` adds file streams
  (`linkfiletotaskstream`/`linkfiletoenvstream`), function streams on the env
  (`linkfunctoenvstream`, inherited by tasks created afterwards), the unlinks and
  the echo helpers (`echotask`/`echoenv`/`echointro`). `T133` adds the bar sparsity
  and per-block index info (`getbarasparsity`/`getbaraidxinfo`/`getbaraidx` and the
  C variants; `idx` is this solver's vectorisation `i*numbarvar+j` for A, `j` for
  C, a declared deviation). `T134` adds `inputdata`/`inputdata64` (load the linear
  part in one call; the task must be empty) and `getmemusagetask` (a byte
  estimate). `removecons` (T134) removes constraints and compacts A/qcon/barA,
  remapping the remaining rows. `removevars` (T135) removes variables and
  reshapes the dense qcon blocks at the new stride, remapping qobj and the cone
  member lists. `T136` adds `strtoconetype`/`strtosk` (the inverse of the symbolic
  names) and `getnumparam` (parameters per type in the declarative table). `T137` starts the AFE layer: `appendafes`/`getnumafe` and the sparse-per-row storage of `F` and `g` (`putafefentry`/`putafefrow`/`putafeg` and the getters). `T138` adds the conic domains (`append*domain` and `getnumdomain`/`getdomaintype`/`getdomainn`, with MSKdomaintypee's values) that an affine conic constraint is a member of. `T139` closes the layer with the reference-style `appendacc(domidx, numafeidx, afeidxlist, b)` (AFE + domain, adapter over the internal encoder), its getters `getnumacc`/`getaccn`/`getaccdomain`/`getaccafeidxlist`/`getaccb`, and the linear domains `R/RZERO/RPLUS/RMINUS` encoded as rows. The reference convention that `b` is **subtracted** (`F x + g - b`, as documented for both `appendacc` and `putdjc`) is applied here; the previous `+b` reading was a sign error, and `T139` passes `b = +2` for the same `(x0+x1-2) ∈ R+` model. `T140` adds the reference-style **DJC** layer: `appenddjcs` (empty slots) + `putdjc(djcidx, numdomidx, domidxlist, numafeidx, afeidxlist, b, numterms, termsizelist)` (an OR of clauses, each a conjunction of domains over affine expressions), `putdjcslice`, the getters `getnumdjc`/`getdjcnumdomain`/`getdjcnumafe`/`getdjcnumterm`/`getdjcdomainidxlist`/`getdjcafeidxlist`/`getdjcb`/`getdjctermsizelist` and the totals, plus `putdjcname`/`getdjcname`/`getdjcnamelen`. The big-M encoder is now the internal `djc_encode`, and only linear domains are representable in a DJC (a conic domain is `ERR_ARG`, a declared deviation). `T141` ports the transaction-cost LP of the StackOverflow document (free variables, absolute-value epigraph written as two rows, a cost-free degenerate column the presolve removes) and asserts the **face** optimum — value `0.8` plus `x_i >= x0_i` and `e'x = 0`, not the posted vector. `T142` covers the two parity hooks of arXiv:2508.03704 / *Seven Sins*: `||w||_1 <= 2` on sign-free `w`, and the non-integer powers `|x|^{3/2}`/`|x|^{4/3}` as `PPOW(2/3)`/`PPOW(3/4)` (§3.6). `T143` solves the max-Sharpe ratio of case A3 through the Charnes-Cooper transform (`y = kappa*w`, `y'Sigma y <= 1`, `e'y = kappa`), the ratio itself being non-convex. `T144` adds the bulk AFE surface (`emptyafefrow`/`emptyafefcol`, `putafeglist`/`putafegslice`/`getafegslice`, `putafefentrylist`, `getafeftrip`, `getafefnumnz`). `T145` adds the bulk ACC surface (`appendaccs`/`getaccs`/`getaccntot`/`putaccb`) and the sixth name table (`putaccname`/`getaccname`/`getaccnamelen`). `getdimbarvarj`/`getlenbarvarj` (lower triangle `d(d+1)/2`) and the
  `getmaxnum*` counters (`T116`; for var/con they are the current count — the
  arrays grow in place, a declared deviation — while cone/bar capacities are real).
- **Information items (T160)**: the reference's information database is readable
  end to end. `PRIMALinftypee` (DOU/INT/LINT) and the three contiguous enums
  `PRIMALdinfiteme` (0..115), `PRIMALiinfiteme` (0..136), `PRIMALliinfiteme` (0..21)
  carry the reference's own indices (read from `constants.html`, MOSEK 11.2.4,
  2026-09-19). `PRIMAL_getdouinf`/`getintinf`/`getlintinf` read by index,
  `PRIMAL_getnadouinf`/`getnaintinf` by name, and `PRIMAL_getinfmax` (max index + 1
  = `END`), `PRIMAL_getinfname` (index -> name) and `PRIMAL_getinfindex` (name ->
  index) enumerate the database without knowing the symbols. The items this solver
  actually measures (objective values, primal/dual violations, solution norms,
  model sizes and bound-type breakdowns, problem/solution status, `OPTIMIZER_TIME`,
  matrix density, `RD_*` counts) are computed; the items of the phases it does not
  have (MIO cut separators, folding, remote, per-phase times) answer 0 — a declared
  deviation, not an invented number. The **symbolic constants** are also covered:
  `PRIMAL_getsymbcondim`/`PRIMAL_getsymbcon` (1438 entries, max name length 60),
  `PRIMAL_symnamtovalue` (name -> decimal value) and `PRIMAL_iparvaltosymnam`
  (parameter value -> symbolic name). That table was **not** copied from a doc page
  (the live docs do not expose it): it was extracted by calling the reference
  library's own `MSK_getsymbcondim`/`MSK_getsymbcon`, so it is the reference's own
  data.
- **Reference parity surface (category C, T159)**: the remaining implementable
  entry points of the reference API are now present — `getlintparam`/`putlintparam`
  (64-bit parameter read/write, `putparam` by name, `writeparamfile`/`readparamfile`,
  `printparam`), the name generators (`generate{varnames,connames,conenames,barvarnames,accnames,djcnames}`),
  `getconeinfo`/`readsummary`, `optimizetrm`/`optimizebatch` (solve a batch of tasks
  over one env), `primalrepair`/`dualsensitivity`/`primalsensitivity`/`toconic`,
  the ACC/cone/bulk writers `putacc`/`putacclist`/`putafefrowlist`/`putcone`/
  `putbararowlist`, the "new" solution API (`solutiondef`/`getsolutionnew`/
  `putsolutionnew`/`putsolution`/`getsolutioninfonew` plus the per-index
  `putconsolutioni`/`putsolutionyi`/`putvarsolutionj`), string/handle I/O
  (`readlpstring`; `readopfstring` reads OPF, `readptfstring` remains a declared
  deviation returning `ERR_ARG`; `readdatahandle`/`writedatahandle`), basis solve
  (`initbasissolve`/`solvewithbasis`/`basiscond`), `computesparsecholesky`
  (sparse Cholesky via `spchol_factor_ord`, real AMD `perm`; `T264`), the UTF-8
  <-> wide-char conversions (`utf8towchar`/`wchartoutf8`, `T265`), and
  `clonetask`/`getdualproblem`/
  `getinfeasiblesubproblem`. Public surface **514** functions. The numerics of the
  new entry points are cross-checked against hand values in `T159` (Cholesky of a
  dense SPD, `B x = b` on the identity basis, the LP dual of `min e'x, e'x = 2`,
  a cloned task solving to the same optimum).
- **KKT robustness**: adaptive regularization for degenerate systems
## Build and run

```sh
make               # build (gcc -std=c99 -Wall -Wextra -pedantic -O2, 0 warnings)
make test          # reliability suite: 4706 checks (out/run_tests)
make run-samples   # 171 runnable examples (out/samples/*)
make clean         # remove the whole out/ directory
```

All build artifacts go into `out/` (gitignored); nothing is written into the
repository root or into `samples/`. No dependencies beyond `libm`. Clean under
AddressSanitizer + UndefinedBehaviorSanitizer across the whole suite and all
samples.

## Layout

| File | Role |
|---|---|
| `primal.h/.c` | Public API (task/env, optimize dispatcher, all `PRIMAL_*` functions) |
| `stdform.c/.h` | General problem → standard form, **sparse CSC** for A and Q (fixed-variable substitution, free-variable split, ranged rows); dense materialization on demand |
| `presolve.c/.h` | LP presolve (empty/singleton rows, empty columns) with exact primal+dual postsolve, plus a **homogeneous lift** that carries a Farkas direction back through the reduction log |
| `simplex.c/.h` | Two-phase primal simplex, source of both **Farkas rays** (positive phase-1 optimum → dual ray; ratio test with no leaving row → primal ray) |
| `ipm.c/.h` | Mehrotra interior point: dense augmented system (LP/QP) + sparse normal-equations variants for large LPs and large sparse QPs |
| `socp.c/.h` | Primal-dual conic IPM (SOCP): dense augmented LU, plus a sparse **NT hybrid** (Nesterov-Todd scaling → symmetric normal equations + sparse Cholesky, falling back to sparse LU on the augmented system when the scaling point degenerates near the cone boundary) |
| `sdp.c/.h` | Primal-dual SDP IPM (Nesterov-Todd) **and the unified conic path**: PSD blocks plus SOC/RQUAD plus exponential/power blocks in one augmented KKT system (arrow `A(z)/A(s)` for conic blocks, NT Schur complement for PSD blocks, Hessian-NT `Thin` rows for exp blocks, Jacobi equilibration before the dense LU, iterative refinement of the polish solve with Wilkinson stopping); the exp/power point is accepted only if its **measured** relative primal/dual infeasibility and duality gap meet the tolerances declared on the task, otherwise the solver reports "not solved" and the dispatcher falls back to λ_min/tangent cuts |
| `expcone.c/.h` | Closed-form barrier, gradient and Hessian of the **non-symmetric** exponential and power cones (`EXPCONE_PEXP/PPOW/RPOW`), dual-cone membership `K*`, the scaling point `W` solving `-∇f(W) = s`, and the primal-dual Hessian-NT scaling used by the native exp/power IPM |
| `linalg.c/.h` | Partial-pivot dense LU, **sparse LU (partial pivoting)**, dense & sparse Cholesky, Jacobi eigenvalues |
| `scaling.c/.h` | Row/column equilibration (exact power-of-two factors) |
| `mpsio.c/.h` | MPS/LP file I/O — the readers fill the task's name tables, the writer is still positional |
| `cbf.c/.h` | CBF v4 file I/O |
| `test_primal.c` | Reliability suite (4706 checks, T1..T252) |
| `samples/mosek_comparison/*.c` | 48 ports of public MOSEK examples (matching values) |
| `samples/*.c` | deterministic larger examples across the problem classes (LP/QP/MILP/SOCP/SDP/exp cones, bar+quadratic shapes, SOS certificates, secure-EE beamforming, log-contrast LASSO logistic, quantum separability, max-clique Lovasz theta, smallest enclosing ball, MOSEK Tutorials ports) |
| `samples/finance/*.c` | 18 portfolio/regression examples (conic finance): the 7 from arXiv:1310.3397 plus `transaction_cost` (Doc 5), `lsq_pos` (MosekRegression's exact `0<=w<=1` form) and `sharpe_ratio` (arXiv:2508.03704 case A3 via Charnes-Cooper) and `evar_portfolio` (MOSEK Portfolio Cookbook ch8, entropic value-at-risk via exponential cones) and `market_neutral` (factor-neutral long-short with leverage, turnover and a variance cap) |

## Dual conventions

For the min-normalized problem (max → multiplied by s=−1):

- stationarity: `c + Qx + A'y + z = 0`, with `z = slx + sux`
- `y_i > 0` ⇒ row i at its **upper** bound, `y_i < 0` ⇒ at its **lower** bound
- `z_j > 0` ⇒ variable j at its **upper** bound, `z_j < 0` ⇒ at its **lower** bound
- `Y`/`Z` above are the min-normalized multipliers — `s·` the reported `t->y` and
  `t->slx + t->sux` (the getters split them into `slc/suc` and `slx/sux` by sign) —
  and `b_i^act`/`xb_j^act` is whichever of `blc/buc` (or `blx/bux`) the multiplier's
  own sign selects at the point published.
- dual objective: `dobj = cfix − Σ Y_i b_i^act − Σ Z_j xb_j^act − ½ x'Qx`, published in the
  problem's own sense. The constant is a term of the objective **as written**, so it is
  added outside the `s` that normalizes the sense — multiplied by `s` it misses by
  `2·cfix` on a max problem (measured: `T92` case A).
- strong duality `|pobj − dobj| ~ 0` verified by `T17` and `T92` (five routes: LP in max
  sense, QP, dense conic, bar/SDP, power cone). Before `T92` the two sums building `dobj`
  omitted `cfix` while `pobj` started from it: on the LP of `T17` (`cfix = 7.5`) the
  published dual was `2.0` against a primal `9.5` — the dual of the model minus its
  constant — and the dense conic report had the same shape.

## Conventions against the reference solver

MOSEK's documentation is readable from this machine (11.2.4: `capi/constants.html`
for the enumerations, `capi/parameters.html` for the 277 identifiers with their
defaults and accepted ranges, `capi/response-codes.html` for the result codes,
`capi/accessing-solution.html` for which status each optimizer reports), and the
comparisons below were read from it rather than inferred. Where a number is
shared it is pinned by a test.

Solution status (`PRIMAL_getsolsta`) carries MOSEK's numbering, so a raw status
read against `MSKsolsta` means the same thing here:

| Value | Here | MOSEK 11.2.4 |
|---|---|---|
| 0 | `PRIMAL_SOL_STA_UNKNOWN` | `MSK_SOL_STA_UNKNOWN` |
| 1 | `PRIMAL_SOL_STA_OPTIMAL` | `MSK_SOL_STA_OPTIMAL` |
| 5 | `PRIMAL_SOL_STA_PRIM_INFEAS_CER` | `MSK_SOL_STA_PRIM_INFEAS_CER` |
| 6 | `PRIMAL_SOL_STA_DUAL_INFEAS_CER` | `MSK_SOL_STA_DUAL_INFEAS_CER` |
| 9 | `PRIMAL_SOL_STA_INTEGER_OPTIMAL` | `MSK_SOL_STA_INTEGER_OPTIMAL` |

Only the five statuses this solver produces are declared. Before `T93` the
numbers were this solver's own and three of them read as a different verdict:
`INTEGER_OPTIMAL` was 2, which `MSKsolsta` calls `PRIM_FEAS`; `PRIM_INFEAS_CER`
was 4 (`PRIM_AND_DUAL_FEAS`); `DUAL_INFEAS_CER` was 5 (`PRIM_INFEAS_CER`).

Problem status is reported too, by `PRIMAL_getprosta`, with MOSEK's
`MSKprostae` numbering:

| Value | Here |
|---|---|
| 0 | `PRIMAL_PRO_STA_UNKNOWN` |
| 1 | `PRIMAL_PRO_STA_PRIM_AND_DUAL_FEAS` |
| 2 | `PRIMAL_PRO_STA_PRIM_FEAS` |
| 3 | `PRIMAL_PRO_STA_DUAL_FEAS` |
| 4 | `PRIMAL_PRO_STA_PRIM_INFEAS` |
| 5 | `PRIMAL_PRO_STA_DUAL_INFEAS` |
| 6 | `PRIMAL_PRO_STA_PRIM_AND_DUAL_INFEAS` |
| 7 | `PRIMAL_PRO_STA_ILL_POSED` |
| 8 | `PRIMAL_PRO_STA_PRIM_INFEAS_OR_UNBOUNDED` |

The reference fixes which pair of numbers may appear together (Tables 7.2 and
7.3 of *accessing solution*), so `prosta` is **derived** from `solsta` by one
function rather than stored beside it — the two cannot drift out of sync the way
the objective constant did:

| Outcome | `prosta` | `solsta` | Solution getters |
|---|---|---|---|
| continuous optimum | `PRIM_AND_DUAL_FEAS` (1) | `OPTIMAL` (1) | answer the point |
| primal infeasible, certificate | `PRIM_INFEAS` (4) | `PRIM_INFEAS_CER` (5) | refuse; `getdualray` answers |
| dual infeasible / unbounded | `DUAL_INFEAS` (5) | `DUAL_INFEAS_CER` (6) | refuse; `getprimalray` answers |
| primal infeasible, **no vector** | `PRIM_INFEAS` (4) | `UNKNOWN` (0) | refuse |
| unbounded, **no vector** | `DUAL_INFEAS` (5) | `UNKNOWN` (0) | refuse |
| integer optimum proved | `PRIM_FEAS` (2) | `INTEGER_OPTIMAL` (9) | answer the point |
| integer point, optimality unproven | `PRIM_FEAS` (2) | `PRIM_FEAS` (2) | answer the incumbent |
| node cap hit, **no incumbent** | `UNKNOWN` (0) | `UNKNOWN` (0) | refuse |
| infeasible integer problem | `PRIM_INFEAS` (4) | `UNKNOWN` (0) | refuse |
| nothing solved | `UNKNOWN` (0) | `UNKNOWN` (0) | refuse |

The last-but-one row is not hypothetical: when the node cap stops the search with
work still on the stack, the incumbent is a feasible point, not a proof, and
`sched` does exactly that (100000 nodes, 27 open, incumbent 97.328509 — the true
makespan, never established). Before `T94` that case printed
`INTEGER_OPTIMAL` with `rc = OK`. The pair is asserted on the raw numbers in
`T94` H; the rows where the two halves disagree are the ones `T98` added —
an infeasible *integer* problem used to be the only one, and now every verdict
that has no Farkas vector beside it is in the same situation. `t->prosta` is an
override field precisely for those: `prosta` is still derived by one function, so
a solve either follows the table or says, in one place, that the certificate is
missing.

The fourth column is `T99`. It is read from the getters and not from the flag:
six of those rows carry a verdict and have no answer to hand out, and two of
them — the certificated ones — hand out a *ray* while refusing the point. `T96`
had already made a *not solved* answer refuse; what `T99` closes is the routes
that did reach a verdict, which until then raised the flag to be readable.

The same lesson applied to the input side. A bound key, a cone type or a solution
item passed *in* by its reference number used to select a different object here,
with no error to read: `PRIMAL_BK_FX` was 4, which is `FR` in
`MSKboundkeye`, so a caller fixing a variable asked this solver to free it. The
enumerations now carry the reference's numbers (`T94`):

| Enum | Reference numbers adopted |
|---|---|
| `PRIMALboundkeye` | `LO 0, UP 1, FX 2, FR 3, RA 4` |
| `PRIMALsolt` | `ITR 0, BAS 1, ITG 2` (`ITG` added) |
| `PRIMALconettypee` | `QUAD 0, RQUAD 1, PEXP 2, DEXP 3, PPOW 4`; `RPOW 7` (`DPOW 5`, `ZERO 6` left unused) |
| slice items (`PRIMAL_getsolutionslice`) | `XC 0, XX 1, Y 2, SLC 3, SUC 4, SLX 5, SUX 6, SNX 7` |
| `PRIMALoptimizer` | `DUAL_SIMPLEX 1, FREE 2, INTPNT 4, PRIMAL_SIMPLEX 8` |
| parameter `kind` (`PRIMAL_getparaminfo`) | `DOU 1, INT 2` (0 and 3 refused) |
| `PRIMALvartypee` | `CONT 0, INT 1` (already matched) |

Parameter defaults are the reference's where the reference states one and this
solver's behaviour matches (`T84` reads each default back from a fresh task and
checks the declarative table against itself; the values below are the ones the
reference publishes):

| Here | MOSEK 11.2.4 | default |
|---|---|---|
| `PRIMAL_IPAR_INTPNT_MAX_ITERATIONS` | `MSK_IPAR_INTPNT_MAX_ITERATIONS` | 400 |
| `PRIMAL_IPAR_SIMPLEX_MAX_ITERATIONS` | `MSK_IPAR_SIM_MAX_ITERATIONS` | 10000000 |
| `PRIMAL_DPAR_INTPNT_TOL_{PFEAS,DFEAS,REL_GAP}` | same names | 1e-8 each |
| `PRIMAL_DPAR_INTPNT_TOL_NEAR_REL` | `MSK_DPAR_INTPNT_CO_TOL_NEAR_REL` | 1000 |
| `PRIMAL_DPAR_MIP_TOL_ABS_GAP` | `MSK_DPAR_MIO_TOL_ABS_GAP` | 0.0 |
| `PRIMAL_DPAR_MIP_TOL_REL_GAP` | `MSK_DPAR_MIO_TOL_REL_GAP` | 1e-4 |
| `PRIMAL_DPAR_MIP_TOL_INTHER` | `MSK_DPAR_MIO_TOL_ABS_RELAX_INT` | 1e-5 |
| `PRIMAL_DPAR_MIP_TOL_FEAS` | `MSK_DPAR_MIO_TOL_FEAS` | 1e-6 |
| `PRIMAL_IPAR_OPTIMIZER` | `MSK_IPAR_OPTIMIZER` | `FREE` (2) |
| `PRIMAL_DPAR_OPTIMIZER_MAX_TIME` | `MSK_DPAR_OPTIMIZER_MAX_TIME` | -1 (no limit) |
| `PRIMAL_DPAR_MIO_MAX_TIME` | `MSK_DPAR_MIO_MAX_TIME` | -1 (no limit) |
| `PRIMAL_DPAR_LOWER_OBJ_CUT` | `MSK_DPAR_LOWER_OBJ_CUT` | -inf (no cut) |
| `PRIMAL_DPAR_UPPER_OBJ_CUT` | `MSK_DPAR_UPPER_OBJ_CUT` | +inf (no cut) |
| `PRIMAL_DPAR_SEMIDEFINITE_TOL_APPROX` | `MSK_DPAR_SEMIDEFINITE_TOL_APPROX` | 1e-10 |

`PRIMAL_IPAR_OPTIMIZER` used to default to 0, which after the numbering was
adopted is the reference's `CONIC`; the value is behaviourally the free choice
here, but a fresh task now reports what the reference reports.

`PRIMAL_DPAR_MIP_TOL_REL_GAP` used to default to 0 (off) as a declared
deviation, on the argument that pruning on the relative gap alone accepts a
deliberately suboptimal incumbent. With the reference's 1e-4 the whole corpus is
unchanged: the suite (1503 checks at the time of that measurement, now 2607) ran
green at `-O0/-O1/-O2/-O3` and so did the 68 samples. The
deviation was therefore dropped and the default aligned.

Deviations that remain, each stated because the reference says otherwise:

- **There is no third verdict, and there did not need to be one.** `MSKsolsta`
  has no `NEAR_` member; MOSEK's categories are optimal, certificate, and
  unknown, and a stalled or numerically uncertain iterate is reported
  `MSK_SOL_STA_UNKNOWN` with `MSK_PRO_STA_UNKNOWN`. What MOSEK has instead is
  `MSK_DPAR_INTPNT_CO_TOL_NEAR_REL` (default 1000, range [1,+inf]): if the
  prescribed accuracy is not reached, the point is checked against the
  termination criteria with *all tolerances multiplied by that factor*, and if
  it passes it **is declared optimal**. This solver now has the same rule, as
  `PRIMAL_DPAR_INTPNT_TOL_NEAR_REL` (setting it to 1 disables it), and applies
  it where a measured triple decides: the unified conic IPM's gate. **The policy
  is two-sided.** Where an alternative algorithm exists — exp/power, the tangent
  cut outer approximation — the **declared** tolerances keep choosing the route,
  because a factor may not make the solver hand back a point the alternative
  beats. Where this IPM is the only conic route — bar and SOC models, no exp/power
  block — the factor **is** the verdict, so `maxcut_sdp` (`rel_gap` 1.32e-8) and
  the `T88` model (`rel_pri` 1.01e-8) are now published `OPTIMAL` exactly as the
  reference would, instead of being reported "not solved". Nothing else had to be
  re-tuned: the iterate a path integrates does not depend on the scale of the
  three tolerances, because the snapshot selector ranks by
  `viol = max(rel_i/tol_i)` and its improvement test `viol < 0.99·viol_best` is
  homogeneous of degree 0 — moving a tolerance moves the *judgement*, never the
  point (measured on the whole corpus, and locked by **T96**).
- **Two integer tolerances, as the reference has them.** Declaring an integer
  constraint satisfied and declaring a point feasible are different questions,
  and conflating them publishes an infeasible solution: with
  `PRIMAL_DPAR_MIP_TOL_INTHER` (the reference's `MSK_DPAR_MIO_TOL_ABS_RELAX_INT`,
  1e-5) adopted and nothing else changed, `samples/djc1` answered the max case
  with `x0 = 10` — outside both disjuncts (`x0 <= 2` OR `6 <= x0 <= 7`) — at
  `rc = OK` and `solsta = INTEGER_OPTIMAL`. A big-M disjunction amplifies the
  rounding: a selector at `1 - 5e-6` is integral within 1e-5, and rounding it
  moves its row by **5**, not by 5e-6. The reference keeps a second tolerance for
  the second question, `MSK_DPAR_MIO_TOL_FEAS` (1e-6, accepted [1e-9;1e-3]), and
  so does this solver now: `PRIMAL_DPAR_MIP_TOL_FEAS` is what
  `mip_point_measures` judges a candidate incumbent against — bounds, rows,
  cones, semi domains and SOS sets — while `MIP_TOL_INTHER` governs only the
  search. A candidate that does not measure is not discarded: the integers are
  pinned at their rounded values and the node's relaxation is re-solved
  (fix-and-optimize), which is still a relaxation of that node and loses no
  integer point. Measured: `djc1` answers `x0 = 0` (min) and `x0 = 7` (max) with
  `getprimalinfeas = 0` at every threshold tried — default, 1e-9, 1e-5, 1e-3
  (**T95**) — so the tolerance that used to be a deviation is the reference's
  own.
- **Mixed-integer parameters live in MOSEK's `MIO` group**, and there is no
  `MSK_IPAR_MIP_MAX_NODES`: the node budget is `MSK_IPAR_MIO_MAX_NUM_BRANCHES`
  (default -1, MOSEK chooses). `PRIMAL_IPAR_MIP_MAX_NODES` keeps this solver's
  name and a concrete default of 100000.
- **Three interior-point tolerance sets, as the reference has them.** MOSEK splits
  `INTPNT_TOL_*` (LP), `INTPNT_CO_TOL_*` (conic) and `INTPNT_QO_TOL_*` (quadratic);
  this solver now has all three, each read by its own route: the plain set by the LP
  route, `CO_TOL_*` by the conic route (`sdp_ipm`/`socp_solve`) and `QO_TOL_*` by the
  quadratic route (`ipm_solve_qp_csc` and the dense QP `ipm_solve_std`). Defaults and
  accepted ranges are the reference's (`parameters.html` 11.2.4: 1e-8, `[0,1]`). The
  previous single-set behaviour was a declared deviation, now closed.
- **Accepted ranges now match the reference's.** Read again from
  `parameters.html` (11.2.4) and aligned 2026-09-22: `TOL_PFEAS`/`TOL_DFEAS` are
  `[0,1]` (the earlier table widened them to `[DBL_MIN, DBL_MAX]`, a misread —
  only `TOL_REL_GAP` is `[1e-14,+inf]`), `MIO_TOL_REL_GAP` is `[0,+inf]` (was
  `[0,1]`) and `MIO_TOL_ABS_RELAX_INT` is `[1e-9,+inf]` (was `[DBL_MIN,1]`).
  The iteration limits used to require `>= 1` where MOSEK accepts 0 (its "no
  limit"): **closed** — the range is `[0, INT_MAX]`, the stored value is 0 (what
  a getter reads back) and every use site maps 0 to the largest int (`iter_cap`),
  locked by **T84** B2. `T94` F reads the `TOL_PFEAS` row.
- **The optimizer time cap is enforced.** `PRIMAL_DPAR_OPTIMIZER_MAX_TIME`
  (default -1 = no limit, range `[-inf,+inf]`) sets a wall-clock deadline that
  `PRIMAL_optimize` publishes to the IPM loops (`ipm_set_deadline`) and to the
  B&B loop; a fired cap returns `PRIMAL_RES_TRM_MAX_TIME` (1008), and the conic
  verdicts are skipped (a timed-out solve has no answer to judge). MOSEK names
  the code `MSK_RES_TRM_MAX_TIME`; the numbering is our own (as for every `rc`).
  `PRIMAL_DPAR_MIO_MAX_TIME` (same default/range) caps the **mixed-integer phase
  only**: the B&B obeys the tighter of the two deadlines. Locked by **T174**/
  **T175**.
- **The lower objective cut is enforced.** `PRIMAL_DPAR_LOWER_OBJ_CUT` (default
  `-inf` = no cut, range `[-inf,+inf]`) is checked in the IPM loops: a
  **primal-feasible** point whose objective is below the cut proves the optimum is
  below it, and the solve terminates with `PRIMAL_RES_TRM_OBJECTIVE_RANGE` (1009).
  `PRIMAL_DPAR_UPPER_OBJ_CUT` (default `+inf`) is its dual-side twin: a
  **dual-feasible** point whose dual objective (`b'y` for an LP) is above the cut
  proves the optimum is above it. The upper cut is enforced on the **LP route**
  only (the QP/conic routes do not compute that bound), and both cuts are wired
  for **minimization** — a maximization would need the mirrored pair. Locked by
  **T176**/**T177**.
- **The PSD tolerance is declared.** `PRIMAL_DPAR_SEMIDEFINITE_TOL_APPROX`
  (default 1e-10, range `[1e-15,+inf]`) is read as the **relative** factor of the
  encoder's convexity threshold (`bad_thr = tol*max(lmax,1)`) and of the witness
  PSD-cone check (`e >= -tol*(1+emax)`). The reference's is an **absolute**
  tolerance — a declared deviation. Locked by **T178**.
- **`getprimalinfeas` does not measure membership of the point in the cones.**
  Both getters now
  read the row as written: scalar coefficients, the bar product `⟨A_ij, B_j⟩`
  (off-diagonal counted twice, the same pairing `sdp_bar_row()` uses) and, for a
  `PRIMAL_putqconk` row, `quad_row_value()` — the helper `[cones]` already uses, so
  there is no second sign convention to keep in step. `PRIMAL_getdualinfeas` adds
  `max(0, -λmin(barsj[j]))` for every bar block that was actually published, and
  since **T101** the dual cone of every *readable* conic block (`d = s·(c + A'y)` in
  `K*`, relative to the block, with a block left unmeasured when a member carries a
  finite bound or the model has a quadratic term). Before
  this, a row whose whole content was a bar lost its left side and kept its right
  side as a violation: the 2×2 AM-GM model of **T96** A, solved to `rel_pri = 1.7e-9`,
  published `getprimalinfeas = 1` (**T97**). What the getters still do **not** do:
  primal membership in a QUAD/RQUAD/PEXP/PPOW cone is the fourth figure, printed as
  `[cones] rel_slack=` and asserted by **T90**, and the signs of
  the row multipliers are not part of `getdualinfeas`. The reference's own convention
  for either was not read, so neither was invented.
- **Solution information is exposed per index and as a summary (`T113`).**
  `PRIMAL_getpviolcon`/`getpviolvar`/`getpviolbarvar`/`getpviolcones` and the dual
  `PRIMAL_getdviolcon`/`getdviolvar`/`getdviolbarvar`/`getdviolcones` write the
  violation of the indices listed in `sub`, and `PRIMAL_getsolutioninfo` reports
  the maxima of those plus `pobj`/`dobj`. The primal definitions are the
  reference's exactly: row `max(l − a'x, a'x − u)` with `a'x` from `row_activity`
  (the three doors of `T97`), variable the bound/domain of `var_violation`, bar
  `max(−λmin(X), 0)`, cone `max(0, −cone_signed_slack)`, integer
  `min(x−⌊x⌋, ⌈x⌉−x)`. A per-index getter reads the **current** model against the
  published point, so perturbing a bound after the solve yields a known violation
  (on a row 2, on a variable 4, and — for the dual — a multiplier left on a side
  with no bound reads 1), and `getprimalinfeas == max(per-row, per-var)`. A refusal
  (bad index, no solution, null buffer) writes nothing. **Dual convention and
  declared scope:** the `dviol*` values follow this solver's dual convention (see
  «Dual conventions», whose sign is the reference's mirror), `dviolcones` is the
  reference's exact quadratic-cone formula but this solver's signed slack for the
  other cone types (the page does not spell out their generalization), and a
  variable that is a cone member is left **unmeasured** rather than measured wrong
  — the same boundary `getdualinfeas` draws.
- **The bar route publishes no bound slacks on a cone's own variables.** On a task
  with bars *and* cones the conic build answers (the cut route is skipped by
  `numcones > 0`), and it writes `slx/sux = 0` for the members of a cone block where
  the block's real dual is a nonzero vector — measured on `T101` I, where the dual of
  `(x0,x1,x2)` is `(5/3,-4/3,1)` and the published `slx+sux` reads `0`, while the same
  model without the bar (`T101` D) publishes `-d`. The consequence is not cosmetic:
  the dual feasibility of a cone cannot be read off those slots, so `cone_dual_worst`
  recomputes the block from `c` and `y` instead, which is also why the zeros do not
  turn into a reported violation.
- **Result-code numbers are this solver's own.** `PRIMAL_RES_ERR_ARG = 1001`,
  `TRM_MAX_ITER = 1007` and friends collide with MOSEK's license-error codes
  (1001 `LICENSE_EXPIRED`, 1007 `FILE_LICENSE`); MOSEK's iteration limit is
  `MSK_RES_TRM_MAX_ITERATIONS = 100000`. They are left as they are because MOSEK
  does not report infeasibility or unboundedness as a result code at all:
  `MSK_optimizetrm` returns `MSK_RES_OK` and the verdict is read from
  `MSK_getprosta`/`MSK_getsolsta`. Matching the numbers would mean matching that
  policy, which is an API change rather than a correction.
- **`PRIMAL_SOL_ITG` is an alias, not a third point.** MOSEK keeps the integer
  relaxation under `ITR`/`BAS` and the integral solution under `ITG`; this
  solver's branch and bound stores one point, so all three keys read it. `T94` C
  asserts the equality, so the deviation is visible rather than silent.
- **One solution item is not produced.** `SNX` ("lagrange multipliers
  corresponding to the conic constraints on the variables") has no counterpart
  here: its conic duals live per cone block rather than per variable.
  `PRIMAL_getsolutionslice` answers `PRIMAL_RES_ERR_ARG` for it. `XC` ("solution
  for the constraints") **is** served now: it is the row activity, read from the
  model (the three doors of `T97`), and `T94` D asserts it against `x0+x1`.
- **A name that is not there answers `PRIMAL_RES_ERR_ARG`.** Here "no such name" is
  the same code with which every other getter of this API says no — the ray getters
  of `T85` included — so one code means "no answer". The reference is recalled to
  have a dedicated code for a missing name, but its response-code page was
  unreachable this round (`curl` blocked by the permission classifier, `WebFetch`
  blocked on the retry), so that recollection is **not** the reason for the choice
  and is not stated as a fact: the deviation is declared against what this code
  measurably returns, which `getidxvar`/`getidxcon` document.
- **The whole-name getters hand out borrowed pointers, not characters.** The
  reference's `getallvarname`/`getallconname` fill caller-owned buffers of fixed
  name length; here the array elements are the task's own strings — valid until
  `PRIMAL_deletetask`, never to be freed — which is the same ownership
  `getvarnameidx` already documents. An unnamed slot reads `""` in both readings:
  the empty string is not a borrowed object, so it is asserted by content and not
  by address.
- **The readers and the writer name the task.** This deviation is **closed**: the
  writer emits each row and column under the name the task carries
  (`getconnameidx`/`getvarnameidx`/`getobjname`), falling back to `c%d`/`x%d`/`obj`
  only for an empty name or one containing whitespace (MPS/LP names are
  whitespace-delimited), and it writes the reference's `OBJNAME` section for the
  objective. A read → write → read round-trip now preserves the labels as well as
  `numcon` (`T123` fixed the row count, `T112` G asserts the names). The two
  tables keep their own namespaces and cannot collide: a name is unique *inside*
  its table (`T102`/`T105`), and the file puts variables and constraints in
  different sections.
- **The bar-name namespace is *our* decision, not a read rule.** That a bar's name is
  independent of the variable and constraint tables is the coherent extension of the
  rule measured for the two scalar tables in `T102`, applied to a table that MPS-style
  naming never had: the reference's rule for **naming bar variables** was not read (the
  documentation fetch stayed blocked at this session's permission level) and was not
  invented. What is measured is the consequence, both halves of it — one string naming
  a variable, a constraint and a bar at once, and the same string refused on a second
  bar (`T105` C).
- **The cone and objective namespaces are the same kind of decision, and one surface is
  deliberately absent.** The reference's rule for naming a **cone block** was not read
  either (the documentation fetch stayed blocked at this session's permission level), so
  the independence of the cone namespace is ours, stated as such and measured as a
  consequence. The gap list also names a wider surface we do **not** implement: the
  reference addresses some entities by a **(type, index)** pair (`getidxfun` over
  `MSKfunctiontypee`). It is absent on purpose, not by oversight: the numbering of that
  enum was not readable in this round, and giving it our own numbers is precisely the
  defect `T93` and `T94` spent two rounds correcting. A fake enum is worse than a
  missing getter, so the gap is declared rather than filled.
- **The two whole-`Q` readers answer in the shape of *our* counter.** `getqobj`
  enumerates the triplet store (write order, stored zeros, one entry per cross term)
  and `getqconk` the upper triangle of the dense symmetrized store — the very tables
  `getnumqobjnz`/`getnumqconknz` count, so `numret` cannot disagree with the number
  the user was able to ask for. The reference's shape for these two readings was
  **not read** (the documentation fetch stayed blocked at this session's permission
  level) and was not invented: the two `*ij` getters answer the same value on both
  halves of a cross term either way, and `x'Qx` multiplies it by 2, so which list is
  handed out does not change the operator that gets solved.
- **Two contracts, one store: the *store* reads back entries, the *scalar* reads
  back the operator.** An `(i,j)` written twice is stored twice — `putarow` clears the
  row and appends, nothing deduplicates — and the store-reading surface says so:
  `getnumanzs` counts both entries and `getarow`/`getacol` hand both back. The scalar
  read of one `(i,j)`, `getaij`, answers the **operator** instead: it returns the
  **sum** of the stored entries, which is the coefficient every route that solves the
  model actually used. It used to read the **first**, so the number describing the
  answer contradicted the answer (measured: `putarow(0,{0,0},{1,2})` with `3x <= 3`
  solves at `x=1` on the coefficient 3 while the getter said 1). Chosen on **parity**,
  not preference: `getqobjij` already sums and `putqconk` accumulates into the dense
  matrix so `getqconkij` already summed — `getaij` was the only exception. The
  reference's write-time policy was **read** (2026-09-25): `MSK_putarow`/`putacol`
  clear and then **assign in sequence** (`a_{i,subi[k]}=vali[k]`), so a repeated
  index keeps the **last** value (no summing `appendrow`); we keep the divergence —
  storage unchanged — because our standard form sums, so `getaij` stays the operator.
  `T107` asserts
  the binding invariant `getaij(i,j) == Σ getarow(i) entries == Σ getacol(j) entries`
  over the whole matrix, plus that the summed number equals *neither* stored entry.

## How it compares

PrimalSolver is a research-grade convex solver, not a drop-in replacement for the
industrial LP/MIP engines — but the **combination** of what it solves is rare
among the permissively-licensed open solvers, and that combination is its niche.
This is a comparison of **capability, licence and accuracy class**, not of speed.

| feature | PrimalSolver | MOSEK | Gurobi / CPLEX | HiGHS | Clarabel | SCS |
|---|---|---|---|---|---|---|
| LP / QP | ✅ | ✅ | ✅ | ✅ | LP (QP via cones) | ✅ |
| SOCP | ✅ | ✅ | Gurobi only | — | ✅ | ✅ |
| **SDP** (PSD cone) | ✅ | ✅ | — | — | **—** | ✅ first-order |
| **Exp / power cones** | ✅ native | ✅ | — | — | ✅ | ✅ |
| MIP / MIQP | ✅ | ✅ | ✅ | ✅ | — | — |
| Method | interior-point | interior-point | simplex + IPM | simplex + IPM | interior-point | **first-order (ADMM)** |
| Certificates | ✅ | ✅ | ✅ | ✅ | ✅ | partial |
| Dependencies | **libm only (C99)** | — | — | C++ | Rust / C++ | C |
| I/O | MPS / LP / OPF / CBF | MPS / LP | MPS / LP | MPS / LP | native API | native API |
| Licence | **Apache-2.0** | commercial | commercial | MIT | Apache-2.0 | MIT |

**What no other permissively-licensed open solver covers** — said once:
**SDP + exponential/power cones + MIP + an interior-point (not first-order)
method, in dependency-free C99.**

- **Clarabel** (Apache-2.0) is the closest open conic IPM, but it has **no PSD cone**.
- **SCS** (MIT) has PSD and exp cones, but is **first-order** — accuracy around
  `1e-4`/`1e-5`, not the `1e-8` an interior-point method reaches.
- **CVXOPT / SDPA / Sedumi** have SDP, but are **GPL** (viral for commercial use).
- **HiGHS** is fast on LP/MIP, but has **no SDP and no non-symmetric cones**.

The niche that follows: embedded/enterprise code that wants SDP, exponential
cones and integer variables in C, with no dependencies and no licence server, and
a migration path from a MOSEK-style API (the 48 C ports of MOSEK examples are
reproduced with identical optimal values). Outside that niche — large LP/MIP at
industrial scale — it is not (yet) a substitute.

## Benchmarks: MOSEK 11.0 examples reproduced

PrimalSolver ports the published MOSEK 11.0 examples to its own API and reproduces
their optimal values — **48 C ports covering 47 distinct MOSEK examples** (the
`acc1` example is ported from both the Julia API and the C tutorial, with two
different models). Each port is independently verified: hand-derived values, KKT
via public getters, exhaustive enumeration, or a reference algorithm. A selection:

| Benchmark | Verification |
|---|---|
| `lo1`, `lo2`, `simple` | known optima, pobj=dobj |
| `qo1`, `portfolio_1/2` | Markowitz QP, monotone γ frontier |
| `qcqo1` | quadratic constraints, obj=√2 |
| `cqo1` | QUAD + RQUAD cones, obj=1/√2 |
| `ceo1`, `pow1` | exponential/power cones, e⁻¹ and power cone |
| `logistic` | softmax logistic regression, cross-checked against numerical gradient descent |
| `gp1` | geometric programming via log-transform + PEXP, x=(0.5, 0.5) |
| `mil1`, `milo1`, `mi1`, `mioinfeas1` | MILPs with optima confirmed by enumeration |
| `mil2` | semi-continuous + semi-integer, x=(3,0) |
| `sos1`, `sos2` | SOS1 (obj=5/6), SOS2 piecewise-linear y(1.5)=0.75 |
| `mioinitsol` | MILP with initial solution, optimum by enumeration |
| `mico1`, `portfolio_4/5/6` | MIP+cones, MIQP with binaries, cardinality, factor models |
| `sdo1`, `sdo2`, `sdo_lmi` | SDP: tr(X), dual bar variables, active LMI at the PSD boundary |
| `dual_sdo_l1` | separate primal/dual SDPs, strong duality 0=0 |
| `sparsecholesky` | tridiagonal A: PSD via SDP = direct Cholesky factorization |
| `nearestcorrelation` | min ‖A−X‖_F s.t. X PSD, diag=1 — same matrix as Higham alternating projections (‖·‖_F = 1.8107 in both) |
| `pinfeas`, `mioinfeas1` | infeasibility with certificates |
| `solvebasis` | basis → file → solve from basis, pobj=75 restored |
| `sensitivity` | cost range confirmed by re-solve |
| `callback`, `response` | progress and log callbacks |
| `acc1`, `acc1_max`, `djc1` | ACC on affine expressions (obj=−4), the MOSEK acc1 tutorial (max c'x s.t. Σx=1, γ≥‖Gx+h‖, obj=3.280511), disjunctions (min=0, max=7) |
| `reoptimization`, `parameters`, `solutionquality`, `feasrepairex1` | incremental re-solve, parameters, ~0 violations, minimum-violation repair |
| `solvelinear` | 3×3 linear system via LP, ~0 residual |
| `concurrent1`, `parallel` | 4 optimizers on the same problem, same optimum; independent tasks |
| `sched` | 30-task/6-processor scheduling, makespan 97.3285 (proven optimal) |
| `hello` | end-to-end smoke test |

This is compatibility validation against the public examples: it demonstrates
that PrimalSolver solves those problems and reproduces the expected optimal
values. It is **not** a claim of equivalence with, or of industrial robustness
comparable to, MOSEK on arbitrary large-scale problems. The example models are
MOSEK's (Copyright (c) MOSEK ApS); PrimalSolver is an independent implementation
that re-solves the same problems.

## Scaling examples

Beyond the MOSEK ports, `samples/` (root) contains larger, deterministic
examples. Most stress the engine across every problem class: each one
checks its own result (strong duality, an exact dynamic program, a closed form, or
an independent numerical method) and reports its size and solve time, so it can
also be used as a scaling probe (`./out/samples/<name> [args]`). Others probe an
API *shape* or an algorithm rather than a size — `barqcqp` puts symmetric bars in
the same model as quadratic rows/objectives, `lyapunov_roa` and
`sos_m1_certificate` build SOS certificates, `secure_ee_sdma` runs the
Dinkelbach + SCA outer loop of `rezarhp/Secure-EE-SDMA` on a K=1/J=1 downlink
(rotated SOC, power SOC and exponential cone in one model), and
`logcontrast_lasso` casts `audreyolmsted/admm-portfolio`'s log-contrast LASSO
logistic regression as exponential cones and compares the interior-point answer
with the repository's ADMM (Newton + joint proximal); `quantum_separability`
solves the PPT white-noise robustness SDP of `y1-zhu/quantum-correlations` over
a 9x9 semidefinite bar (checked against the closed form) and the CCNR
realignment criterion; `binary_quadratic` (Shor SDP), `exact_cover` (MIP), `pwl_convex`
(rotated cone regression), `equilibrium` (hanging masses, SOCP), `dist_robust`
(Wasserstein DRO, LP), `facility_location` (k-disk cover, MISOCP),
`wasserstein` (barycenter, LP), `mle_density` (log-concave MLE, exp cone), `rank_one` (best-subset MIP with
DJC), `kmeans` (MISOCP+DJC),
`filter_design` (trigonometric-polynomial SDP; its n=3 case needed the
multi-block PSD rescue, see below),
`f_sparc` (subcarrier/power allocation, MIP+exp),
`hard_uncertain` (robust uncertain inequality on box vertices, exp cones),
`transformer_design` (geometric program, exp cones), `surface_cycles` (shortest cycle
in a homology class), `gp_toolbox` (geometric program) and `min_enclosing_ball`
port MOSEK Tutorials; `lovasz_theta` solves the max-clique upper bound SDP
of `Rudolfovoorg/Improving_Upper_Bounds_of_MCP_using_Reduction_Rules` (Lovasz
theta of the complement) and checks it against `sqrt(5)` on C5:

| Example | Class | Default size | Verification |
|---|---|---|---|
| `transport` | LP | 60x60 = 3600 vars | strong duality |
| `assignment` | LP | n=80 (6400 vars) | integrality + strong duality |
| `network_flow` | LP | 200 nodes / 800 edges | balance + strong duality |
| `cvx_regression` | LP (L1) | m=400 / d=30 | strong duality |
| `lp_large` | LP (sparse, Ax=b) | 300x6000 = 6000 vars | planted KKT optimum + strong duality |
| `lp_ineq_large` | LP (sparse, Ax≤b) | 300x6000 = 6000 vars | planted KKT optimum + strong duality |
| `qp_sparse` | QP (sparse, Laplacian Q) | n=2000, m=8 | strong duality |
| `knapsack` | MILP | n=120 | exact O(nC) DP |
| `facility` | MILP | 15x40 | MIP >= LP bound |
| `portfolio_large` | QP | n=100 assets | strong duality |
| `socp_robust` | SOCP | m=300 / d=40 | normal equations |
| `maxcut_sdp` | SDP | K(4,4) | closed form -|E|/4 |
| `logistic_large` | exp cones | 40 cones | gradient descent |
| `secure_ee_sdma` | SOCP + exp cone (Dinkelbach/SCA) | K=1, J=1, Nt=2 | brute-force 2-D on the true objective |
| `logcontrast_lasso` | exp cones (logistic + L1 + log-contrast) | n=16, p=3 | conic vs ADMM vs brute-force 3-D |
| `quantum_separability` | SDP (9x9 bar) | qutrit isotropic | SDP vs closed form, CCNR realignment, witness |
| `lovasz_theta` | SDP (n x n bar) | C5, K3, E5 | SDP vs closed form, X valid, floor(theta)=omega |
| `min_enclosing_ball` | SOCP (QUAD cones) | 2 or 3 points | closed form d/2 and a/sqrt(3) |
| `binary_quadratic` | SDP (Shor relaxation) | n=4 | relaxation <= brute force 2^n |
| `exact_cover` | MIP (exact cover) | 2x3 / 3x3 | hand tiling (3) and infeasible case |
| `pwl_convex` | conic QP (rotated cone) | n=3 | hand solution t=1/3, SSE=2/3 |
| `equilibrium` | SOCP (QUAD cones) | 3 masses, 2 strings | closed form (0,-sqrt(3)) |
| `dist_robust` | LP (Wasserstein DRO) | m=2, N=3 | LP = brute force 1.1633333 |
| `facility_location` | MISOCP (k balls) | 4 points, k=2 | hand radius 0.5, cover checked |
| `wasserstein` | LP (barycenter) | 3 bins, K=2 | hand value 1.0, mu on the simplex |
| `mle_density` | exp cones | n=3 grid | hand value -3 log(4/3) |
| `rank_one` | MISOCP + DJC | n=3, p=2, k=1 | hand value 1.5, cardinality 1 |
| `kmeans` | MISOCP + DJC | 4 points, K=2 | hand inertia 1.0 (centroids 0.5, 2.5) |
| `filter_design` | SDP (Toeplitz) | n=3 | specs checked on a dense grid; multi-block PSD rescue in sdp.c |
| `f_sparc` | MIP + exp cone | I=1, J=1 | scan of t (z*=0.2206127) |
| `hard_uncertain` | exp cones (box vertices) | L=1, n=1, p=2 | scan of x (x*=1.08673) |
| `transformer_design` | geometric program (exp cones) | 15 vars, 28 constraints | loss=4.578609, all 28 posynomials <= 1 |
| `surface_cycles` | LP (shortest homologous cycle) | 16 edges, 8 triangles | min = 4, x is a cycle, x ~ C |
| `gp_toolbox` | geometric program (exp cones) | 3 vars, 2 constraints | obj=10.981644 |
| `min_circle` | SOCP (min enclosing circle) | 2/3/8 points | SOCP = exact brute force (2, 2.5, 0.5768) |
| `welzl` | Welzl exact vs SOCP | 20 sets x 10 points | same radius on all 20 |
| `steering_robustness` | SOCP (2x2 Hermitian PSD = Qr^4) | Werner, w in [0,1] | SR matches the reference dat |
| `barqcqp` | SDP + QCQP | 1 bar (2x2) + 2 scalars | hand-derived optimum, PSD minors of the bar, refusal code |

The default sizes are chosen to complete in seconds; larger sizes can be passed
on the command line (e.g. `./out/samples/transport 150 150`). Large LPs
(`m·n > 1.5e6`, `m ≤ 2000`) are routed automatically to the sparse
normal-equations interior point — e.g. `transport 150 150` (22500 vars) solves
in ~0.07 s vs ~12.7 s for the dense simplex, and `transport 300 300` (90000
vars) in ~0.61 s. `lp_large` (Ax=b) and `lp_ineq_large` (Ax≤b) are generic
sparse LPs with a planted KKT optimum, so their exact objective is known
independently of the solver; both route to the sparse IPM at default size
(~0.06 s for 6000 vars). Beyond the sparse-LP path, the dense QP/conic engine
and the outer-approximation round budgets remain the limiting factors — the
frontier described under Performance.

## Financial examples

`samples/finance/` collects portfolio-optimisation models, several following
Schmelzer, Hauser, Andersen and Dahl, *"Regression techniques for Portfolio
Optimisation using MOSEK"* (arXiv:1310.3397). Each is deterministic and checks
its own result:

| Example | Model | Class | Verification |
|---|---|---|---|
| `regression_ls` | min \|Xw-y\|_2 (constrained / portfolio form) | SOCP | normal equations |
| `regression_regularized` | ridge, LASSO, \|·\|^{3/2} penalties | QP + conic | closed form / cone feasibility |
| `markowitz_conic` | max μ'x s.t. \|G'x\| ≤ √γ | SOCP | risk bound, best-asset optimum |
| `cvar_portfolio` | CVaR (Rockafellar-Uryasev) | LP | empirical tail CVaR |
| `risk_parity` | equal risk contribution | exp cones | risk-contribution spread |
| `market_impact` | max μ'x - δ't, t_i ≥ x_i^{3/2} | power cones | cone feasibility |
| `portfolio_mgmt` | min-variance, tracking, max-return, 130/30, robust | QP + SOCP | budget / bounds / cones |
| `transaction_cost` | transaction costs (StackOverflow 37586543): `min t'z`, `z ≥ \|x−x0\|`, `e'x=0` | LP | value 0.8 + membership of the optimal face |
| `lsq_pos` | MosekRegression's exact form `min ‖Xw−rhs‖₂`, `(v,Xw−rhs)∈Q`, `0≤w≤1`, `e'w=1` | SOCP | QP-equivalent (`min ‖Xw−rhs‖²`) agrees to 1e-8 |
| `sharpe_ratio` | max `(μ'w−rf)/√(w'Σw)` (arXiv:2508.03704 A3) via Charnes-Cooper | QP + quadratic constraint | hand value √10 at `w=(3/4,1/4)` |
| `factor_model` | min `w'(FF'+diag(D))w`, `e'w=1` | QP | hand value 1/2 at `w2=1/2` |
| `robust_cvar` | worst-case CVaR over a return box | LP | hand value -0.10 robust vs -0.15 nominal |
| `risk_budgeting` | `min 1/2 w'Σw − Σ b_i ln(w_i)` (log via exp cone) | QP + exp cones | risk contributions = budgets `(0.8,0.2)` |
| `option_pricing` | no-arbitrage price bounds (risk-neutral polytope) | LP | call bounds `[0, 0.25]` at `K=1` |
| `cardinality_portfolio` | min variance, at most K assets used | MIQP | hand value 1/2 (K=2 of 3, Σ=I) |
| `mean_cvar` | max `mu'w − λ·CVaR_β(w)` (Rockafellar-Uryasev) | LP | hand value 12/35 at the kink `w=(4/7,3/7)` |
| `multi_period` | two-period rebalancing with turnover cost | LP | hand value 0.9 (move once, then hold) |

## License

All code is original, written from scratch in C99.


## TODO — algorithms still open

PrimalSolver solves LP, QP, SOCP, SDP, exponential/power-cone and mixed-integer
problems. The algorithmic work still open (the maintained task list lives outside
the repository):

- **Basis warm start** — no real crash basis, and `solvebasis`/`solvewithbasis`
  re-solve from scratch when the given basis does not measure instead of
  re-optimizing *from* it.
- **I/O formats** — TASK and PTF have no reader/writer (`readdataformat` refuses
  both; MPS, LP, OPF and CBF are supported).
- **Sparse augmented system in `sdp.c`** — the unified conic KKT is assembled
  densely (`Sys[Nsys²]`), which caps the problem size.
- **Symmetric KKT in `socp.c`** — the SOCP KKT is assembled in an unsymmetric
  form, so the sparse symmetric `LDLᵀ` cannot replace the sparse `LU` there yet.
- **Parameter coverage** — sixteen parameters are wired against the reference's
  hundreds (declared deviation; the table is declarative, so adding rows is cheap).
- **Exp/power polish floor** — on `logistic_large` no iterate enters the declared
  relative triple, so the tangent cuts answer, and the native route itself can
  move with the optimization level (the KKT is conditioned past `1/eps`).
- **A conic extreme not attained inside a bar** has no recession direction and is
  reported as "no verdict" rather than a value.
