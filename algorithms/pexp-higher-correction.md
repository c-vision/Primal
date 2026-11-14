# Higher-order PEXP correction

The mixed SOC/PEXP native route uses a higher-order predictor-corrector term.
It can converge without the second, secant-mode native attempt on models such
as `logistic_large`. The scalar tolerances, conic acceptance criteria and
fallback policy retain their existing definitions.

## Equation

For the primal exponential barrier

    f(z) = -log(z1*log(z0/z1)-z2) - log(z0) - log(z1),

let H be its Hessian and dz_a, ds_a the full affine directions. Solve

    H u = ds_a,
    eta = -0.5 D³f(z)[u, dz_a].

The corrector's scaled cone right side becomes

    rt = (-s - sigma*mu*grad(f) - eta) / scale,
    scale = <z,s>/3.

The third derivative is contracted analytically; no finite differences or
third-order tensor are used at runtime. For the orthant analogue this gives
eta=ds_a*dz_a/z, the familiar Mehrotra cross term. A failed Hessian Cholesky
or nonfinite contraction yields eta=0. Directions used here precede the
fraction-to-boundary step.

The metric and gradient depend only on the current iterate. They are cached
between predictor and corrector, with only rt recomputed for sigma*mu. Each
outer iteration invalidates the cache. The same cache covers both Hessian-NT
and secant mode; their existing eligibility rules are retained.

## Scope and implementation

Automatic selection requires a PEXP block, a SOC/RQUAD block and no PSD
blocks. Power-only, pure exp and PSD models use the existing engine.
`GMB_NO_EXP_HIGHER=1` selects the previous engine for diagnostics.

`expcone_corrector.c` implements the contraction and cache. `sdp_corrected.c`
compiles the shared `sdp.c` implementation with `SDP_IPM_CORRECTED`; the
constructor in `primal_sdptask.c` selects its private CSR entry point.
The mathematical implementation is maintained once. There are two compiled
engines: on the measured build this increases the static library by about
110 KiB (1,090,296 to 1,203,200 bytes). There is no new runtime dependency.
The original `sdp.o` is byte-identical to the isolated baseline's object.

## Measurements, 2026-11-05

Baseline: archived sources at `06f767b`, built in an isolated source directory.
Candidate: F110, Apple clang 21, C99 `-O2`, same machine, single process per
solve. Two independent batches of nine alternating baseline/candidate pairs;
each cell shows the two medians. Timings include `PRIMAL_optimize`, with model
construction excluded on both sides. These are local measurements of these
families, not a general speed guarantee.

| Model | Baseline ms | Candidate ms | Reduction |
|---|---:|---:|---:|
| Exponential regression 20×4 | 28.00 / 27.57 | 14.95 / 14.41 | 46.6–47.7% |
| Exponential regression 40×8 | 91.47 / 94.60 | 61.27 / 61.69 | 33.0–34.8% |
| Exponential regression 80×12 | 121.67 / 126.28 | 92.51 / 94.34 | 24.0–25.3% |
| Pure exp entropy, N=120 (control) | 23.13 / 23.16 | 24.03 / 23.02 | −3.9% / +0.6% |

The 40×8 case finishes on the first native attempt. Its public getters give
pobj=47.093377298528971, dobj=47.093377284348769, primal infeasibility
1.09e-13, dual infeasibility 0 and relative objective gap 1.49e-10.
The entropy control has identical numerical trace and unchanged engine.

Clarabel 0.11.1, on the same mathematical 40×8 model, takes about 1.70 ms
median over nine runs including constructor+solve, excluding construction of
canonical sparse matrices. Its settings tolerances were 1e-10, with an
additional absolute feasibility check <=1e-8 on the original exponential
and quadratic inequalities. The final reported objective is
47.09337729630259. This reference check was separate from the interleaved
Primal comparison: it establishes a remaining gap of tens of times on this
case, not a controlled cross-solver speed ratio or a comparison of identical
internal representations.

## Sparse KKT selection

The corrected SOC/PEXP engine now estimates the density of both KKT forms
before selecting the sparse path. If that attempt fails, the solver retries
with the previous routing policy.

When both sparse forms are eligible, the corrected engine starts with
the scalar variables in the augmented KKT if their diagonal entries and
links occupy fewer positions than the Schur block. On the 80×12 model,
the order increases from 588 to 853 and the structural nonzeros decrease
from 11,404 to 7,705. The final route still checks the restored solution
against the task's tolerances.

Two batches of nine alternating measurements against `5be1e52` gave:

| Model | Previous routing (ms) | Sparse selection (ms) |
|---|---:|---:|
| Exponential regression 20×4 | 12.01 / 11.94 | 3.60 / 3.40 |
| Exponential regression 40×8 | 49.27 / 48.09 | 9.66 / 9.70 |
| Exponential regression 80×12 | 71.56 / 72.75 | 44.18 / 45.93 |

Clarabel 0.11.1 at one thread took 1.38 ms in both batches on the 40×8
problem, including construction and solve. Its model substitutes out the
affine arguments of the exponential cones and retains the quadratic
epigraph. Both solvers met the `1e-8` benchmark thresholds for objective
agreement, feasibility and relative gap.

The README comparison uses a separate clean build and covers all three
sizes. Clarabel uses a tolerance of `1e-9` for feasibility and gap. At
`1e-10` the 80×12 case returned `AlmostSolved`, although its measured
metrics met the `1e-8` benchmark limits. The published run uses one
setting across all sizes and requires `Solved` as well as the numerical
checks. Timings from the two settings belong to separate comparisons.

The focused group passed 928 checks in default, forced dense and legacy
modes. Forced sparse mode reproduced the same two DEXP checks that fail
on the baseline. An injected sparse factorization failure verified the
retry and reproduced the dense result.

### Mixed SOC and exponential cones, 2026-11-12

Three changes closed most of the gap on the mixed class, each measured on
`bench/pexp_bench.c` (the instance that used to live only in a scratch
directory; `bench/pexp_bench.py` rebuilds the table and asserts the optima):

- **No size floor on the sparse trial.** The trial required `order >= 128`
  on top of the density estimate. With cone-member absorption (below) the
  20×4 model lands at order 112 and took the dense route, where the sparse
  one is 2.8× faster (0.98 ms against 2.75 ms). The density estimate is now
  the only criterion.
- **Cone-member absorption.** A cone member whose variable occurs in exactly
  one cone position and is free — or carries only a bound the cone implies —
  stops being a scalar column plus a linking row: its coefficients act on the
  block directly and its cost joins the block's cost. Measured: order
  222 → 112, 438 → 238, 854 → 454 on 20×4/40×8/80×12.
- **Secant first on the mixed class.** With both a SOC block and exp/power
  blocks the secant engine runs first and the Hessian-NT chain is the retry
  (it was the other way round). The recorded counter-example that kept the
  chain first, `gp1`, has three PEXP cones and no SOC, so it is outside the
  class; `logistic_large` (40 PEXP + 1 RQUAD) is in the class and is solved
  by the secant.

Medians of 15 interleaved runs per build, before and after:

| Model | Before (ms) | After (ms) | Gain |
|---|---:|---:|---:|
| 20×4 | 0.99 | 0.70 | 29% |
| 40×8 | 2.37 | 1.45 | 39% |
| 80×12 | 7.39 | 3.68 | 50% |

The published point was compared field by field between the two builds
(`x`, `y`, `slx`, `sux` in `--dump` mode): it agrees to 1e-5 relative, well
inside what two different trajectories of a scaled cone problem can be
expected to agree, and it measures in the model (`rel_pri`, `rel_dual`,
`rel_gap` inside `1e-8`, `[cones] rel_slack = 0`). The focused conic/exp
group (T36, T47, T75-T83, T86, T87, T90-T92, T96, T97, T100, T101, T161,
T172) passes with `PASS: 993 FAIL: 0`.

Known deviation: absorption makes `PRIMAL_putxx` silently partial — the warm
value of an absorbed variable is not used to seed the cone coordinate. The
infeasible-start engine does not need it; a caller that relies on the warm
point for those variables loses it.

## Stopping with a verified point

The corrected mixed route can finish the polish once its absolute primal
and dual equation residuals meet the corresponding task tolerances,
its relative gap is at most half the requested gap tolerance, and both
cone margins are nonnegative. The final route gate still measures the
restored point against all three declared relative tolerances. The gap
margin allows for the different normalizations used in the solver and
in public verification.

On the three regression models, the number of completed predictor–corrector
iterations fell from 29, 26 and 37 to 16, 16 and 21. The original-model
quality checks passed at `1e-8`. The [README](../README.md#mixed-soc-and-exponential-cones)
reports timings for the current routing and stopping policy.

## Refactoring the sparse LU

The sparse KKT pattern is fixed for the whole solve, only its values move. After
the first factorization of an iteration sequence the corrected engine keeps the
factor and, at the start of every later iteration, recomputes only its values
with `splu_refactor` (`linalg.c`): the row and column permutations and the
structure of L and U are reused, no pivot search is done and nothing is
allocated. The first factorization uses `splu_factor_keep`, which keeps entries
that are exactly zero (a zero multiplier, a zero fill) so that the structure
does not depend on cancellations: the arrow matrices of a SOC block start as the
identity and fill in later.

The pivot order of an earlier iterate can be less stable than a fresh one, so
both predictor and corrector directions using a refactored factor are
checked on the system that was factored. Their componentwise backward
error must be finite and at most `1e-12` (a fresh factor reaches
about `1e-16`; the refactored solutions measured `1e-15` to `3e-13`), with up to
two refinement steps with the same factor. A solution that does not pass, or a
refactorization that does not fit the old structure, falls back to a full
factorization with a new pivot order. On the three regression models the
refactorization was accepted in 14 of 15, 14 of 15 and 18 of 19 factorizations.
`GMB_NO_LU_REFACTOR` disables it. Test `T308` checks refactored solutions
against fresh factors, repeated refactorization of one factor, the refusal of a
zero pivot and the kept zeros.

Median solve time of nine runs against the previous routing (`9f09b1c`):
20×4 from 2.29 to 1.82 ms, 40×8 from 6.53 to 4.56 ms and 80×12 from 20.1 to
12.7 ms. The [README](../README.md#mixed-soc-and-exponential-cones) reports the
timings of the benchmark harness.

## Cone step bounds

The corrected route tests each exponential or power block at the smallest
step allowed by the blocks examined so far, rounded upward to the grid of
the existing 50-round bisection. If that point is interior, the block cannot
tighten the step and needs no bisection. A rejected probe uses the original
search. The solver checks primal and dual cone membership separately.

## Corrector assembly and ordering

The sparse corrected route can keep the predictor's equilibrated KKT
matrix, row and column scales, and cone residual terms for the corrector.
This requires the same orthant form and valid scaling caches for all
exponential and power blocks. The corrector still forms its own centered
right-hand side and runs the residual check when using a refactored LU.

The minimum-degree ordering uses a bounded index of active nodes by
degree. It keeps the original lowest-index tie break. Large problems
and allocation failures use the linear search.

## Validation and limits

T299 checks the contraction against independent Taylor jets, log-homogeneity
and bitwise cache equivalence for predictor/corrector at different sigma
values in both scaling modes. T300 checks a mixed-cone problem with known
optimum 1 through public objective, point and feasibility getters, with both
engines. The focused group totals **928 passed, 0 failed**, also with forced
dense KKT, forced sparse KKT and the legacy-engine switch.

The `logistic`, `market_impact` and `risk_parity` examples pass their checks.
This change has not rerun the full suite or sanitizer suite. It does not
establish faster LP/MIP, pure exp, power-only or SDP solves.

An initial benchmark baseline was accidentally rebuilt with the first cache
prototype. Those intermediate comparisons are invalid as comparisons with
HEAD and are preserved as measurement failures in the local attempts log.
The table above uses the replacement isolated baseline, whose library hash
was checked before and after the comparisons. Future comparisons must never
run the shared-tree Makefile against a frozen baseline output directory.

## Reusing the 3x3 eigendecomposition

`nt_eval` diagonalises three 3x3 matrices per block and iteration: the dual
Hessian at `z` (for `condz`), the inverse Hessian at the scaling point `W` (for
`conds` and `Hs^(1/2)`), and the sandwiched `Hs^(1/2) Hz Hs^(1/2)` (for `rel`).
The scaling step then asked for `Hs^(-1/2)` and the sandwich's `+1/2` power,
re-diagonalising the same two matrices with the same deterministic Jacobi
routine: six eigendecompositions where three suffice. The decompositions are now
stored in `expnt` (`evHs/VHs`, `evMid/VMid`) and the spectral function is applied
from them. The arithmetic is unchanged, so objectives and getters are
bit-identical, not merely within tolerance.

Measured 2026-11-11 with interleaved baseline/candidate probes on the three
regression models (machine load ~9.3): median reduction **5.6% at 40×8** and
**4.0% at 80×12** across the series; 20×4 is in the noise. Targeted guards
(T77–T83, T90, T91, T171, T172, T298–T300, T308) pass 945/0.

## Metrics only when asked

`nt_eval` also produced `condz`, the two cone margins and `wn`. They are read by
`expcone_nt_metrics`, which the solver calls only under `GMB_DBG`; the scaling
path uses none of them. `nt_eval` now takes `want_metrics`: the scaling passes 0
and the metrics pass passes 1. The dual Hessian at `z` is still evaluated — it is
the fallback scaling — while its eigendecomposition and the two margin
logarithms are skipped. The `[expm]` trace under `GMB_DBG` is unchanged (1320
lines compared byte for byte on 40×8).

Interleaved medians against the previous engine (three series of 31 pairs):
**5.0% at 20×4**, **3.5% at 40×8** and **3.0% at 80×12**; objectives are
bit-identical and the targeted guards pass 945/0.

## Spectral factors evaluated once

`spd_from_eig` applies `V diag(lam^p) V'` to a 3×3 matrix whose eigendecomposition
is already known. The same `pow(ev[k], p)` used to be evaluated inside the
triple loop, 27 times per application; the factors do not depend on `i` or `j`.
They are now computed once (three `pow` calls) and reused. The function and the
order of the products are unchanged, so the results are bit-identical.

Interleaved medians (three series of 31 pairs, plus the README harness):
**4–5% at 80×12** and **5–6% at 40×8**; 20×4 ranges from −0.5% to −5.3%.
The targeted guards pass 945/0.

## Cone members absorbed into their block

The conversion in `primal_sdptask.c` used to give every cone member a new
coordinate tied to its variable by a linking row, and to keep the variable as a
scalar column (two columns when it is free). A continuous variable that sits in
exactly one position of one cone, and is free or carries only a lower bound the
cone already implies, is now the cone coordinate itself: no linking row, no
scalar column. Its coefficients in the data rows act on the block and its cost
goes to the block. For the first two members of a rotated quadratic cone,
`u = (z_0 + z_1)/sqrt2` and `v = (z_0 - z_1)/sqrt2`; when only one of them
qualifies, a single row `(z_0 ± z_1)/sqrt2 = other member` remains. Exponential
and power cone members were absorbed first, quadratic cone members followed.
`GMB_NO_EXP_ABSORB` and `GMB_NO_SOC_ABSORB` restore the linking rows.

On the README regression model at 80×12 the quadratic step takes the sparse KKT
system from 454 unknowns and 5550 nonzeros to 415 and 4144, and the iteration
count from 18 to 12 (16 → 10 at 20×4, 16 → 11 at 40×8). Interleaved medians of
`bench/pexp_bench.py --runs 21`, three series, 2026-11-13: **0.71 → 0.45 ms**
at 20×4, **1.65 → 0.92 ms** at 40×8 and **4.48 → 2.21 ms** at 80×12, objectives
within `1e-8` of the known optima. The pure exponential and pure SOC families of
`bench/pexp_matrix.c` are unchanged. Test T310 solves every combination of
absorbed and kept members both ways and requires the same point.

## No polish rounds after a verified refactorization

A refactored LU is verified on every right-hand side: its backward error must
be at most `1e-12`, and it takes refinement steps of its own when the first
solve misses that bound. The polish rounds of the sparse KKT solve then ran on
top, each costing two residuals and one solve, on a solution already at that
level. They are now skipped when the verification has just passed; a full
factorization, which is not verified, still gets them. `GMB_LU_REFINE_ALL`
restores the rounds. The step probes of the exponential cones also stopped
asking for the barrier value they never read (three logarithms per probe).

Minimum of 41 runs against the previous engine, 2026-11-13: **−8% at 20×4**,
**−9% at 40×8**, **−7% at 80×12**, with the same iteration counts (10/11/12).
Over 126 generated exponential and mixed instances the total iteration count
is 1996 before and 1997 after; the mixed ones are unchanged one by one.

## Degree order for the sparse KKT pattern

The column order of the sparse KKT factorization came from an explicit-fill
minimum-degree pass, 7% of the solve at 80×12. The pattern is now ordered by
ascending structural degree (a counting sort, `O(nnz)`). On the conic patterns
measured — exponential blocks of three coordinates, one dense quadratic block,
a few dense rows — the fill of `L + U` is the same or within 0.4%. The order
has no fill model, so the first factorization checks it: when `nnz(L + U)`
exceeds `2 nnz + 4 n` the minimum-degree order is computed and kept if its
factor is smaller. `GMB_KKT_AMD` asks for the minimum-degree order directly.

Minimum of 41 runs, 2026-11-14: **−6% at 20×4**, **−7% at 40×8**, **−11% at
80×12**, −18% on a mixed model with 160 blocks; pure exponential models gain
2–9%. Over 126 generated instances the total iteration count goes from 1997
to 1996. All 171 C examples keep their verdict.

## Singleton pairs solved outside the factorization

A cone member fixed by its bound — the `1` in every `(v, 1, u)` exponential
cone of the regression model — leaves two trivial lines in the sparse KKT
system: a row whose only entry is `K[r,c]` (coordinate `c` equals a constant)
and the column of its multiplier, whose only entry `K[c,r]` sits in the row of
that coordinate. Such a pair is solved exactly, outside the LU:

$$u_c=\frac{g_r}{K_{rc}},\qquad u_r=\frac{g_c-\sum_j K_{cj}u_j}{K_{cr}},$$

with column `c` moved to the right-hand side first. The detection is
structural and generic (`sdp_sp_reduce`), not tied to a cone type. For the rows
to be single, the zero-valued data links are now left out of the pattern also
when the model has no scalar columns. `GMB_NO_KKT_PIN` keeps the full system.

At 80×12 the factored system goes from 415 to 255 unknowns and from 3972 to
2772 nonzeros. The same measurements showed that the dense route was up to
twice as slow as the sparse one on models with 3 to 15 blocks, so systems of
order at most 128 now take the sparse form up to a structural density of 0.3
(0.1 above). The engine also reads `GMB_DBG` once per call instead of at every
trace point.

Minimum of 41 runs against the previous engine, 2026-11-14: **−10% at 20×4**,
**−12% at 40×8**, **−8.5% at 80×12**; a mixed model with 10 blocks goes from
0.43 to 0.24 ms and a pure exponential one from 0.87 to 0.43 ms. Over 144
generated instances the total iteration count is 2523 before and 2525 after.
Test T311 solves one model with the reduction, without it and on the dense
route, and requires the same point.
