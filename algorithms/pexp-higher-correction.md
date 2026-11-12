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
