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
