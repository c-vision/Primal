# SOCP second-order corrector

The sparse production NT and augmented-LU engines now include the affine
predictor product in the Mehrotra corrector. Previously the second solve only
changed the centering target `sigma*mu`. The dense solver is unchanged.

For SOC blocks, let `W=Q(sqrt(w))`, where `Q(w) lambda=s`, and
`v=W lambda=W^-1 s`. In balanced coordinates the dual shift is

```
z = (W^-1 ds_aff) o (W dlambda_aff)
shift = -W^-1 A(v)^-1 z
```

Here `o` is the Jordan product and `A(v) q = v o q`. For nonnegative
coordinates the shift is `-ds_aff*dlambda_aff/s`. The unscaled augmented-LU
system receives the raw negative Jordan product. The affine solve, centering
rule, ratio test, stopping tolerances and iteration limits are unchanged.
If balanced scaling cannot be evaluated, the existing centering-only step
remains available.

The experimental `GMB_SOCP_LDL` backend retains its previous corrector.
Applying the new shift there caused T72 to stagnate without FMA contraction
at gap 9.05e-10, above its requested 1e-10; the unchanged backend solves it.
This is a deferred numerical limitation, not a reason to relax that test.

## Verification and reproduction

T289 reconstructs W by an independent eigendecomposition of Q(w), instead
of the Jordan square root used by the implementation. It checks
`A(v) W shift + z = 0` on 16 SOC inputs, R+ values and rejected boundary
inputs (37 checks). The maximum SOC residual in the initial oracle run was
3.99e-17. A zero-correction negative control fails 18 of those checks.

From the repository root, create the [benchmark Python environment](../docs/benchmarks.md#reproduce).
The baseline is `aa6b378`, the parent of the corrector commit `0180c52`.
Use a clone that contains both revisions.

```sh
baseline=out/socp-baseline-aa6b378
mkdir -p "$baseline"
git archive aa6b378 | tar -xf - -C "$baseline"
make -C "$baseline" lib
gcc -std=c99 -O2 -I"$baseline" bench/socp_compare.c \
  "$baseline/out/libprimal.a" -lm -o "$baseline/out/socp_compare"
out/benchmark-venv/bin/python tools/socp_compare.py --runs 31 \
  --out out/socp_compare-corrector --baseline-worker "$baseline/out/socp_compare"
```

The worker links the archived library. The runner builds the candidate from
the current checkout, alternates execution order, warms up three times,
removes inherited GMB flags, and saves raw observations, source hashes,
quartiles, public solution quality and target verdicts in `results.json`.
Use revision `0180c52` for the original candidate and run without competing
builds or tests. The runner writes the results before returning a nonzero
status when a timing threshold is missed.

Primal timings cover PRIMAL_optimize, including internal model conversion;
Clarabel timings cover construction plus solve, with solve-only also reported.
Public model construction is excluded for both. Clarabel has explicit 1e-8
feasibility and gap tolerances; Primal retains its defaults. Both must report
optimal status and agree with the closed-form objective. Cases include the
canonical family, unequal weights with two equalities, and rotated cones.

Predeclared acceptance: SOCP200 at least 40% faster and within 1.5x Clarabel
construction+solve; SOCP120 at least 20% faster; SOCP40 and each held-out case
at most 5% slower. These criteria concern the default path. They do not claim
parity on all SOCP models, especially rotated cones.

See the [current benchmark results](../README.md#benchmarks).
