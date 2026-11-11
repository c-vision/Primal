# Homogeneous self-dual embedding (F4, prototype)

The next conic route (plan phase F4) embeds the primal-dual pair directly, so an
interior-point method needs **no Phase I** and an infeasible or unbounded model
returns a **certificate** instead of a status. This is the structure Clarabel
uses; the reference is Chen & Goulart 2023 (paper in the vault, readable with
`pdftotext`), eq. 8-9 and algorithm 6.2.

## The embedding (eq. 8)

For `min c'x  s.t.  Gx = h,  Ax + s = b,  s in K` and its dual
`max -h'y - b'z  s.t.  A'z + G'y + c = 0,  z in K*`:

```
G'y + A'z + c tau = 0
-G x + h tau      = 0
s = -A x + b tau
kappa = -c'x - h'y - b'z
(s, z, kappa, tau) in K x K* x R+ x R+
```

The system is homogeneous: if `tau > 0` the pair is feasible and
`(x, y, z, s)/tau` is optimal; if `kappa > 0` the model is primal or dual
infeasible, which is read off `h'y` and `c'x`/`z`.

## The algorithm (6.2)

Predictor-corrector on the central path `s = -mu g*(z)`, `tau kappa = mu`, with
Mehrotra's corrector, `sigma = (1 - alpha_a)^3`, and a **regularized** linear
system (the paper regularizes the augmented sparse LDL; the prototype adds a
small diagonal shift to the dense solve). The residual is the current one (the
`-mu q` form of eq. 9 is the central-path definition, not the affine RHS).

## Prototype status (`bench/hsd_probe.c`, not linked)

Dense, LP standard form (`K = R_+`), verified on:

| model | verdict |
|---|---|
| `min -x0`, `x0+x1=1` | OPTIMAL `(1,0)`, obj -1 |
| `min -x2`, `x0-x2=0` | DUAL INFEASIBLE / unbounded (`c'x<0`, `z` out of `K*`) |
| `x0+x1=1` and `=2` | PRIMAL INFEASIBLE (Farkas `h'y>0`, `G'y<=0`) |
| `x0=1` and `x0=2` | PRIMAL INFEASIBLE |
| `min -x0`, `x0-x1=0` | DUAL INFEASIBLE |

The certificate regime works end to end. Open in the prototype: a **degenerate
feasible** case (`x` fully determined by the equalities) drifts to the `kappa`
branch, and the anti-collapse rescaling interacts with the certificate
thresholds (the HSD is homogeneous, so the scale has to be fixed without moving
the branch).

## In-tree HSD (`hsd_psd`, opt-in `GMB_SDP_HSD`)

Separate from the prototype above: `sdp.c` carries an opt-in **PSD + scalar** HSD
embedding (`hsd_psd`), default **off**. Its only regression cover is `T197` — the
degree-3 axisymmetric M1 max-margin SDP of `jcpaik/p2-kkt-flag-sos`, reference
margin `-0.85506573` (agreed by Clarabel / SCS / MOSEK). It is the smallest
instance where W's own NT scaling matrix becomes ill-conditioned near the
optimum, which is what the bar-block `dS` elimination had to get right.

## Reproducing the sanitizer-only failure (#11)

An ordinary `-O2` build solves T197; the **same** source under LLVM 22
`-fsanitize=address,undefined` returns `PRIMAL_RES_TRM_MAX_ITER` with **no**
ASan/UBSan report. Recipe (LLVM 22 is not the Apple clang; Homebrew carries it):

```sh
brew install llvm@22                      # 22.1.8 on arm64 macOS
CC=/opt/homebrew/opt/llvm@22/bin/clang
# library objects, then the isolated probe (test_primal.c renamed main):
$CC -std=c99 -O2 -g -fno-omit-frame-pointer -pthread -I. -fsanitize=address,undefined \
    -c <LIBSRCS> 
$CC ... -o t197.c <objs> -lm              # ASAN_OPTIONS=detect_leaks=0 at run time
```

Measured (this tree): plain `-O2` passes, the sanitized build fails the two T197
checks. `MallocScribble=1` / `MallocGuardEdges=1` do **not** change the plain
result, so it is not an allocator-layout sensitivity. The trace shows the two
builds identical through it≈20 and then divergent (the `dual` residual already
differs at 1e-13 by it=10); at it≈31 the merit backtracking collapses:

```
[HSD] step  it=31 alpha=8.077e-07 alpha0=0.8469 sigma=0.01 accepted=1
[HSD] merit it=31 old=1.192e-08 ref=1.192e-08 T.pf=9.169e-08 T.df=3.6e-15 T.mu=2.4e-10
```

The cone allows `alpha0 = 0.847`, but the `pf+df+mu` merit rejects every length:
the Newton direction **raises `pf` ~8x while lowering `mu`**. From there `pf`
plateaus near 1e-8, so `pri = pf/(tau (1+|b|))` stays around 3.9e-8 and the
termination criteria are never met. A centring rescue (force `sigma=1` for a
burst of 15 when the accepted step collapses below 1e-4, taking the first
feasible step besides the merit — the same diagnosis that closed #22 in the
primary path) fires at it=40, 80, 167 but `pf` plateaus anyway, so it was **not**
kept. The step/merit policy is not robust to a codegen-level floating-point
difference on this degenerate model; the fix belongs here, in the HSD rewrite.

Related: the same LLVM 22 ASan+UBSan recipe reproduces the old **#22** (T244,
mixed SDP/SOC ellipse) on `b4799b0` and `7e2ea11` and it **passes** from
`6735a28` on, so that one was a primary-path convergence issue already fixed by
the centring rescue, not an HSD issue.

## Attempt 2026-10-01 — the step made literal does NOT reproduce the 5/7

Read algorithm 6.2 from the paper (`pdftotext -layout`, lines 1343-1400 of the
extract) and compared it with `hsd_step`. The prototype deviates in **four**
places, plus one missing constraint:

1. affine dual-Hessian coefficient `μ H*(z)` is **0** in the probe (paper: `μ`);
2. corrector coefficient is `σμ` (paper: `μ` — the coefficient is the same in
   both steps, only the RHS changes);
3. combined linear RHS is the **full** residual (paper: **`(1-σ)·`** residual);
4. the Mehrotra corrections are absent (`η(Δs_a,Δz_a)` and `Δτ_aΔκ_a`);
5. `α` never limits `Δz`, so the step can push the **dual** iterate out of `K*`
   (paper: `α` keeps `(s,z,τ,κ)` in `F`, so `z + αΔz ≥ 0` too).

Making the step literal (1-4) regresses the probe: `min -x0`, `x0+x1=1` goes
from `OPTIMAL (1,0)` to DUAL INFEASIBLE; the 3-var case returns a wrong
`obj=-17.3` with negative `x`. Adding the `z ≥ 0` limit on top, or applying it
alone to the original RHS, makes it unstable (`tau`/`kap` → `nan`/`1e60`).

**Conclusion (measured, both variants reverted):** the prototype's 5/7 is **not**
a rigorous base — its four deviations are *compensating*, so "fixing" the step
toward the paper breaks it. Tuning the step against the whole solve is exactly
the loop the handoff warns against. The next move is the **unit test on the
single step**: given a point, verify the affine direction satisfies
`μH*(z)Δz + Δs = −s` and the linearized rows to machine precision, and the
combined satisfies the paper's RHS — *then* re-tune. That test would have
caught deviation 1 immediately.

## Next steps

1. Unit-test the single step (see above) before touching the solve again.
2. Fix the scale handling / degenerate-feasible branch selection in the probe.
3. Generalize `K` to SOC/SDP/exp, reusing the barriers and scalings already in
   `sdp.c` / `expcone.c`.
4. Add the `(tau, kappa)` row to the unified conic KKT in `sdp.c` as route
   `CONIC2`, out of the default until it matches on the corpus.
