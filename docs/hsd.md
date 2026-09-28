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

## Next steps

1. Fix the scale handling / degenerate-feasible branch selection in the probe.
2. Generalize `K` to SOC/SDP/exp, reusing the barriers and scalings already in
   `sdp.c` / `expcone.c`.
3. Add the `(tau, kappa)` row to the unified conic KKT in `sdp.c` as route
   `CONIC2`, out of the default until it matches on the corpus.
