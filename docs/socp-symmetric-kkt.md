# Experimental symmetric SOCP KKT

The sparse SOCP solver has an opt-in backend selected by the **presence** of
`GMB_SOCP_LDL`. It is disabled by default. It applies to nonnegative cones
and SOC blocks of at most 32 coordinates, the existing NT scaling limit.
Larger SOC blocks retain the existing path. This does not change the SDP
solver or implement a homogeneous embedding.

## Linear equations

With residuals `rd = c + E'y - G'lambda`, `rp = Ex-d`, and
`rg = Gx+h-s`, the direct NT direction satisfies

```
E dx                         = -rp
E' dy - G' dlambda           = -rd
ds                           = rg + G dx
ds + Q(w) dlambda            = -Q(w)lambda + sigma*mu*Q(w)s^{-1}
```

`Q(w)` is the SOC Jordan quadratic representation; on a nonnegative
coordinate it is `s/lambda`. Eliminating `ds` gives the symmetric system

```
[ 0   E'   -G'  ] [ dx      ]   [ -rd                                  ]
[ E   0     0   ] [ dy      ] = [ -rp                                  ]
[-G   0   -Q(w) ] [ dlambda ]   [ rg + Q(w)(lambda - sigma*mu*s^{-1})    ]
```

This is an NT system, not the nonsymmetric unscaled arrow system used by
the legacy LU fallback. The old matrix cannot simply be handed to LDL.
The direct path also avoids constructing a numerical inverse of `Q(w)`.
Near a cone boundary, that inverse and the original matrix need not be
mutually accurate enough to serve as independent numerical oracles.

## Factorization and acceptance

`SocpKkt` retains the CSC pattern, locations of the changing cone entries,
ordering, elimination tree, factor storage and solve/refinement buffers.
Only cone values and numerical factors change between iterations.
The predictor and corrector share the same numerical factors.

`SpQd` applies symmetric max-entry equilibration and a signed diagonal
shift of `1e-10` in equilibrated coordinates: positive on primal variables,
negative on equality and cone variables. Fixed scalar pivots are accepted
only with the expected sign. This backend is intended for quasi-definite
systems, not arbitrary symmetric indefinite matrices.

Every solve is checked against the **unregularized** symmetric matrix.
The maximum residual divided by `1 + max(abs(rhs))` must be at most `1e-8`.
Up to four correction solves are allowed, stopping on stagnation. Failure
leaves the caller's RHS unchanged. A rejected IPM direction disables this
backend for that solve and restarts the same iteration through the legacy
engine, before modifying the iterate. Scaling or factorization refusal also
retains the legacy path. Final publication uses the existing IPM stopping
criteria and public solution checks; tolerances are unchanged.

## Evidence and limits

T284 checks sparse LDL against known points and dense LU, including numerical
refactorization with structural zeros and changed scaling. T285 checks all
equations of independently assembled symmetric systems and rejection of
invalid data. T286 checks public primal/dual feasibility, objective and gap
on coupled SOCs, unequal costs and rotated cones, against the legacy solver.

The initial inverse-based integration failed its residual gate and was
slower; it was replaced rather than accepted with a looser tolerance.
In the first 31 alternating before/after pairs, at `-O2`, SOCP 120 changed
from 1.566 to 1.185 ms and SOCP 200 from 2.820 to 1.087 ms. These are
targeted measurements, not a refreshed canonical benchmark table or proof
of parity with Clarabel. Full-suite and sanitizer results are pending at
this prototype checkpoint. Default routing must be evaluated after those
checks and on less symmetric problem families.
