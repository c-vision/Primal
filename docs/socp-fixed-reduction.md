# Fixed-coordinate reduction and sparse LU incidence

The sparse conic route substitutes fixed scalar variables before solving.
The existing dense/sparse routing is unchanged; dense solves keep their
original call without a reduced attempt. If
`x_f=b_f`, the reduced data are `d_r=d-E_f b_f`, `h_r=h+G_f b_f`,
`E_r=E_free`, `G_r=G_free` and `c_r=c_free`. The two opposite nonnegative
rows used to encode each fixed bound are removed. SOC rows remain intact,
including constant shifts on rotated-cone heads. A zero equality is dropped
only when its shifted right-hand side is exactly zero. An inconsistent row,
allocation failure or unsuccessful reduced solve uses the original model.
An all-fixed model also retains the original path.

After success, fixed coordinates are restored exactly and the original
constraint multipliers are lifted. For fixed coordinate j, compute
`r_j=c_j+(E'y)_j-(G_other'lambda)_j`. The removed lower and upper bound
multipliers become `max(r_j,0)` and `max(-r_j,0)`. They are nonnegative,
restore stationarity in the original model and include the fixed objective
constant in the dual objective. Dropped zero-equality multipliers are zero.

Relative stopping criteria are kept at least as strict after translation.
The reduced gap tolerance is divided by `1+abs(c_f'b_f)`. The feasibility
tolerance is multiplied by the smaller of 1 and the original/shifted
`1+infinity-norm` ratios for d and h. The reduced c norm cannot increase.
No declared tolerance, centering rule or iteration limit is increased.

## Sparse factorization

The augmented LU fallback orders columns `[lambda,x,y]`: local cone
coordinates before the equality separator. This is a column permutation of
the existing matrix; the existing row pivoting and solution mapping remain.

The LU kernel keeps exact column-to-row incidence in optional bitsets.
It enumerates candidate rows in increasing physical row order, preserving
pivot ties and elimination order for a given column permutation. A row
rebuild clears its old active bits and installs the nonzero updated pattern;
row swaps move the union of both patterns. Columns already eliminated need
no further updates. Cancellation, fill and swaps therefore cannot hide an
active entry. The accelerator uses at most 8 MiB. Larger matrices or an
allocation failure retain the original row scan. The lowest set bit is found
with a portable De Bruijn multiply (`ctz64` in linalg.c, checked on all 64
positions), not a compiler builtin: the library compiles with any C99
compiler.

An independent comparison against the committed legacy kernel produced
byte-identical L/U factors and permutations on 86 matrices, including
singular cases and dimensions 63/64/65/127/128/129/257 and diagonal cases at 8192/8193, on either
side of the accelerator memory budget. T290 checks original
matrix residuals and known solutions with cyclic swaps and fill across
bitset boundaries. T291 checks the public primal and lifted dual on 30
rotated cones with fixed members, nonzero fixed objective cost, a redundant
fixed-only equality and a shifted inequality, in both objective senses;
an inconsistent fixed-only equality cannot publish a point.

## Performance acceptance

Baseline: bb1beaf, including the earlier Mehrotra corrector and direct
rotated-cone encoding. Before experiments the targets were at least 30%
less time on rotated SOCP200, at most 3x contemporaneous Clarabel
construction+solve, and no more than 5% regression on any of the other six
cases. Quality and status must pass unchanged. The final acceptance profile also
guards small dense rotated models at sizes 2/10/40 against the same 5%
regression limit. Failed standalone
substitution, inverse/scaling formulas and alternative orderings remain in
the local attempts register and its vault mirror.

Reproduce with a preserved statically linked baseline worker:

```sh
/Users/gaetano/ai/env-bench/bin/python3.14 tools/socp_compare.py --runs 31 \
  --target rotated --small-fixed --out out/socp_compare \
  --baseline-worker /absolute/path/to/baseline/build/socp_compare
```

The runner includes all reduction, allocation, ordering and incidence
maintenance inside PRIMAL_optimize. It reports Clarabel construction+solve
and solve-only separately, records raw observations and source/binary
hashes, and fails if any predeclared acceptance gate fails. Final timings
and validation are recorded in README.md.
