# `primal` examples

The command-line front end `primal` reads a model file, solves it and prints
the solution.  **The model file is the interface**: you write the problem in one
of the formats below and pass its path; the solver picks the algorithm from what
the model contains, so LP, QP, SOCP, SDP, exponential/power and MIP all go
through the same command.

Build it (it comes with the default `make`):

```sh
make
./out/primal lp_examples/lp_diet.lp
```

Options: `--full` (also the duals and slacks), `--brief`, `--solution-file FILE`,
`--write FILE`, `--max-iter N`, `--tol-pfeas/--tol-dfeas/--tol-gap V`,
`--param NAME=VALUE`, `-h`.  Run `./out/primal --help`.

## The input formats

| format | extension | expresses |
|---|---|---|
| CPLEX LP | `.lp` | linear objective/constraints, `General`/`Binaries` |
| MPS | `.mps` | the standard exchange format (linear, quadratic objective, `QCMATRIX`) |
| OPF | `.opf` | linear + quadratic objective, conic domains (`quad`, `rquad`, `pexp`, `ppow`, …), integer |
| CBF | `.cbf` | conic + semidefinite (bar variables) |

Detection is automatic (by extension; an OPF file that starts with `[` is
recognised by content too).

The CPLEX LP reader is written for real files: terms may be glued
(`2x0`, `x0+x1`, `-x`, `2*x`, `1.5x0`, `1e-3x0`), constraints may be labelled or
not, a constraint may be ranged (`1 <= x + y <= 3`), bounds accept
`free`/`inf`/`-inf`, and `\` (and a leading `*`) start a comment.

## The examples

Each file is a complete model with its optimum written in the header.

### `lp_diet.lp` — a linear program (`.lp`)

`max 3 x0 + 2 x1` s.t. `x0 + x1 <= 4`, `x0 + 3 x1 <= 6`, `0 <= x0,x1 <= 3`.

```sh
./out/primal lp_examples/lp_diet.lp
```
```
status  : OPTIMAL   prosta=PRIM_AND_DUAL_FEAS  solsta=OPTIMAL
obj     : primal = 11   dual = 11
x[x0] = 3
x[x1] = 1
```

### `mip_knapsack.lp` — a binary knapsack (MIP, `.lp`)

`max 8 x0 + 5 x1 + 4 x2` s.t. `5 x0 + 4 x1 + 3 x2 <= 7`, `x in {0,1}^3`.

```sh
./out/primal lp_examples/mip_knapsack.lp
```
```
status  : INTEGER_OPTIMAL   prosta=PRIM_FEAS  solsta=INTEGER_OPTIMAL
obj     : primal = 9   dual = 9
x[x0] = 0
x[x1] = 1
x[x2] = 1
```

### `qp_portfolio.opf` — a quadratic program (`.opf`)

`min x0^2 + x1^2` s.t. `x0 + x1 = 1`, `x >= 0`.

```sh
./out/primal lp_examples/qp_portfolio.opf
```
```
status  : OPTIMAL   prosta=PRIM_AND_DUAL_FEAS  solsta=OPTIMAL
obj     : primal = 0.5   dual = 0.5
x[x0] = 0.5
x[x1] = 0.5
```

### `socp_robust.opf` — a second-order cone program (`.opf`)

`min t` s.t. `||(x0,x1)||_2 <= t`, `x0 + x1 = 1`, `x0 - x1 = 0`.

```sh
./out/primal lp_examples/socp_robust.opf
```
```
status  : OPTIMAL   prosta=PRIM_AND_DUAL_FEAS  solsta=OPTIMAL
obj     : primal = 0.7071067812   dual = 0.7071067811
x[t] = 0.7071067812
x[x0] = 0.5
x[x1] = 0.5
```

### `exp_power.opf` — an exponential-cone program (`.opf`)

`min t` s.t. `(t,u,v)` in the primal exponential cone (`t >= u e^(v/u)`),
`u = 1`, `v = 1`.

```sh
./out/primal lp_examples/exp_power.opf
```
```
status  : OPTIMAL   prosta=PRIM_AND_DUAL_FEAS  solsta=OPTIMAL
obj     : primal = 2.718281828   dual = 2.718281828
x[t] = 2.718281828
x[u] = 1
x[v] = 1
```

### `sdp_lmi.cbf` — a semidefinite program (`.cbf`)

`min x0 + x1` s.t. `X = [[x0, 1], [1, x1]] >= 0`; by AM-GM the optimum is `2`.

```sh
./out/primal lp_examples/sdp_lmi.cbf
```
```
status  : OPTIMAL   prosta=PRIM_AND_DUAL_FEAS  solsta=OPTIMAL
obj     : primal = 1.999999997   dual = 1.999999991
x[0] = 1.000000001
x[1] = 0.9999999958
barx[0] =        # the PSD matrix X
   1 1
   1 1
```

## Larger inputs

The `bench/instances/` directory has real MPS files (NETLIB and generated):

```sh
./out/primal bench/instances/afiro.mps --brief
./out/primal bench/instances/lp_400x200.mps -o /tmp/sol.txt
```
