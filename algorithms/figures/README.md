# Figures of the teaching guide

Every figure here is produced by a program in this directory, from the solver or from a
closed form that the program checks against the solver, except the small static `lorentz-section.svg`.

| File | Kind | Made by | What is checked before it is written |
|---|---|---|---|
| `simplex-pivots.svg`, `simplex-walk.gif` | analytic / animation | `make_figures.c`, `make_animations.c` | the tableau run ends at the optimum `PRIMAL_optimize` reports |
| `lp-central-path.svg`, `lp-central-path-trace.csv`, `ipm-approach.gif` | measured | same | the iterates come from `ipm_solve_std`; the last one equals the API optimum |
| `qp-contours.svg` | analytic | `make_figures.c` | KKT conditions of the QP, optimum and multiplier against the API |
| `supporting-hyperplane.svg` | analytic | `make_figures.c` | the optimal vertex and its multipliers |
| `presolve-reduction.svg` | computed | `make_figures.c` | the reductions of `lp_presolve` and the lifted multipliers |
| `cholesky-ordering.svg`, `ordering-fill.csv` | measured | `make_figures.c` | the factors of `spchol_factor` and `spchol_factor_ord` solve the same system |
| `elliptope-slices.svg` | analytic | `make_figures.c` | the minimum on each slice equals the closed form, via `PRIMAL_optimize` |
| `exp-cone-section.svg`, `power-cone-section.svg` | analytic | `make_figures.c` | membership of points on both sides of every curve with `expcone_barrier`; the optimum against the API |
| `scaling-equilibration.svg` | computed | `make_figures.c` | every scaled entry equals `r_i d_j a_ij` exactly and every factor is a power of two |
| `dispatch.svg` | static | `make_figures.c` | none: a diagram read from `opt_routes`, checked by the three traces of the architecture chapter |
| `lorentz-section.svg` | analytic | by hand (static) | none |

The generators exit with a non-zero status if a check fails.

## Regenerating

Run these commands from the repository root. The presolve figure requires
the correction in `2c48c87` for reduced rows stored with their sign reversed.

```sh
make lib
mkdir -p out/figures
cc -std=c99 -O2 -I. -o out/figures/make_figures \
  algorithms/figures/make_figures.c out/libprimal.a -lm -lpthread
out/figures/make_figures "$(git rev-parse --short HEAD)" out/figures
```

The animated GIFs need two steps, and no Python: the generator writes SVG frames and a list of
display times, a shell script renders them with `rsvg-convert`, converts each to a
single-image GIF with `sips` (macOS) and joins them with `gifjoin`, which only rewrites the container
and copies the compressed image data.

```sh
mkdir -p out/figures/frames
cc -std=c99 -O2 -I. -o out/figures/make_animations \
  algorithms/figures/make_animations.c out/libprimal.a -lm -lpthread
cc -std=c99 -O2 -o out/figures/gifjoin algorithms/figures/gifjoin.c
out/figures/make_animations "$(git rev-parse --short HEAD)" out/figures/frames
sh algorithms/figures/make_gifs.sh out/figures/frames out/figures out/figures/gifjoin
```
