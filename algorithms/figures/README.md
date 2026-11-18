# Figures of the teaching guide

Every figure here is produced by a program in this directory, from the solver or from a
closed form that the program checks against the solver, except the small static `lorentz_section.svg`.

| File | Kind | Made by | What is checked before it is written |
|---|---|---|---|
| `simplex_pivots.svg`, `simplex_walk.gif` | analytic / animation | `make_figures.c`, `make_animations.c` | the tableau run ends at the optimum `PRIMAL_optimize` reports |
| `lp_central_path.svg`, `lp_central_path_trace.csv`, `ipm_approach.gif` | measured | same | the iterates come from `ipm_solve_std`; the last one equals the API optimum |
| `central_path.gif` | analytic animation | `make_animations.c` | each barrier minimizer is interior and satisfies stationarity |
| `exponential_sections.gif` | analytic animation | `make_animations.c` | `expcone_barrier` accepts and rejects samples on the appropriate sides of every section |
| `branch_and_bound.gif` | worked-example animation | `make_animations.c` | enumeration of the feasible integer points recovers the illustrated objective bound |
| `qp_contours.svg` | analytic | `make_figures.c` | KKT conditions of the QP, optimum and multiplier against the API |
| `supporting_hyperplane.svg` | analytic | `make_figures.c` | the optimal vertex and its multipliers |
| `presolve_reduction.svg` | computed | `make_figures.c` | the reductions of `lp_presolve` and the lifted multipliers |
| `cholesky_ordering.svg`, `ordering_fill.csv` | measured | `make_figures.c` | the factors of `spchol_factor` and `spchol_factor_ord` solve the same system |
| `elliptope_slices.svg` | analytic | `make_figures.c` | the minimum on each slice equals the closed form, via `PRIMAL_optimize` |
| `exp_cone_section.svg`, `power_cone_section.svg` | analytic | `make_figures.c` | membership of points on both sides of every curve with `expcone_barrier`; the optimum against the API |
| `scaling_equilibration.svg` | computed | `make_figures.c` | every scaled entry equals `r_i d_j a_ij` exactly and every factor is a power of two |
| `dispatch.svg` | static | `make_figures.c` | none: a diagram read from `opt_routes`, checked by the three traces of the architecture chapter |
| `lorentz_section.svg` | analytic | by hand (static) | none |

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
