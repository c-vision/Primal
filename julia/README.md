# PrimalSolver.jl

Julia bindings for **PrimalSolver**, the C99 solver of this repository
(LP/QP/SOCP/SDP/exp-power/MIP), exposed through
[MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) so that it
can be used from [JuMP](https://jump.dev).

The package is the `julia/` subdirectory of the parent project; the solver
itself is the C library documented in the
[top-level README](../README.md). The public library `libprimal` is **not**
compiled by this package: it comes prebuilt from
[`PrimalSolver_jll`](https://github.com/JuliaBinaryWrappers/PrimalSolver_jll.jl),
so installing the package needs no C compiler.

## Install

```julia
using Pkg
Pkg.add(["JuMP", "PrimalSolver"])
```

```julia
using JuMP, PrimalSolver
model = Model(PrimalSolver.Optimizer)
```

## Documentation

- Usage, modelling surface and solver parameters: [top-level README](../README.md#julia).
- The C API that the bindings mirror one-to-one: `primal.h` in the parent
  project, and the [Python guide](../examples/python/README.md) for the same
  model-building pattern in another language.
- Full example programs: [`../examples/julia/`](../examples/julia/) (171 files,
  one per C example).

## Working on the package from a checkout

The Julia layer has two pieces: `src/MOI_wrapper.jl` (the JuMP/MOI model
front-end) and `src/Model.jl` plus the generated `src/bindings.jl` (the 1:1
`ccall` surface over `libprimal`).

```sh
make shared                       # out/libprimal.dylib (or .so / .dll)
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
julia --project=julia -e 'using Pkg; Pkg.test()'         # MOI + JuMP + examples
julia --project=julia julia/test/examples.jl             # only the examples
```

`Pkg.test()` runs the MathOptInterface and JuMP suites and then the 171
programs in `../examples/julia`, which are what covers the low-level API. The
examples are part of the repository and are not shipped in the package
tarball, so from an installed package that block skips itself.

### Point the package at a local build

A development checkout must be told where the shared library is; the package
carries no lookup path of its own (no `deps/build.jl`, no environment
variables). The registered JLL reads the standard JLLWrappers preference, which
is a two-line file in `julia/LocalPreferences.toml` (already gitignored):

```toml
[PrimalSolver_jll]
libprimal_path = "/abs/path/to/Primal/out/libprimal.dylib"
```

Equivalently, from Julia — the call writes the same file:

```julia
using Preferences, PrimalSolver_jll
set_preferences!(PrimalSolver_jll, "libprimal_path" => "/abs/path/to/libprimal.dylib";
                 force = true)
```

The preference is read when `PrimalSolver_jll` loads, i.e. before the first
`using PrimalSolver`. Remove it (or delete the file) to go back to the
registered artifact.

`src/bindings.jl` is generated: after changing `primal.h` run
`julia julia/tools/gen_bindings.jl` (never edit that file by hand).

## Tests and continuous integration

`.github/workflows/julia.yml` runs `Pkg.test()` on the supported Julia
versions, on Linux. `src/Model.jl` is covered by the example programs, which the
test suite runs with `Base.active_project()`, so `Pkg.test()` covers the
low-level API as well as the JuMP/MathOptInterface layer. The workflow uploads
nothing: there is no coverage service and no badge, by decision (a step that
fails with `Token required` while staying green proves nothing). For a local
report, run the examples with `PRIMALSOLVER_COVERAGE=1` and
`--code-coverage=user`, as documented in `julia/test/examples.jl`.

## License

Apache-2.0, the same as the parent project (see [`LICENSE`](LICENSE)).
