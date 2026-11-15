#!/usr/bin/env julia
# examples.jl — Run every program in examples/julia and require exit code 0.
#
# These programs are the only cover of the low-level API in src/Model.jl: the
# JuMP and MathOptInterface suites in this directory go through
# src/MOI_wrapper.jl instead.  They live in the repository and not in the
# package tarball, so test/runtests.jl includes this file and the run skips
# itself when ../examples/julia is absent (installed package).
#
# Usage (from the checkout):
#     julia --project=julia julia/test/examples.jl
#     PRIMALSOLVER_COVERAGE=1 julia --project=julia julia/test/examples.jl
#
# PRIMALSOLVER_COVERAGE=1 makes each example write .cov files next to the
# sources, which is what the CI job aggregates; it is off by default because a
# local run should not litter the tree.
#
# Only Base is used here on purpose: this file also runs inside `Pkg.test`,
# where the environment contains the test dependencies of julia/test/Project.toml
# and nothing else.

const PKG_DIR = dirname(@__DIR__)                       # .../PrimalSolver/julia
const EXAMPLES_DIR = joinpath(dirname(PKG_DIR), "examples", "julia")

"""Run every `examples/julia/*.jl` against this package; return the failures."""
function run_examples(; coverage::Bool = get(ENV, "PRIMALSOLVER_COVERAGE", "") == "1")
    if !isdir(EXAMPLES_DIR)
        println("examples/julia not found at $EXAMPLES_DIR: skipping the low-level API programs")
        return String[]
    end
    covflag = coverage ? `--code-coverage=user` : ``
    files = sort(filter(f -> endswith(f, ".jl"), readdir(EXAMPLES_DIR)))
    failed = String[]
    for f in files
        cmd = `$(Base.julia_cmd()) $covflag --startup-file=no --project=$PKG_DIR $(joinpath(EXAMPLES_DIR, f))`
        # `Pkg.test` exports JULIA_LOAD_PATH="@:<temp test env>", which would
        # hide the standard library (`using Printf` is in every example): every
        # program would fail for a reason that has nothing to do with the
        # solver.  Run them with the load path a user has.
        cmd = addenv(cmd, "JULIA_LOAD_PATH" => "@:@stdlib")
        out = IOBuffer()
        if success(pipeline(cmd; stdout = out, stderr = out))
            println("PASS: ", f)
        else
            println("FAIL: ", f)
            print(String(take!(out)))
            push!(failed, f)
        end
    end
    println("examples/julia: ", length(files) - length(failed), " passed, ", length(failed), " failed")
    return failed
end

if abspath(PROGRAM_FILE) == @__FILE__
    failed = run_examples()
    if isempty(failed)
        println("All examples passed!")
        exit(0)
    end
    println("Failed: ", join(failed, ", "))
    exit(1)
end
