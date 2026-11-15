# runtests.jl — PrimalSolver.jl test suite: `Pkg.test("PrimalSolver")`, or
#
#     julia --project=julia/test julia/test/runtests.jl

using Test

include("jump.jl")          # JuMP models with known answers
include("MOI_wrapper.jl")   # MathOptInterface conformance (MOI.Test)

# The low-level API (src/Model.jl) is covered by the programs in
# examples/julia, which are part of the repository and not of the package
# tarball: run them when a checkout provides them, skip them otherwise.
# Nothing has to be configured — after a checkout the directory is there.
include("examples.jl")
failed = run_examples()
isempty(failed) || println("failing examples: ", join(failed, ", "))
@test isempty(failed)
