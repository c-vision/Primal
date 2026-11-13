# runtests.jl — PrimalSolver.jl test suite: `Pkg.test("PrimalSolver")`, or
#
#     julia --project=julia/test julia/test/runtests.jl

using Test

include("jump.jl")          # JuMP models with known answers
include("MOI_wrapper.jl")   # MathOptInterface conformance (MOI.Test)
