#!/usr/bin/env julia
# runtests.jl — Execute all Julia examples and verify exit code 0.
# Usage: julia runtests.jl  (from julia/Primal/)

using Printf

const EXAMPLES_DIR = joinpath(@__DIR__, "examples")

example_files = sort(filter(f -> endswith(f, ".jl"), readdir(EXAMPLES_DIR)))

passed = Ref{Int}(0)
failed = String[]

for f in example_files
    path = joinpath(EXAMPLES_DIR, f)
    cmd = `julia $path`
    ok = success(cmd)
    if ok
        passed[] += 1
        @printf("PASS: %s\n", f)
    else
        push!(failed, f)
        @printf("FAIL: %s\n", f)
    end
end

@printf("\n========================================\n")
@printf("PASS: %d   FAIL: %d\n", passed[], length(failed))
if !isempty(failed)
    @printf("Failed: %s\n", join(failed, ", "))
    exit(1)
else
    @printf("All examples passed!\n")
    exit(0)
end
