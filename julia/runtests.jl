#!/usr/bin/env julia
# runtests.jl — Execute all Julia examples and verify exit code 0.
# Usage (from the PrimalSolver checkout):
#     julia --project=julia julia/runtests.jl
# Each example runs in its own process against this package (--project=julia/).

using Printf

const PKG_DIR = @__DIR__
const EXAMPLES_DIR = joinpath(dirname(PKG_DIR), "examples", "julia")

example_files = sort(filter(f -> endswith(f, ".jl"), readdir(EXAMPLES_DIR)))

passed = Ref{Int}(0)
failed = String[]

for f in example_files
    path = joinpath(EXAMPLES_DIR, f)
    cmd = `$(Base.julia_cmd()) --project=$PKG_DIR $path`
    ok = success(pipeline(cmd; stdout = devnull, stderr = devnull))
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
