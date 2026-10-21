#!/usr/bin/env julia
# hello.jl — Port of c_examples/mosek_comparison/hello.c
#
# Simplest possible task: a free variable, min x, x ∈ [2, 4]  ->  x = 2, obj = 2

include("../src/Primal.jl")
using .Primal

m = Model(0, 0)
add_vars(m, 1)
cj(m, 0, 1.0)
var_bounds(m, 0, BK[:RA]; lo=2.0, up=4.0)
sense(m, SENSE[:MIN])
r = solve(m)

ok = length(r.x) == 1 && abs(r.x[1] - 2.0) < 1e-6 && abs(r.objective - 2.0) < 1e-6
println(ok ? "OK (expected x = 2, obj = 2)" : "FAIL: x=$(r.x), obj=$(r.objective)")
free(m)
exit(ok ? 0 : 1)
