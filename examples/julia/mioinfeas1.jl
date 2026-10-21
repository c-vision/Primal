#!/usr/bin/env julia
# mioinfeas1.jl — MIP infeasible

include("../src/Primal.jl")
using .Primal

m = Model(0, 0)
add_vars(m, 1)
add_cons(m, 1)

obj(m, [0], [1.0])
sense(m, SENSE[:MIN])
var_bounds(m, 0, BK[:LO]; lo=0.0, up=Inf)
a_row(m, 0, [0], [1.0])
con_bounds(m, 0, BK[:FX]; lo=5.0, up=3.0)  # x=5 and x=3 -> infeasible
var_type(m, 0, VAR_TYPE[:INT])

r = solve(m)
ok = r.solsta != SOLSTA[:OPTIMAL]  # infeasible, no optimal
println(ok ? "OK (mioinfeas1 infeasible)" : "FAIL: x=$(r.x), obj=$(r.objective), solsta=$(r.solsta)")
free(m)
exit(ok ? 0 : 1)
