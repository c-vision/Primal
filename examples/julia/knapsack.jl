#!/usr/bin/env julia
# knapsack.jl — Simple MIP knapsack

include("../src/Primal.jl")
using .Primal

m = Model(0, 0)
add_vars(m, 4)
add_cons(m, 1)

obj(m, [0, 1, 2, 3], [10.0, 8.0, 6.0, 4.0])
sense(m, SENSE[:MAX])
for j in 0:3
    var_bounds(m, j, BK[:LO]; lo=0.0, up=1.0)
    var_type(m, j, VAR_TYPE[:INT_BIN])
end
a_row(m, 0, [0, 1, 2, 3], [5.0, 4.0, 3.0, 2.0])
con_bounds(m, 0, BK[:UP]; lo=-Inf, up=8.0)

r = solve(m)
ok = r.solsta == SOLSTA[:INTEGER_OPTIMAL]
println(ok ? "OK (knapsack solved, obj=$(r.objective))" : "FAIL: x=$(r.x), obj=$(r.objective)")
free(m)
exit(ok ? 0 : 1)
