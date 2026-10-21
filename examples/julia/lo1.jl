#!/usr/bin/env julia
# lo1.jl — Port of c_examples/mosek_comparison/lo1.c
#
# min 2x + 3y  s.t. x+y >= 1, x-y <= 0, x,y >= 0
# Expected solution: x = 0.5, y = 0.5, obj = 2.5

include("../src/Primal.jl")
using .Primal

m = Model(0, 0)
add_vars(m, 2)
add_cons(m, 2)

for j in 0:1
    var_bounds(m, j, BK[:LO]; lo=0.0, up=Inf)
end

obj(m, [0, 1], [2.0, 3.0])
sense(m, SENSE[:MIN])

a_row(m, 0, [0, 1], [1.0, 1.0])
con_bounds(m, 0, BK[:LO]; lo=1.0, up=Inf)

a_row(m, 1, [0, 1], [1.0, -1.0])
con_bounds(m, 1, BK[:UP]; lo=-Inf, up=0.0)

r = solve(m)

ok = length(r.x) == 2 &&
     abs(r.x[1] - 0.5) < 1e-6 && abs(r.x[2] - 0.5) < 1e-6 &&
     abs(r.objective - 2.5) < 1e-6 && r.solsta == SOLSTA[:OPTIMAL]
println(ok ? "OK (expected x=0.5, y=0.5, obj=2.5)" : "FAIL: x=$(r.x), obj=$(r.objective), solsta=$(r.solsta)")
free(m)
exit(ok ? 0 : 1)
