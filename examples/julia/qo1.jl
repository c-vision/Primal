#!/usr/bin/env julia
# qo1.jl — Port of c_examples/mosek_comparison/qo1.c
#
# min x^2 + y^2  s.t. x+y >= 1, x,y >= 0
# Expected solution: x = 0.5, y = 0.5, obj = 0.5

include("../src/Primal.jl")
using .Primal

m = Model(0, 0)
add_vars(m, 2)
add_cons(m, 1)

for j in 0:1
    var_bounds(m, j, BK[:LO]; lo=0.0, up=Inf)
end

# Quadratic objective: min x^2 + y^2  (Q = diag(2): 0.5*x'Qx = x^2+y^2)
qobj(m, [(0, 0, 2.0), (1, 1, 2.0)])
sense(m, SENSE[:MIN])

a_row(m, 0, [0, 1], [1.0, 1.0])
con_bounds(m, 0, BK[:LO]; lo=1.0, up=Inf)

r = solve(m)

ok = length(r.x) == 2 &&
     abs(r.x[1] - 0.5) < 1e-6 && abs(r.x[2] - 0.5) < 1e-6 &&
     abs(r.objective - 0.5) < 1e-6
println(ok ? "OK (expected x=0.5, y=0.5, obj=0.5)" : "FAIL: x=$(r.x), obj=$(r.objective)")
free(m)
exit(ok ? 0 : 1)
