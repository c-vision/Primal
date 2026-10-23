#!/usr/bin/env julia
# mi1.jl — Port of c_examples/mosek_comparison/mi1.c
#
# max x1 + 0.64*x2
# s.t. 50*x1 + 31*x2 <= 250
#      3*x1 - 2*x2 >= -4
#      x1, x2 integer >= 0
# Expected solution: x1 = 5, x2 = 0, obj = 5.0

using Primal
using Primal: Model   # not exported: JuMP exports its own Model

m = Model(0, 0)
add_cons(m, 2)
add_vars(m, 2)

for j in 0:1
    var_bounds(m, j, BK[:LO]; lo=0.0, up=Inf)
end

obj(m, [0, 1], [1.0, 0.64])
sense(m, SENSE[:MAX])

a_col(m, 0, [0, 1], [50.0, 3.0])
a_col(m, 1, [0, 1], [31.0, -2.0])

con_bounds(m, 0, BK[:UP]; lo=-Inf, up=250.0)
con_bounds(m, 1, BK[:LO]; lo=-4.0, up=Inf)

var_type(m, 0, VAR_TYPE[:INT])
var_type(m, 1, VAR_TYPE[:INT])

r = solve(m)

ok = length(r.x) == 2 &&
     abs(r.x[1] - 5.0) < 1e-6 && abs(r.x[2] - 0.0) < 1e-6 &&
     abs(r.objective - 5.0) < 1e-6 && r.solsta == SOLSTA[:INTEGER_OPTIMAL]
println(ok ? "OK (expected x1=5, x2=0, obj=5.0)" : "FAIL: x=$(r.x), obj=$(r.objective), solsta=$(r.solsta)")
free(m)
exit(ok ? 0 : 1)
