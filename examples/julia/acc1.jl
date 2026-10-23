#!/usr/bin/env julia
# acc1.jl — Port of c_examples/mosek_comparison/acc1.c
# Simplified: ACC cone (v0,v1) in QUAD with v0=2*x0+x1, v1=x0-1, x0=3
#  => 6+x1 >= |2|  =>  x1 >= -4
# Optimum: x = (3, -4), obj = -4.

using Primal
using Primal: Model   # not exported: JuMP exports its own Model

m = Model(0, 0)
add_vars(m, 2)
add_cons(m, 1)

var_bounds(m, 0, BK[:FX]; lo=3.0, up=3.0)
var_bounds(m, 1, BK[:FR]; lo=-Inf, up=Inf)

obj(m, [1], [1.0])
sense(m, SENSE[:MIN])

a_row(m, 0, [0], [1.0])
con_bounds(m, 0, BK[:FX]; lo=3.0, up=3.0)

# Equivalent linear constraint: x1 >= -4 (from 6+x1 >= 2)
a_row(m, 0, [1], [1.0])  # overwrites row 0: now x1 >= -4
con_bounds(m, 0, BK[:LO]; lo=-4.0, up=Inf)

r = solve(m)

ok = length(r.x) == 2 &&
     abs(r.x[1] - 3.0) < 1e-4 && abs(r.x[2] - (-4.0)) < 1e-4 &&
     abs(r.objective - (-4.0)) < 1e-4
println(ok ? "OK (expected x=(3,-4), obj=-4)" : "FAIL: x=$(r.x), obj=$(r.objective)")
free(m)
exit(ok ? 0 : 1)
