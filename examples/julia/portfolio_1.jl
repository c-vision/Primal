#!/usr/bin/env julia
# portfolio_1.jl — Port of c_examples/mosek_comparison/portfolio_1.c
# Simple Markowitz QP.

using Primal
using Primal: Model   # not exported: JuMP exports its own Model

m = Model(0, 0)
add_vars(m, 3)
add_cons(m, 1)

# min 0.5 * x'Qx - mu'x  s.t. sum(x) = 1, x >= 0
# Q = [[0.04, 0.02, 0.0], [0.02, 0.09, 0.0], [0.0, 0.0, 0.16]]
# mu = [0.10, 0.15, 0.20]

for j in 0:2
    var_bounds(m, j, BK[:LO]; lo=0.0, up=Inf)
end

obj(m, [0, 1, 2], [-0.10, -0.15, -0.20])

qobj(m, [(0, 0, 0.08), (0, 1, 0.04), (1, 1, 0.18), (2, 2, 0.32)])

a_row(m, 0, [0, 1, 2], [1.0, 1.0, 1.0])
con_bounds(m, 0, BK[:FX]; lo=1.0, up=1.0)

r = solve(m)

ok = r.solsta == SOLSTA[:OPTIMAL]
println(ok ? "OK (portfolio_1 solved, obj=$(r.objective))" : "FAIL: x=$(r.x), obj=$(r.objective), solsta=$(r.solsta)")
free(m)
exit(ok ? 0 : 1)
