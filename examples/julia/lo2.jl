#!/usr/bin/env julia
# lo2.jl — Port of c_examples/mosek_comparison/lo2.c
#
# max 3x0 + x1 + 5x2 + x3
# s.t. 3x0 + x1 + 2x2       = 30
#      2x0 + x1 + 3x2 + x3 <= 25
#       x0 + 2x1 +  x2 + 2x3 >= 15
#      0 <= x0, 0 <= x1 <= 10, x2, x3 >= 0

include("../src/Primal.jl")
using .Primal

m = Model(0, 0)
add_vars(m, 4)
add_cons(m, 3)

# Objective: max 3x0 + x1 + 5x2 + x3
obj(m, [0, 1, 2, 3], [3.0, 1.0, 5.0, 1.0])
sense(m, SENSE[:MAX])

# Variable bounds
var_bounds(m, 0, BK[:LO]; lo=0.0, up=Inf)
var_bounds(m, 1, BK[:RA]; lo=0.0, up=10.0)
var_bounds(m, 2, BK[:LO]; lo=0.0, up=Inf)
var_bounds(m, 3, BK[:LO]; lo=0.0, up=Inf)

# Constraint 0: 3x0 + x1 + 2x2 = 30
a_row(m, 0, [0, 1, 2], [3.0, 1.0, 2.0])
con_bounds(m, 0, BK[:FX]; lo=30.0, up=30.0)

# Constraint 1: 2x0 + x1 + 3x2 + x3 <= 25
a_row(m, 1, [0, 1, 2, 3], [2.0, 1.0, 3.0, 1.0])
con_bounds(m, 1, BK[:UP]; lo=-Inf, up=25.0)

# Constraint 2: x0 + 2x1 + x2 + 2x3 >= 15
a_row(m, 2, [0, 1, 2, 3], [1.0, 2.0, 1.0, 2.0])
con_bounds(m, 2, BK[:LO]; lo=15.0, up=Inf)

r = solve(m)

ok = length(r.x) == 4 && r.solsta == SOLSTA[:OPTIMAL]
println(ok ? "OK (lo2 solved, obj=$(r.objective))" : "FAIL: x=$(r.x), obj=$(r.objective), solsta=$(r.solsta)")
free(m)
exit(ok ? 0 : 1)
