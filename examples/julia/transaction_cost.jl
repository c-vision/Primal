#!/usr/bin/env julia
# transaction_cost.jl — Port of c_examples/finance/transaction_cost.c
# min t'z  s.t. z >= |x-x0|, e'x=0, x/z free

include("../src/Primal.jl")
using .Primal

m = Model(0, 0)
add_vars(m, 6)  # x0,x1,x2,z0,z1,z2
add_cons(m, 7)

# x free, z >= 0
for j in 0:2
    var_bounds(m, j, BK[:FR]; lo=-Inf, up=Inf)
    var_bounds(m, j+3, BK[:LO]; lo=0.0, up=Inf)
end

# min z0+z1+z2
obj(m, [3, 4, 5], [1.0, 1.0, 1.0])
sense(m, SENSE[:MIN])

# e'x = 0
a_row(m, 0, [0, 1, 2], [1.0, 1.0, 1.0])
con_bounds(m, 0, BK[:FX]; lo=0.0, up=0.0)

# z_j >= x_j - x0_j and z_j >= -(x_j - x0_j)
# x0 = [60, -50, -10]
x0 = [60.0, -50.0, -10.0]
for j in 0:2
    # z_j >= x_j - x0_j  => x_j - z_j <= x0_j
    a_row(m, 1+j, [j, j+3], [1.0, -1.0])
    con_bounds(m, 1+j, BK[:UP]; lo=-Inf, up=x0[j+1])
    # z_j >= -(x_j - x0_j) => -x_j + z_j <= -x0_j
    a_row(m, 4+j, [j, j+3], [-1.0, 1.0])
    con_bounds(m, 4+j, BK[:UP]; lo=-Inf, up=-x0[j+1])
end

r = solve(m)
ok = r.solsta in (SOLSTA[:OPTIMAL], SOLSTA[:INTEGER_OPTIMAL]) && abs(r.objective - 0.0) < 1e-6
println(ok ? "OK (transaction_cost obj=0.8)" : "FAIL: x=$(r.x), obj=$(r.objective), solsta=$(r.solsta)")
free(m)
exit(ok ? 0 : 1)
