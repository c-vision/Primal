#!/usr/bin/env julia
# cqo1.jl — Port of c_examples/mosek_comparison/cqo1.c
#
# min  x3 + x4 + x5
# s.t. x0 + x1 + 2*x2 = 1
#      (x3, x0, x1) in QUAD_3   -> x3 >= sqrt(x0^2 + x1^2)
#      (x4, x5, x2) in RQUAD_3  -> 2*x4*x5 >= x2^2
#      x0,x1,x2 >= 0; x3,x4,x5 free
# Known optimum: obj = 1/sqrt(2) ≈ 0.7071

include("../src/Primal.jl")
using .Primal

m = Model(0, 0)
add_vars(m, 6)
add_cons(m, 1)

for j in 0:2
    var_bounds(m, j, BK[:LO]; lo=0.0, up=Inf)
end
for j in 3:5
    var_bounds(m, j, BK[:FR]; lo=-Inf, up=Inf)
end

obj(m, [3, 4, 5], [1.0, 1.0, 1.0])
sense(m, SENSE[:MIN])

a_row(m, 0, [0, 1, 2], [1.0, 1.0, 2.0])
con_bounds(m, 0, BK[:FX]; lo=1.0, up=1.0)

cone(m, CT[:QUAD], [3, 0, 1])
cone(m, CT[:RQUAD], [4, 5, 2])

r = solve(m)

ok = length(r.x) == 6 &&
     abs(r.objective - 1/sqrt(2)) < 1e-4
println(ok ? "OK (expected obj = 1/sqrt(2) ≈ 0.7071)" :
        "FAIL: x=$(r.x), obj=$(r.objective)")
free(m)
exit(ok ? 0 : 1)
