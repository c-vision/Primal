#!/usr/bin/env julia
# max_volume_cuboid.jl — the largest axis-aligned cuboid inscribed in a
# polyhedron {x : a_i'x <= b_i} (MOSEK Modeling Cookbook, geometric programming):
#
#   max prod_j (2 r_j)  s.t.  a_i'c + |a_i|'r <= b_i,  r >= 0
#
# With n = 2 the product is a geometric mean, carried by the power cone
# (r0, r1, t) in POW^{1/2}; the volume is (2t)^2. Hand optima: unit square ->
# 1.00, right triangle {x, y >= 0, x + y <= 1} -> 0.25.
#
#   julia max_volume_cuboid.jl [square|triangle]   (default square)
#
# Port of examples/c/max_volume_cuboid.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# each face is a0*x0 + a1*x1 <= b
const SQ = [-1.0 0 0; 0 -1 0; 1 0 1; 0 1 1]
const TRI = [-1.0 0 0; 0 -1 0; 1 1 1]

# Solve the max-volume cuboid GP on one polyhedron and check its volume.
function solve_case(name, F, want)
    nf = size(F, 1)
    m = Model(nf, 5)
    ok = false
    try
        var_bounds(m, 0, BK_FR; lo=-Inf, up=Inf)    # 0=c0  1=c1  2=r0  3=r1  4=t
        var_bounds(m, 1, BK_FR; lo=-Inf, up=Inf)
        var_bounds(m, 2, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 3, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 4, BK_FR; lo=-Inf, up=Inf)
        cj(m, 4, -1.0)                              # maximize t
        for i in 1:nf
            sub, val = Int[], Float64[]
            F[i, 1] != 0.0 && (push!(sub, 0); push!(val, F[i, 1]))
            F[i, 2] != 0.0 && (push!(sub, 1); push!(val, F[i, 2]))
            F[i, 1] != 0.0 && (push!(sub, 2); push!(val, abs(F[i, 1])))
            F[i, 2] != 0.0 && (push!(sub, 3); push!(val, abs(F[i, 2])))
            a_row(m, i - 1, sub, val)
            con_bounds(m, i - 1, BK_UP; lo=-Inf, up=F[i, 3])
        end
        cone(m, CT_PPOW, [2, 3, 4]; param=0.5)      # sqrt(r0*r1) >= |t|
        r = solve(m)
        ok = r.rc == 0
        if ok
            x = r.x
            vol = 4.0 * x[5] * x[5]
            ok = abs(vol - want) < 1e-5
            @printf("%-8s c=(%.4f,%.4f) r=(%.4f,%.4f) volume=%.6f (expected %.4f) %s\n",
                    name, x[1], x[2], x[3], x[4], vol, want, ok ? "OK" : "FAIL")
        else
            @printf("%-8s rc=%d FAIL\n", name, r.rc)
        end
    finally
        free(m)
    end
    return ok
end

function main()
    which = length(ARGS) > 0 ? ARGS[1] : "square"
    ok = startswith(which, "t") ? solve_case("triangle", TRI, 0.25) : solve_case("square", SQ, 1.00)
    return ok ? 0 : 1
end

exit(main())
