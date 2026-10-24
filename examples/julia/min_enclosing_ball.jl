#!/usr/bin/env julia
# min_enclosing_ball.jl — smallest ball enclosing a finite point set
# (MOSEK/Tutorials, minimum-ellipsoid, primal model):
#
#   minimize r0  s.t.  r0 >= ||p0 - p_i||_2  (one quadratic cone per point)
#
# Hand instances: 2 points at distance d -> r* = d/2; equilateral triangle of
# side a -> r* = a/sqrt(3). Port of examples/c/min_enclosing_ball.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# Smallest enclosing ball of the k points in P (k x 2); returns (ok, radius, center).
# Variables: px, py, r, then a per-point difference (dx_i, dy_i).
function min_ball(k, P)
    PX, PY, R, D = 0, 1, 2, 3
    m = Model(0, 0)
    r = try
        add_vars(m, 3 + 2k)
        add_cons(m, 2k)
        var_bounds(m, PX, BK_FR; lo=-Inf, up=Inf)
        var_bounds(m, PY, BK_FR; lo=-Inf, up=Inf)
        var_bounds(m, R, BK_LO; lo=0.0, up=Inf)
        for i in 0:k-1
            var_bounds(m, D + 2i, BK_FR; lo=-Inf, up=Inf)
            var_bounds(m, D + 2i + 1, BK_FR; lo=-Inf, up=Inf)
            # dx_i - px = -P[i][0] ; dy_i - py = -P[i][1]
            a_row(m, 2i, [D + 2i, PX], [1.0, -1.0])
            con_bounds(m, 2i, BK_FX; lo=-P[i+1, 1], up=-P[i+1, 1])
            a_row(m, 2i + 1, [D + 2i + 1, PY], [1.0, -1.0])
            con_bounds(m, 2i + 1, BK_FX; lo=-P[i+1, 2], up=-P[i+1, 2])
        end
        for i in 0:k-1                              # r >= ||p0 - p_i||
            cone(m, CT_QUAD, [R, D + 2i, D + 2i + 1])
        end
        cj(m, R, 1.0)
        sense(m, SENSE_MIN)
        solve(m)
    finally
        free(m)
    end
    r.rc == 0 || return false, 0.0, [0.0, 0.0]
    return true, r.x[R+1], [r.x[PX+1], r.x[PY+1]]
end

function main()
    P2 = [0.0 0.0; 2.0 0.0; 0.0 0.0]
    P3 = [0.0 0.0; 2.0 0.0; 1.0 1.7320508075688772]  # side 2
    ok2, r2, _ = min_ball(2, P2)
    ok3, r3, _ = min_ball(3, P3)
    want2 = 1.0                                     # d/2, d = 2
    want3 = 2.0 / sqrt(3.0)                         # a/sqrt(3), a = 2
    ok = ok2 && ok3 && abs(r2 - want2) < 1e-7 && abs(r3 - want3) < 1e-7
    @printf("min_enclosing_ball\n")
    @printf("  2 points (d=2):    r*=%.10f   expected d/2=%.10f\n", r2, want2)
    @printf("  triangle (a=2):    r*=%.10f   expected a/sqrt(3)=%.10f\n", r3, want3)
    @printf("%s\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
