#!/usr/bin/env julia
# acc1_max.jl — the MOSEK "acc1.c" tutorial (affine conic constraints), the
# example of the OptimalCNC/MOSEK_EXAMPLE CMake template (not the Julia acc1):
#
#   maximize c'x, c = (2, 3, -1)  s.t.  sum(x) = 1,  gamma >= ||G x + h||_2
#
# with gamma = 0.03, G = [1.5 0.1; 0.3 2.1] (on x0,x1 / x0,x2), h = (0, 0.1).
# The AFE rows are spelled as variables z1, z2 plus a quadratic cone.
# Hand value: obj = 3.2805112648 at x = (-0.07838011, 1.12891290, -0.05053279).
# Port of examples/c/acc1_max.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    X0, X1, X2, Z1, Z2, GAMMA = 0, 1, 2, 3, 4, 5
    m = Model(0, 0)
    r = try
        add_vars(m, 6)
        add_cons(m, 3)
        sense(m, SENSE_MAX)
        for j in X0:Z2
            var_bounds(m, j, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(m, GAMMA, BK_FX; lo=0.03, up=0.03)
        a_row(m, 0, [X0, X1, X2], [1.0, 1.0, 1.0])         # sum(x) = 1
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        a_row(m, 1, [Z1, X0, X1], [1.0, -1.5, -0.1])       # z1 = 1.5 x0 + 0.1 x1
        con_bounds(m, 1, BK_FX; lo=0.0, up=0.0)
        a_row(m, 2, [Z2, X0, X2], [1.0, -0.3, -2.1])       # z2 = 0.3 x0 + 2.1 x2 + 0.1
        con_bounds(m, 2, BK_FX; lo=0.1, up=0.1)
        cone(m, CT_QUAD, [GAMMA, Z1, Z2])                   # gamma >= ||(z1, z2)||
        cj(m, X0, 2.0); cj(m, X1, 3.0); cj(m, X2, -1.0)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    x, obj = zeros(8), 0.0
    if ok
        x, obj = r.x, r.objective
    end
    z1 = 1.5 * x[X0+1] + 0.1 * x[X1+1]
    z2 = 0.3 * x[X0+1] + 2.1 * x[X2+1] + 0.1
    nrm, s = sqrt(z1 * z1 + z2 * z2), x[X0+1] + x[X1+1] + x[X2+1]
    good = ok && abs(obj - 3.2805112648) < 1e-6 && abs(x[X0+1] + 0.0783801115) < 1e-6 &&
           abs(x[X1+1] - 1.1289128998) < 1e-6 && abs(x[X2+1] + 0.0505327883) < 1e-6 &&
           nrm <= 0.03 + 1e-9 && abs(s - 1.0) < 1e-9
    @printf("acc1_max  obj=%.10f (expected 3.2805112648)  x=(%.8f,%.8f,%.8f)\n", obj, x[X0+1], x[X1+1], x[X2+1])
    @printf("  ||Gx+h||=%.10f (<=0.03)  sum=%.10f\n", nrm, s)
    @printf("%s\n", good ? "OK" : "FAIL")
    return good ? 0 : 1
end

exit(main())
