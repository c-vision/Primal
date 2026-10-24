#!/usr/bin/env julia
# qcqp_sdo_relaxation.jl — the SDO relaxation of a mixed-integer QCQP (MOSEK
# Cookbook 11.10). X = x x' is relaxed to Z = [X x; x' 1] PSD with
# diag(X) >= x, minimizing <P, X> + 2 q'x, and compared with the integer
# least-squares min |A x - b| over integer x (a QUAD cone).
#
# Hand instance: A = I, c = (2/5, 3/5). diag(X) >= x cuts x = c, the optimum is
# the integral x = (0, 1) with value -1/5; the integer least squares is
# |(0,1) - c| = sqrt(8)/5. Port of examples/c/qcqp_sdo_relaxation.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# The SDO relaxation; returns (rc, value).
function sdo_relax(c)
    X00, X01, X11, X0, X1, NV = 0, 1, 2, 3, 4, 5
    m = Model(8, NV)
    r = try
        sense(m, SENSE_MIN)
        barvars(m, [3])                             # Z : 3x3 PSD
        for v in 0:NV-1
            var_bounds(m, v, BK_FR; lo=-Inf, up=Inf)
        end
        cj(m, X00, 1.0); cj(m, X11, 1.0)            # <I,X>
        cj(m, X0, -2.0 * c[1]); cj(m, X1, -2.0 * c[2])   # 2 q'x, q = -c
        E00, E01, E11 = sparsesymmat(m, 3, [(0, 0, 1.0)]), sparsesymmat(m, 3, [(0, 1, 1.0)]), sparsesymmat(m, 3, [(1, 1, 1.0)])
        E02, E12, E22 = sparsesymmat(m, 3, [(0, 2, 1.0)]), sparsesymmat(m, 3, [(1, 2, 1.0)]), sparsesymmat(m, 3, [(2, 2, 1.0)])
        row = 0                                     # Z entries <- scalar variables (off-diagonal counted twice)
        for (v, E, coef) in ((X00, E00, 1.0), (X01, E01, 0.5), (X11, E11, 1.0), (X0, E02, 0.5), (X1, E12, 0.5))
            a_row(m, row, [v], [-1.0]); baraij(m, row, 0, [(E, coef)])
            con_bounds(m, row, BK_FX; lo=0, up=0); row += 1
        end
        baraij(m, row, 0, [(E22, 1.0)])             # Z22 = 1
        con_bounds(m, row, BK_FX; lo=1.0, up=1.0); row += 1
        a_row(m, row, [X00, X0], [1.0, -1.0])      # diag(X) >= x
        con_bounds(m, row, BK_LO; lo=0.0, up=Inf); row += 1
        a_row(m, row, [X11, X1], [1.0, -1.0])
        con_bounds(m, row, BK_LO; lo=0.0, up=Inf); row += 1
        solve(m)
    finally
        free(m)
    end
    return r.rc, (r.rc == 0 ? r.objective : 0.0)
end

# The integer least-squares min |x - c| over integer x, via (t, x - c) in QUAD.
function int_ls(c)
    X0, X1, TT, E0, E1 = 0, 1, 2, 3, 4
    m = Model(1, 3)
    r = try
        sense(m, SENSE_MIN)
        for j in 0:1
            var_bounds(m, j, BK_LO; lo=-100.0, up=Inf)
            var_type(m, j, VT_INT)
        end
        var_bounds(m, TT, BK_LO; lo=0.0, up=Inf)
        cj(m, TT, 1.0)
        add_vars(m, 2)                              # the cone members are auxiliary
        var_bounds(m, E0, BK_FR; lo=-Inf, up=Inf)
        var_bounds(m, E1, BK_FR; lo=-Inf, up=Inf)
        a_row(m, 0, [E0, X0], [1.0, -1.0])
        con_bounds(m, 0, BK_FX; lo=-c[1], up=-c[1])
        add_cons(m, 1)
        a_row(m, 1, [E1, X1], [1.0, -1.0])
        con_bounds(m, 1, BK_FX; lo=-c[2], up=-c[2])
        cone(m, CT_QUAD, [TT, E0, E1])              # t >= ||x - c||
        solve(m)
    finally
        free(m)
    end
    return r.rc, (r.rc == 0 ? r.objective : 0.0)
end

function main()
    c = [2.0 / 5.0, 3.0 / 5.0]
    r1, relax = sdo_relax(c)
    r2, ils = int_ls(c)
    expect_relax = -1.0 / 5.0                       # diag(X) >= x cuts x = c
    expect_ils = sqrt(8.0) / 5.0                    # sqrt(8/25)
    good = r1 == 0 && r2 == 0 && abs(relax - expect_relax) < 1e-6 && abs(ils - expect_ils) < 1e-6
    @printf("qcqp_sdo_relaxation: relax=%.8f (expected -1/5=%.8f)  int_ls=%.8f (expected sqrt8/5=%.8f)  %s\n",
            relax, expect_relax, ils, expect_ils, good ? "OK" : "FAIL")
    return good ? 0 : 1
end

exit(main())
