#!/usr/bin/env julia
# solvebasis.jl — MOSEK port "solvebasis": write a basis to file, read it into
# a new task and solve from that basis.
#
#   1. build a known LP and solve it
#   2. derive and write the basis of the optimal solution (writebasis)
#   3. build a NEW task, read the basis (readbasis), solve with solvebasis
#
#   max 3x0 + x1 + 5x2 + x3  s.t.  3x0 + x1 + 2x2 + x3 = 30,
#       2x0 + x1 + 3x2 + x3 >= 15,  2x1 + x3 <= 9,  x >= 0,  x3 <= 5
#
# Optimum x* = (0, 0, 15, 0), pobj = 75. Port of examples/c/solvebasis.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model
const B = Primal.Bindings

const SK_BAS, SK_LOW, SK_UPR = 1, 3, 4             # PRIMALstakeye, as primal.h

# Build the reference basis LP.
function build(m)
    add_vars(m, 4)
    add_cons(m, 3)
    sense(m, SENSE_MAX)
    c = [3.0, 1, 5, 1]
    for j in 0:3
        cj(m, j, c[j+1])
    end
    for j in 0:3
        var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
    end
    var_bounds(m, 3, BK_RA; lo=0.0, up=5.0)
    a_row(m, 0, [0, 1, 2, 3], [3, 1, 2, 1]); con_bounds(m, 0, BK_FX; lo=30.0, up=30.0)
    a_row(m, 1, [0, 1, 2, 3], [2, 1, 3, 1]); con_bounds(m, 1, BK_LO; lo=15.0, up=Inf)
    a_row(m, 2, [1, 3], [2, 1]);             con_bounds(m, 2, BK_UP; lo=-Inf, up=9.0)
end

# Derive the basis from the solution: a variable is BAS strictly between its
# bounds, LOW/UPR at a bound; a row is BAS (basic slack) if inactive.
function basis_from_solution(m)
    nvar, ncon = numvar(m), numcon(m)
    x = zeros(nvar)
    B.getxx(m.task[], SOL_BAS, x)
    skx = zeros(Cint, nvar)
    for j in 0:nvar-1
        bk, bl, bu = getvarbound(m, j)
        if bk == BK_FX
            skx[j+1] = SK_BAS; continue
        end
        skx[j+1] = abs(x[j+1] - bl) < 1e-6 ? SK_LOW : abs(x[j+1] - bu) < 1e-6 ? SK_UPR : SK_BAS
    end
    skc = zeros(Cint, ncon)
    for i in 0:ncon-1
        s = 0.0
        for j in 0:nvar-1
            s += getaij(m, i, j) * x[j+1]
        end
        _, bl, bu = getconbound(m, i)
        skc[i+1] = abs(s - bl) < 1e-6 ? SK_LOW : abs(s - bu) < 1e-6 ? SK_UPR : SK_BAS
    end
    return skc, skx
end

function main()
    path = "/tmp/mc_solvebasis.bas"
    t1, t2 = Model(0, 0), Model(0, 0)
    try
        build(t1)                                   # 1. solve and derive the basis
        r1 = solve(t1, SOL_BAS)
        if r1.rc != 0
            @printf("optimize rc=%d\n", r1.rc)
            return 1
        end
        po1 = r1.objective
        @printf("phase 1: pobj = %.4f (expected 75)\n", po1)
        skc, skx = basis_from_solution(t1)

        rc = Int(B.putskc(t1.task[], SOL_BAS, skc)) # 2. write the basis
        rc |= Int(B.putskx(t1.task[], SOL_BAS, skx))
        if rc != 0
            @printf("putsk rc=%d\n", rc)
            return 1
        end
        rc = Int(B.writebasis(t1.task[], path))
        if rc != 0
            @printf("writebasis rc=%d\n", rc)
            return 1
        end
        @printf("phase 2: basis written to %s\n", path)

        build(t2)                                   # 3. new task: read the basis, solvebasis
        rc = Int(B.readbasis(t2.task[], path))
        if rc != 0
            @printf("readbasis rc=%d\n", rc)
            return 1
        end
        rc = Int(B.solvebasis(t2.task[]))
        if rc != 0
            @printf("solvebasis rc=%d\n", rc)
            return 1
        end
        po2, x2 = Ref{Cdouble}(0.0), zeros(4)
        B.getprimalobj(t2.task[], SOL_BAS, po2)
        B.getxx(t2.task[], SOL_BAS, x2)
        @printf("phase 3: solvebasis pobj = %.4f, x = (%.4f, %.4f, %.4f, %.4f)\n", po2[], x2[1], x2[2], x2[3], x2[4])
        ok = abs(po1 - 75.0) < 1e-6 && abs(po2[] - po1) < 1e-6 && abs(x2[1]) < 1e-6 && abs(x2[2]) < 1e-6 &&
             abs(x2[3] - 15.0) < 1e-5 && abs(x2[4]) < 1e-6
        @printf("%s (expected pobj=75, x=(0,0,15,0), basis restored)\n", ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(t1); free(t2)
    end
end

exit(main())
