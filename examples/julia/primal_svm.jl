#!/usr/bin/env julia
# primal_svm.jl — soft-margin primal SVM (MOSEK Cookbook 11.2):
#
#   minimize t + c sum_i xi_i
#   s.t.  y_i (<X_i, w> - b) + xi_i >= 1,  xi >= 0,  (t, 1, w) in RQUAD (t >= ||w||^2/2)
#
# Two hand instances: A) the separable diamond (2,0),(0,2),(-2,0),(0,-2) with
# y = +1,+1,-1,-1, c = 1 -> optimum 1/4, w = (1/2, 1/2), b = 0;
# B) two opposite labels on the same feature -> optimum 2, w = 0, xi = (1, 1).
# Port of examples/c/primal_svm.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# Solve one SVM; returns (ok, obj, w, b).
function svm(nd, nim, X, y, c)
    T, Bv, W, XI = 0, 1, 2, 2 + nd
    ONE = 2 + nd + nim
    NV = 3 + nd + nim
    m = Model(0, 0)
    r = try
        add_vars(m, NV)
        add_cons(m, nim)
        sense(m, SENSE_MIN)
        var_bounds(m, T, BK_FR; lo=-Inf, up=Inf)
        var_bounds(m, Bv, BK_FR; lo=-Inf, up=Inf)
        for j in 0:nd-1
            var_bounds(m, W + j, BK_FR; lo=-Inf, up=Inf)
        end
        for i in 0:nim-1
            var_bounds(m, XI + i, BK_LO; lo=0.0, up=Inf)
        end
        var_bounds(m, ONE, BK_FX; lo=1.0, up=1.0)
        cj(m, T, 1.0)
        for i in 0:nim-1
            cj(m, XI + i, c)
        end
        cone(m, CT_RQUAD, [T; ONE; [W + j for j in 0:nd-1]])   # t >= ||w||^2 / 2
        for i in 0:nim-1                            # y_i(<X_i,w>-b) + xi_i >= 1
            sub, val = Int[], Float64[]
            for j in 0:nd-1
                X[i+1, j+1] != 0.0 && (push!(sub, W + j); push!(val, y[i+1] * X[i+1, j+1]))
            end
            push!(sub, Bv); push!(val, -y[i+1])
            push!(sub, XI + i); push!(val, 1.0)
            a_row(m, i, sub, val)
            con_bounds(m, i, BK_LO; lo=1.0, up=Inf)
        end
        solve(m)
    finally
        free(m)
    end
    r.rc == 0 || return false, 0.0, zeros(2), 0.0
    return true, r.objective, r.x[W+1:W+nd], r.x[Bv+1]
end

function main()
    bad = false
    # A) separable diamond
    X = [2.0 0; 0 2; -2 0; 0 -2]
    ok, obj, w, b = svm(2, 4, X, [1.0, 1, -1, -1], 1.0)
    good = ok && abs(obj - 0.25) < 1e-7 && abs(w[1] - 0.5) < 1e-6 && abs(w[2] - 0.5) < 1e-6 && abs(b) < 1e-6
    @printf("SVM A (separable, c=1): obj=%.8f w=(%.4f,%.4f) b=%.2e  (expected 0.25, w=(1/2,1/2), b=0)  %s\n",
            obj, w[1], w[2], b, good ? "OK" : "FAIL")
    bad |= !good
    # B) soft margin (same feature, opposite labels)
    X = [0.0 0; 0 0]
    ok, obj, w, b = svm(1, 2, X, [1.0, -1], 1.0)
    good = ok && abs(obj - 2.0) < 1e-7 && abs(w[1]) < 1e-6 && abs(b) < 1e-6
    @printf("SVM B (soft, c=1):      obj=%.8f w=(%.4f) b=%.2e      (expected 2, w=0, b=0)          %s\n",
            obj, w[1], b, good ? "OK" : "FAIL")
    bad |= !good
    return bad ? 1 : 0
end

exit(main())
