#!/usr/bin/env julia
# omf_ex312.jl — Exercise 3.12 of "Optimization Methods in Finance"
# (Cornuejols & Tutuncu, 2007), Section 3.2: dedication / cash-flow matching.
# A 9-year liability stream (24, 26, 28, 28, 26, 29, 32, 33, 34 million $) is
# funded by a least-cost portfolio of 16 bonds, cash carried at r = 2%:
#
#   min z(0) + sum_i P(i) x(i)
#   s.t.  sum_{M(i) >= t} C(i) x(i) + sum_{M(i) = t} 100 x(i) + (1+r) z(t-1) - z(t) = L(t)
#
# The book gives no optimum, so the sample recomputes the cash flows and checks
# feasibility and the objective. Port of examples/c/omf_ex312.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const T = 9
const N = 16
const R = 0.02
const L = [0.0, 24, 26, 28, 28, 26, 29, 32, 33, 34]   # L[t+1], t = 0..9
const P = [102.44, 99.95, 100.02, 102.66, 87.90, 85.43, 83.42, 103.82, 110.29, 108.85,
           109.95, 107.36, 104.62, 99.07, 103.78, 64.66]
const C = [5.625, 4.75, 4.25, 5.25, 0.0, 0.0, 0.0, 5.75, 6.875, 6.5, 6.625, 6.125, 5.625, 4.75, 5.5, 0.0]
const M = [1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]

function main()
    m = Model(T, N + T + 1)                         # vars x(i), z(0..9); rows year 1..9
    r = try
        for i in 0:N-1
            var_bounds(m, i, BK_LO; lo=0.0, up=Inf)
        end
        for j in 0:T
            var_bounds(m, N + j, BK_LO; lo=0.0, up=Inf)
        end
        cj(m, N, 1.0)
        for i in 0:N-1
            cj(m, i, P[i+1])
        end
        for yr in 1:T
            idx, val = Int[], Float64[]
            for i in 1:N
                a = 0.0
                M[i] >= yr && (a += C[i])
                M[i] == yr && (a += 100.0)
                a != 0.0 && (push!(idx, i - 1); push!(val, a))
            end
            push!(idx, N + (yr - 1)); push!(val, 1.0 + R)   # +(1+r) z(t-1)
            push!(idx, N + yr); push!(val, -1.0)            # - z(t)
            a_row(m, yr - 1, idx, val)
            con_bounds(m, yr - 1, BK_FX; lo=L[yr+1], up=L[yr+1])
        end
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        z, x = r.objective, r.x
        # independent feasibility recomputation of the cash-flow recurrence
        cost, minz, maxviol, carry = x[N+1], 0.0, 0.0, 0.0
        for i in 1:N
            cost += P[i] * x[i]
        end
        for yr in 1:T
            inflow = 0.0
            for i in 1:N
                M[i] >= yr && (inflow += C[i] * x[i])
                M[i] == yr && (inflow += 100.0 * x[i])
            end
            carry = (1.0 + R) * carry + inflow - L[yr+1]
            carry < minz && (minz = carry)
            zt = x[N+yr+1]
            if abs(zt - carry) > 1e-6
                dd = abs(zt - carry); dd > maxviol && (maxviol = dd)
            end
        end
        ok = abs(cost - z) < 1e-6 && minz > -1e-6 && maxviol < 1e-6
        @printf("omf_ex312  cost=%.4f  min surplus=%.3e  max |z-recurrence|=%.2e  %s\n",
                z, minz, maxviol, ok ? "OK" : "FAIL")
        print("  x = ")
        for i in 1:N
            @printf("%.4g ", x[i])
        end
        println()
    else
        @printf("omf_ex312  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
