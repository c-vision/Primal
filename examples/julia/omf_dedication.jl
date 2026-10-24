#!/usr/bin/env julia
# omf_dedication.jl — the dedication (cash-flow matching) bond portfolio of
# "Optimization Methods in Finance" (Cornuejols & Tutuncu, 2007), Section 3.2
# and Exercise 3.11: fund an 8-year liability stream with a least-cost
# portfolio of 10 bonds (0% reinvestment):
#
#   min z(0) + sum_i P(i) x(i)
#   s.t.  sum_{M(i) >= t} C(i) x(i) + sum_{M(i) = t} 100 x(i) + z(t-1) - z(t) = L(t)
#
# The book: cost $93,944, its holdings, shadow prices 0.971428571,
# 0.915646259, 0.883045779 and reduced costs (Bond 2, Bond 7, z0, z8).
# Port of examples/c/omf_dedication.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const T = 8
const N = 10
const L = [0.0, 12000, 18000, 20000, 20000, 16000, 15000, 12000, 10000]   # L[t+1]
const P = [102.0, 99, 101, 98, 98, 104, 100, 101, 102, 94]
const C = [5.0, 3.5, 5.0, 3.5, 4.0, 9.0, 6.0, 8.0, 9.0, 7.0]
const M = [1, 2, 2, 3, 4, 5, 5, 6, 7, 8]
const XBOOK = [62.0, 0, 125, 152, 157, 123, 0, 124, 104, 93]   # book optimum holdings

function main()
    m = Model(T, N + T + 1)                         # vars x(i), z(0..8); rows year 1..8
    try
        for i in 0:N-1
            var_bounds(m, i, BK_LO; lo=0.0, up=Inf)
        end
        for j in 0:T
            var_bounds(m, N + j, BK_LO; lo=0.0, up=Inf)
        end
        cj(m, N, 1.0)                               # z(0) in the objective
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
            push!(idx, N + (yr - 1)); push!(val, 1.0)    # + z(t-1)
            push!(idx, N + yr); push!(val, -1.0)         # - z(t)
            a_row(m, yr - 1, idx, val)
            con_bounds(m, yr - 1, BK_FX; lo=L[yr+1], up=L[yr+1])
        end
        r = solve(m)
        ok = r.rc == 0
        if ok
            z, x, sp = r.objective, r.x, r.y
            cost_book, hold_err = 0.0, 0.0
            for i in 1:N
                cost_book += P[i] * XBOOK[i]
                d = abs(x[i] - XBOOK[i])
                d > hold_err && (hold_err = d)
            end
            # shadow prices: the book prints them positive (same magnitude)
            sp_book = [0.971428571, 0.915646259, 0.883045779]
            sp_ok = all(k -> abs(abs(sp[k]) - sp_book[k]) < 1e-5, 1:3)
            # reduced costs from the Section 3.3.2 sensitivity table
            rc = getreducedcosts(m, 0, N + T + 1)
            rc_book = [0.830612245, 8.786840002, 0.028571429, 0.524288903]
            rc_ok = abs(rc[2] - rc_book[1]) < 1e-5 &&      # Bond 2
                    abs(rc[7] - rc_book[2]) < 1e-5 &&      # Bond 7
                    abs(rc[N+1] - rc_book[3]) < 1e-5 &&    # z0
                    abs(rc[N+9] - rc_book[4]) < 1e-5       # z8
            ok = abs(z - 93944.0) < 1.0 && hold_err < 1.0 && sp_ok && rc_ok
            @printf("omf_dedication  cost=%.4f (book 93944)  max|x-xbook|=%.4f  %s\n", z, hold_err, ok ? "OK" : "FAIL")
            @printf("  shadow:  sp1=%.6f sp2=%.6f sp3=%.6f  %s\n", abs(sp[1]), abs(sp[2]), abs(sp[3]), sp_ok ? "OK" : "FAIL")
            @printf("  reduced: B2=%.6f B7=%.6f z0=%.6f z8=%.6f  %s\n", rc[2], rc[7], rc[N+1], rc[N+9], rc_ok ? "OK" : "FAIL")
            print("  holdings: ")
            for i in 1:N
                @printf("B%d=%.4g ", i, x[i])
            end
            @printf("\n  (bond-only cost = %.2f, z0 = %.4f, book bond-only = %.2f)\n", z - x[N+1], x[N+1], cost_book)
        else
            @printf("omf_dedication  rc=%d FAIL\n", r.rc)
        end
        return ok ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
