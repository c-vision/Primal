#!/usr/bin/env julia
# omf_ex410.jl — Exercise 4.10 of "Optimization Methods in Finance"
# (Cornuejols & Tutuncu, 2007), Chapter 4: $20,000 to invest in bonds B,
# shares S of XYZ and calls C (bought > 0, sold < 0, |C| <= 50), with three
# equally likely scenarios for XYZ ($20, $40, $12).
#
#   (i)   max 10 B + 4 S, budget 90 B + 20 S + 1000 C <= 20000 -> profit 14000
#   (iii) max mean scenario profit with every P_i >= 2000      -> 2800 shares, sell 36 calls, 11200
#   (iv)  max the riskless profit Z (P_i >= Z)                  -> Z = 7272.73, expected 9090.91
#
# Port of examples/c/omf_ex410.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# scenario rows P1 = 10B + 20S + 1500C, P2 = 10B - 500C, P3 = 10B - 8S - 1000C
function scenario_rows(m)
    a_row(m, 0, [0, 1, 2], [90.0, 20.0, 1000.0])   # budget
    con_bounds(m, 0, BK_UP; lo=-Inf, up=20000.0)
    a_row(m, 1, [0, 1, 2, 3], [10.0, 20.0, 1500.0, -1.0]); con_bounds(m, 1, BK_FX; lo=0.0, up=0.0)
    a_row(m, 2, [0, 2, 4], [10.0, -500.0, -1.0]);          con_bounds(m, 2, BK_FX; lo=0.0, up=0.0)
    a_row(m, 3, [0, 1, 2, 5], [10.0, -8.0, -1000.0, -1.0]); con_bounds(m, 3, BK_FX; lo=0.0, up=0.0)
end

budget(x) = 90.0 * x[1] + 20.0 * x[2] + 1000.0 * x[3]

function part_i()
    m = Model(1, 3)
    r = try
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)     # B
        var_bounds(m, 1, BK_LO; lo=0.0, up=Inf)     # S
        var_bounds(m, 2, BK_RA; lo=-50.0, up=50.0)  # C
        cj(m, 0, -10.0); cj(m, 1, -4.0)             # max 10B + 4S
        a_row(m, 0, [0, 1, 2], [90.0, 20.0, 1000.0])
        con_bounds(m, 0, BK_UP; lo=-Inf, up=20000.0)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        z, x = r.objective, r.x
        ok = abs(z + 14000.0) < 1e-6 && abs(x[1] - 0.0) < 1e-6 && abs(x[2] - 3500.0) < 1e-6 &&
             abs(x[3] + 50.0) < 1e-6 && budget(x) <= 20000.0 + 1e-6
        @printf("omf_ex410 (i)   B=%.4g S=%.4g C=%.4g  profit=%.6g (book 14000) %s\n", x[1], x[2], x[3], -z, ok ? "OK" : "FAIL")
    else
        @printf("omf_ex410 (i)   rc=%d FAIL\n", r.rc)
    end
    return ok
end

function part_iii()
    m = Model(4, 6)                                 # B S C P1 P2 P3 ; rows budget, P definitions
    r = try
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 1, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 2, BK_RA; lo=-50.0, up=50.0)
        for k in 3:5
            var_bounds(m, k, BK_LO; lo=2000.0, up=Inf)
        end
        cj(m, 3, -1.0 / 3.0); cj(m, 4, -1.0 / 3.0); cj(m, 5, -1.0 / 3.0)
        scenario_rows(m)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        z, x = r.objective, r.x
        ok = abs(z + 11200.0) < 1e-6 && abs(x[2] - 2800.0) < 1e-6 && abs(x[3] + 36.0) < 1e-6 &&
             budget(x) <= 20000.0 + 1e-6 && all(k -> x[k] >= 2000.0 - 1e-6, 4:6)
        @printf("omf_ex410 (iii) B=%.4g S=%.4g C=%.4g  P=(%.6g,%.6g,%.6g)  profit=%.6g (book 11200) %s\n",
                x[1], x[2], x[3], x[4], x[5], x[6], -z, ok ? "OK" : "FAIL")
    else
        @printf("omf_ex410 (iii) rc=%d FAIL\n", r.rc)
    end
    return ok
end

function part_iv()
    m = Model(7, 7)                                 # B S C P1 P2 P3 Z ; rows budget, P, P >= Z
    r = try
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 1, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 2, BK_RA; lo=-50.0, up=50.0)
        for k in 3:6
            var_bounds(m, k, BK_FR; lo=-Inf, up=Inf)
        end
        cj(m, 6, -1.0)                              # max Z
        scenario_rows(m)
        for k in 0:2                                # P_i - Z >= 0
            a_row(m, 4 + k, [3 + k, 6], [1.0, -1.0])
            con_bounds(m, 4 + k, BK_LO; lo=0.0, up=Inf)
        end
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        z, x = r.objective, r.x
        exp_profit = (x[4] + x[5] + x[6]) / 3.0
        # the book rounds; the exact vertex is S=2272.73, C=-25.4545, Z=7272.73
        ok = abs(z + 7272.727) < 1e-2 && abs(exp_profit - 9090.909) < 1e-2 && abs(budget(x) - 20000.0) < 1e-6 &&
             all(k -> x[k] >= x[7] - 1e-6, 4:6)
        @printf("omf_ex410 (iv)  B=%.4g S=%.4g C=%.4g  riskless=%.6g (book 7272)  expected=%.6g (book 9091) %s\n",
                x[1], x[2], x[3], -z, exp_profit, ok ? "OK" : "FAIL")
    else
        @printf("omf_ex410 (iv)  rc=%d FAIL\n", r.rc)
    end
    return ok
end

function main()
    a = part_i()
    b = part_iii()
    c = part_iv()
    return (a && b && c) ? 0 : 1
end

exit(main())
