#!/usr/bin/env julia
# omf_ex118.jl — Exercise 11.8 of "Optimization Methods in Finance"
# (Cornuejols & Tutuncu, 2007), Section 11.3.3: Gomory mixed integer cuts.
#
#   max 10 x1 + 13 x2  s.t.  10 x1 + 14 x2 <= 43,  x >= 0 integer
#
# (i) The LP relaxation's tableau row is x1 + 1.4 x2 + 0.1 x3 = 4.3 (z = 43).
# (ii)/(iii) The GMI cut of that row times k = 3 is 3 x1 + 4 x2 <= 12.
# (iv) With the cut the LP optimum is 40 at (4, 0), the integer optimum.
# Port of examples/c/omf_ex118.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# GMI cut for the row sum_j a_j x_j = b multiplied by k: returns (cut_a, cut_rhs).
function gmi_cut(a, b, k)
    kb = Float64(k) * b
    f0 = kb - floor(kb)
    cut = zeros(length(a))
    for j in eachindex(a)
        ka = Float64(k) * a[j]
        fj = ka - floor(ka)
        t = fj - f0
        cut[j] = floor(ka) + (t > 0.0 ? t / (1.0 - f0) : 0.0)
    end
    return cut, floor(kb)
end

# LP relaxation of the IP (optionally with the GMI cut); int_flag = MIP.
function solve_ip(with_cut, int_flag)
    m = Model(with_cut ? 2 : 1, 2)
    r = try
        for j in 0:1
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
            int_flag && var_type(m, j, VT_INT)
        end
        cj(m, 0, -10.0); cj(m, 1, -13.0)            # max
        a_row(m, 0, [0, 1], [10.0, 14.0])
        con_bounds(m, 0, BK_UP; lo=-Inf, up=43.0)
        if with_cut
            a_row(m, 1, [0, 1], [3.0, 4.0])
            con_bounds(m, 1, BK_UP; lo=-Inf, up=12.0)
        end
        solve(m)
    finally
        free(m)
    end
    r.rc == 0 || return false, 0.0, 0.0, 0.0
    return true, r.objective, r.x[1], r.x[2]
end

function main()
    ok = true
    good, z, x1, x2 = solve_ip(false, false)        # (i) LP relaxation
    if !good
        println("omf_ex118 LP FAIL")
        return 1
    end
    relax_ok = abs(-z - 43.0) < 1e-6 && abs(x1 - 4.3) < 1e-6 && abs(x2) < 1e-6
    @printf("omf_ex118 (i)  LP relaxation: x1=%.4f x2=%.4f z=%.6g (book 4.3/0/43) %s\n",
            x1, x2, -z, relax_ok ? "OK" : "FAIL")
    ok &= relax_ok

    cut, rhs = gmi_cut([1.0, 1.4, 0.1], 4.3, 3)     # (ii)/(iii) GMI k = 3 -> 3 x1 + 4 x2 <= 12
    cut_ok = abs(cut[1] - 3.0) < 1e-9 && abs(cut[2] - 4.0) < 1e-9 && abs(cut[3] - 0.0) < 1e-9 &&
             abs(rhs - 12.0) < 1e-9 && 3.0 * 4.3 > 12.0   # cuts off the LP point
    @printf("omf_ex118 (ii) GMI k=3: %.4g x1 + %.4g x2 + %.4g x3 <= %.4g (3*4.3=%.1f>12) %s\n",
            cut[1], cut[2], cut[3], rhs, 3.0 * 4.3, cut_ok ? "OK" : "FAIL")
    ok &= cut_ok

    lp_ok, zc, xc1, xc2 = solve_ip(true, false)     # (iv) LP with the cut, and the MIP
    mip_ok, zi, xi1, xi2 = solve_ip(true, true)
    cut_lp_ok = lp_ok && abs(-zc - 40.0) < 1e-6 && abs(xc1 - 4.0) < 1e-6 && abs(xc2) < 1e-6
    mip_good = mip_ok && abs(-zi - 40.0) < 1e-6 && abs(xi1 - 4.0) < 1e-6 && abs(xi2) < 1e-6
    @printf("omf_ex118 (iv) LP+cut: x=(%.4g,%.4g) z=%.6g   MIP: x=(%.4g,%.4g) z=%.6g  (book 40 at (4,0)) %s\n",
            xc1, xc2, -zc, xi1, xi2, -zi, (cut_lp_ok && mip_good) ? "OK" : "FAIL")
    ok &= cut_lp_ok && mip_good
    return ok ? 0 : 1
end

exit(main())
