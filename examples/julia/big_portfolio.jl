#!/usr/bin/env julia
# big_portfolio.jl — the full mean-variance + correlation-stress pipeline of
# "BIG Portfolio Optimisation" (tbc81/big-portfolio-optimisation, MIT) on the
# real market estimates of that pipeline (18 held assets, cash at 1.5%):
#
#   1) observed snapshot  R0 = mu'x0, vol0 = sqrt(x0'Sigma x0)
#   2) min risk at R0 and 3) max return at vol0 (SOCPs, ||G'x|| with Sigma = G G')
#   4) correlation stress: the nearest valid C* (SDP) with C*_ij >= 0.85 on the
#      technology/growth group, Sigma* = D C* D
#   5) the risk multiplier and 6) min risk at R0 under Sigma*
#
# Reference values: vol0 = 0.229369, stressed vol 0.257033, multiplier
# 1.255767, min risk 0.135508 / 0.137191, max return 0.400353.
# Port of examples/c/big_portfolio.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const N = 18
const RHO = 0.85
const ASSET = ["Rheinmetall", "Parkmead", "ASML", "Roku", "CSH2", "Gold", "Duke Energy", "BAE Systems", "Nucor", "Diageo", "Legal & General", "Apollo", "Cocoa", "Salesforce", "Cameco", "Marvell", "Micron", "Volatus Aerospace"]
const MU = [0.88505759, 0.01637996, 0.25760384, 0.07062742, 0.03748831, 0.21091841, 0.09583463, 0.33645087, 0.29112897, -0.12002662, 0.04296791, 0.28051820, 0.45077128, 0.07685441, 0.48956336, 0.52495055, 0.96592058, 0.13350019]
const X0 = [0.10000000, 0.10000000, 0.10000000, 0.10000000, 0.07500000, 0.06500000, 0.05000000, 0.05000000, 0.05000000, 0.05000000, 0.05000000, 0.05000000, 0.05000000, 0.02500000, 0.01000000, 0.02500000, 0.02500000, 0.01000000]
const SIG = [
    7.8693177181e-01  2.6803537860e-02  5.6538149846e-02  -1.0901486761e-02  3.2487564874e-06  2.9919522273e-02  8.0130282213e-03  2.1544668156e-01  3.9915837454e-02  -8.7715806493e-04  2.1050625025e-02  2.6609613869e-02  -5.3455522446e-02  4.8313824939e-02  1.0499183179e-01  -3.6500946465e-02  2.4902557447e-02  1.3222599613e-01
    2.6803537860e-02  4.5928309389e-01  1.9088877991e-02  -1.0061816532e-02  -1.7632197774e-05  7.4229946741e-03  -7.0077694055e-03  1.4311364548e-02  4.2593072806e-02  5.3360311065e-03  2.0762686147e-03  1.9391039637e-02  2.1180310268e-02  1.5037878828e-02  2.0166003415e-02  3.5629773995e-02  2.9537667670e-02  -1.3758533343e-02
    5.6538149846e-02  1.9088877991e-02  2.5066126592e-01  1.1813502943e-01  9.6443250392e-05  1.3240649054e-03  -1.0765482295e-02  7.0614340589e-03  6.0627095032e-02  2.1869705589e-02  3.7913336317e-02  6.5549905806e-02  1.5194369470e-02  5.8940340094e-02  9.5462362195e-02  2.1598452995e-01  2.6458674616e-01  6.4507143109e-02
    -1.0901486761e-02  -1.0061816532e-02  1.1813502943e-01  6.0904867749e-01  2.7241999281e-04  -1.1933723754e-02  -9.7657364319e-03  1.3790643091e-03  7.6301742916e-02  2.0772719604e-02  4.7275342882e-02  1.3402884736e-01  3.7153299320e-02  1.3623821088e-01  1.2386162419e-01  2.1315465175e-01  2.2927720652e-01  6.8311932735e-02
    3.2487564874e-06  -1.7632197774e-05  9.6443250392e-05  2.7241999281e-04  7.5685315635e-06  3.0942343054e-05  -1.9767915538e-06  4.8237201133e-05  -1.6353524879e-05  -6.7791802392e-05  3.5668333897e-05  4.6410160705e-05  7.7073673639e-05  1.0602503564e-04  6.1340301552e-05  1.5667590176e-04  1.6094265546e-04  1.0938845624e-05
    2.9919522273e-02  7.4229946741e-03  1.3240649054e-03  -1.1933723754e-02  3.0942343054e-05  3.6356132597e-02  7.3390627470e-03  1.3515940803e-02  -9.0856025023e-04  1.9878094451e-03  -2.7543099646e-03  -1.0684186842e-02  -1.1816937025e-02  2.0263475533e-03  2.8071062208e-02  -6.0621001423e-03  2.5661584954e-02  2.1354342979e-02
    8.0130282213e-03  -7.0077694055e-03  -1.0765482295e-02  -9.7657364319e-03  -1.9767915538e-06  7.3390627470e-03  4.0976349236e-02  1.0543056505e-02  9.5639122061e-03  3.3001355599e-03  8.4537581703e-03  1.9695009493e-03  -1.6412670252e-02  -8.3389210171e-03  1.5335439692e-02  -1.1462993782e-02  -2.7969778114e-02  1.2924825501e-02
    2.1544668156e-01  1.4311364548e-02  7.0614340589e-03  1.3790643091e-03  4.8237201133e-05  1.3515940803e-02  1.0543056505e-02  1.4817381301e-01  1.1582694026e-02  1.7261760613e-03  1.1402050587e-02  1.2590886788e-02  -3.6625754020e-02  1.4364919987e-02  4.3143066606e-02  -2.2381650207e-02  2.9376699243e-02  4.6318187274e-02
    3.9915837454e-02  4.2593072806e-02  6.0627095032e-02  7.6301742916e-02  -1.6353524879e-05  -9.0856025023e-04  9.5639122061e-03  1.1582694026e-02  2.5772581003e-01  1.4019697025e-02  3.1909570880e-02  9.1302221879e-02  2.0810635168e-02  5.1315070545e-02  1.0541171559e-01  1.1946151056e-01  1.4960500545e-01  3.4878112774e-02
    -8.7715806493e-04  5.3360311065e-03  2.1869705589e-02  2.0772719604e-02  -6.7791802392e-05  1.9878094451e-03  3.3001355599e-03  1.7261760613e-03  1.4019697025e-02  4.1651678600e-02  1.5098335082e-02  1.2791454191e-02  1.6338995724e-02  1.0370576683e-02  1.5773922448e-02  1.3630426101e-02  3.2894027535e-02  1.2690667367e-03
    2.1050625025e-02  2.0762686147e-03  3.7913336317e-02  4.7275342882e-02  3.5668333897e-05  -2.7543099646e-03  8.4537581703e-03  1.1402050587e-02  3.1909570880e-02  1.5098335082e-02  6.2684950141e-02  3.6932759499e-02  -5.8850168677e-04  2.0251527105e-02  3.9401137957e-02  3.6294288556e-02  4.9918556905e-02  1.1330921374e-02
    2.6609613869e-02  1.9391039637e-02  6.5549905806e-02  1.3402884736e-01  4.6410160705e-05  -1.0684186842e-02  1.9695009493e-03  1.2590886788e-02  9.1302221879e-02  1.2791454191e-02  3.6932759499e-02  2.4372320134e-01  6.7285867733e-02  7.5349253616e-02  1.0744939020e-01  1.7602006115e-01  1.6081480146e-01  1.5054071340e-02
    -5.3455522446e-02  2.1180310268e-02  1.5194369470e-02  3.7153299320e-02  7.7073673639e-05  -1.1816937025e-02  -1.6412670252e-02  -3.6625754020e-02  2.0810635168e-02  1.6338995724e-02  -5.8850168677e-04  6.7285867733e-02  5.7209925474e-01  2.8984962596e-02  4.1329872305e-02  4.8367745132e-02  1.0054468065e-01  -1.2663102393e-02
    4.8313824939e-02  1.5037878828e-02  5.8940340094e-02  1.3623821088e-01  1.0602503564e-04  2.0263475533e-03  -8.3389210171e-03  1.4364919987e-02  5.1315070545e-02  1.0370576683e-02  2.0251527105e-02  7.5349253616e-02  2.8984962596e-02  1.8154987864e-01  7.4519454632e-02  1.1827749460e-01  9.0697380314e-02  4.7172627159e-02
    1.0499183179e-01  2.0166003415e-02  9.5462362195e-02  1.2386162419e-01  6.1340301552e-05  2.8071062208e-02  1.5335439692e-02  4.3143066606e-02  1.0541171559e-01  1.5773922448e-02  3.9401137957e-02  1.0744939020e-01  4.1329872305e-02  7.4519454632e-02  5.3192394146e-01  1.8840523736e-01  2.2317811290e-01  1.1312375259e-01
    -3.6500946465e-02  3.5629773995e-02  2.1598452995e-01  2.1315465175e-01  1.5667590176e-04  -6.0621001423e-03  -1.1462993782e-02  -2.2381650207e-02  1.1946151056e-01  1.3630426101e-02  3.6294288556e-02  1.7602006115e-01  4.8367745132e-02  1.1827749460e-01  1.8840523736e-01  8.3973674794e-01  4.6250714851e-01  6.5406670102e-02
    2.4902557447e-02  2.9537667670e-02  2.6458674616e-01  2.2927720652e-01  1.6094265546e-04  2.5661584954e-02  -2.7969778114e-02  2.9376699243e-02  1.4960500545e-01  3.2894027535e-02  4.9918556905e-02  1.6081480146e-01  1.0054468065e-01  9.0697380314e-02  2.2317811290e-01  4.6250714851e-01  1.3161522295e+00  1.2823281199e-01
    1.3222599613e-01  -1.3758533343e-02  6.4507143109e-02  6.8311932735e-02  1.0938845624e-05  2.1354342979e-02  1.2924825501e-02  4.6318187274e-02  3.4878112774e-02  1.2690667367e-03  1.1330921374e-02  1.5054071340e-02  -1.2663102393e-02  4.7172627159e-02  1.1312375259e-01  6.5406670102e-02  1.2823281199e-01  1.0492689215e+00
]
# technology/growth stress pairs among the held assets (0-based indices)
const PAIRS = [(2, 3), (2, 13), (2, 15), (2, 16), (3, 13), (3, 15), (3, 16), (13, 15), (13, 16), (15, 16)]

# correlation matrix of a covariance, and back with the variances of S
function corr_from_cov(Sm)
    d = [sqrt(Sm[i, i]) for i in 1:N]
    return [Sm[i, j] / (d[i] * d[j]) for i in 1:N, j in 1:N]
end
function cov_from_corr(Sm, C)
    d = [sqrt(Sm[i, i]) for i in 1:N]
    return [d[i] * d[j] * C[i, j] for i in 1:N, j in 1:N]
end

# Cholesky factor G with S = G G'; returns (ok, G).
function chol(Sm)
    G = zeros(N, N)
    for i in 1:N, j in 1:i
        s = Sm[i, j]
        for k in 1:j-1
            s -= G[i, k] * G[j, k]
        end
        if i == j
            s <= 0.0 && return false, G
            G[i, j] = sqrt(s)
        else
            G[i, j] = s / G[j, j]
        end
    end
    return true, G
end

# portfolio variance x'S x
function port_var(x, Sm)
    v = 0.0
    for i in 1:N
        t = 0.0
        for j in 1:N
            t += Sm[i, j] * x[j]
        end
        v += x[i] * t
    end
    return v
end

# min risk at a target return (SOCP): vars x, c, t, y = G'x; returns (ok, vol).
function min_risk(G, target)
    m = Model(N + 2, 2N + 2)
    r = try
        for j in 0:N
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        var_bounds(m, N + 1, BK_LO; lo=0.0, up=Inf)
        for i in 0:N-1
            var_bounds(m, N + 2 + i, BK_FR; lo=-Inf, up=Inf)
        end
        cj(m, N + 1, 1.0)
        a_row(m, 0, collect(0:N), ones(N + 1))
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        a_row(m, 1, collect(0:N-1), MU)
        con_bounds(m, 1, BK_LO; lo=target, up=Inf)
        for i in 0:N-1
            a_row(m, 2 + i, [collect(0:N-1); N + 2 + i], [-G[:, i+1]; 1.0])
            con_bounds(m, 2 + i, BK_FX; lo=0.0, up=0.0)
        end
        cone(m, CT_QUAD, [N + 1; collect(N+2:2N+1)])
        solve(m)
    finally
        free(m)
    end
    return r.rc == 0, r.objective
end

# max return at a risk limit (SOCP): vars x, c, y = G'x, rb; returns (ok, return).
function max_return(G, risk)
    m = Model(N + 1, 2N + 2)
    r = try
        for j in 0:N
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        for i in 0:N-1
            var_bounds(m, N + 1 + i, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(m, 2N + 1, BK_FX; lo=risk, up=risk)
        sense(m, SENSE_MAX)
        for j in 0:N-1
            cj(m, j, MU[j+1])
        end
        a_row(m, 0, collect(0:N), ones(N + 1))
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        for i in 0:N-1
            a_row(m, 1 + i, [collect(0:N-1); N + 1 + i], [-G[:, i+1]; 1.0])
            con_bounds(m, 1 + i, BK_FX; lo=0.0, up=0.0)
        end
        cone(m, CT_QUAD, [2N + 1; collect(N+1:2N)])
        solve(m)
    finally
        free(m)
    end
    return r.rc == 0, r.objective
end

# nearest valid stressed correlation matrix (SDP); returns (ok, C*).
function stress_corr(C)
    pairs = [(i, j) for i in 0:N-1 for j in i+1:N-1]
    NP = length(pairs)
    isgrp = [any(g -> (p == g || p == (g[2], g[1])), PAIRS) for p in pairs]
    m = Model(0, 0)
    try
        add_vars(m, 2NP + 1)                        # vars C* (NP), t, d (NP)
        for k in 0:NP-1
            isgrp[k+1] ? var_bounds(m, k, BK_LO; lo=RHO, up=Inf) : var_bounds(m, k, BK_RA; lo=-1.0, up=1.0)
        end
        var_bounds(m, NP, BK_LO; lo=0.0, up=Inf)
        for k in 0:NP-1
            var_bounds(m, NP + 1 + k, BK_FR; lo=-Inf, up=Inf)
        end
        cj(m, NP, 1.0)
        esym = [sparsesymmat(m, N, [(i, i, 1.0)]) for i in 0:N-1]
        psym = [sparsesymmat(m, N, [(i, j, 1.0)]) for (i, j) in pairs]
        barvars(m, [N])
        add_cons(m, NP + N + NP)
        for k in 0:NP-1                             # d_k - sqrt2 C*_k = -sqrt2 C_k
            i, j = pairs[k+1]
            a_row(m, k, [NP + 1 + k, k], [1.0, -sqrt(2.0)])
            con_bounds(m, k, BK_FX; lo=-sqrt(2.0) * C[i+1, j+1], up=-sqrt(2.0) * C[i+1, j+1])
        end
        for i in 0:N-1                              # <E_ii,X> = 1
            baraij(m, NP + i, 0, [(esym[i+1], 1.0)])
            con_bounds(m, NP + i, BK_FX; lo=1.0, up=1.0)
        end
        for k in 0:NP-1                             # <E_ij,X> - 2 C*_k = 0
            baraij(m, NP + N + k, 0, [(psym[k+1], 1.0)])
            a_row(m, NP + N + k, [k], [-2.0])
            con_bounds(m, NP + N + k, BK_FX; lo=0.0, up=0.0)
        end
        cone(m, CT_QUAD, [NP; collect(NP+1:2NP)])
        r = solve(m)
        r.rc == 0 || return false, zeros(N, N)
        X = getbarxj(m, 0)
        return true, [X[(i-1)*N+j] for i in 1:N, j in 1:N]
    finally
        free(m)
    end
end

function main()
    ok = true
    R0 = 0.0
    for j in 1:N
        R0 += MU[j] * X0[j]
    end
    var0 = port_var(X0, SIG)
    vol0 = sqrt(var0)
    g, G = chol(SIG); ok &= g
    g, vols_base = min_risk(G, R0); ok &= g
    g, Rrisk = max_return(G, vol0); ok &= g
    C = corr_from_cov(SIG)
    g, Cs = stress_corr(C); ok &= g
    Ss = cov_from_corr(SIG, Cs)
    volstr = sqrt(port_var(X0, Ss))
    mult = port_var(X0, Ss) / var0
    g, Gs = chol(Ss); ok &= g
    g, vols_str = min_risk(Gs, R0); ok &= g
    if ok
        ok_c = all(abs(Cs[a+1, b+1] - RHO) <= 1e-3 for (a, b) in PAIRS)
        ok = ok_c && abs(vol0 - 0.229369) < 1e-4 && abs(mult - 1.255767) < 2e-3 && abs(volstr - 0.257033) < 2e-3 &&
             abs(vols_base - 0.135508) < 2e-3 && abs(vols_str - 0.137191) < 2e-3 && abs(Rrisk - 0.400353) < 2e-3
        @printf("big_portfolio  N=%d (%s..%s)  R0=%.6f vol0=%.6f (case 0.229369)\n", N, ASSET[1], ASSET[N], R0, vol0)
        @printf("               group C* entries=0.85  multiplier=%.6f (case 1.255767) stressed vol=%.6f (case 0.257033)\n",
                mult, volstr)
        @printf("               min-risk@R0 base=%.6f stressed=%.6f (case 0.135508 / 0.137191)\n", vols_base, vols_str)
        @printf("               max-return@vol0=%.6f (case 0.400353)  %s\n", Rrisk, ok ? "OK" : "FAIL")
    else
        println("big_portfolio  solve FAIL (rc)")
    end
    return ok ? 0 : 1
end

exit(main())
