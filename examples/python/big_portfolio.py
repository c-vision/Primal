#!/usr/bin/env python3
"""The full mean-variance + correlation-stress pipeline of "BIG Portfolio Optimisation"
(tbc81/big-portfolio-optimisation, MIT) on the real market estimates of that pipeline
(18 held assets, cash at 1.5%):

  1) observed snapshot  R0 = mu'x0, vol0 = sqrt(x0'Sigma x0)
  2) min risk at R0 and 3) max return at vol0 (SOCPs, ||G'x|| with Sigma = G G')
  4) correlation stress: the nearest valid C* (SDP) with C*_ij >= 0.85 on the
     technology/growth group, Sigma* = D C* D
  5) the risk multiplier and 6) min risk at R0 under Sigma*

Reference values: vol0 = 0.229369, stressed vol 0.257033, multiplier 1.255767,
min risk 0.135508 / 0.137191, max return 0.400353. Port of examples/c/big_portfolio.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

N = 18
RHO = 0.85
ASSET = ["Rheinmetall", "Parkmead", "ASML", "Roku", "CSH2", "Gold", "Duke Energy", "BAE Systems", "Nucor",
         "Diageo", "Legal & General", "Apollo", "Cocoa", "Salesforce", "Cameco", "Marvell", "Micron",
         "Volatus Aerospace"]
MU = [0.88505759, 0.01637996, 0.25760384, 0.07062742, 0.03748831, 0.21091841,
      0.09583463, 0.33645087, 0.29112897, -0.12002662, 0.04296791, 0.28051820,
      0.45077128, 0.07685441, 0.48956336, 0.52495055, 0.96592058, 0.13350019]
X0 = [0.10, 0.10, 0.10, 0.10, 0.075, 0.065, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
      0.05, 0.025, 0.01, 0.025, 0.025, 0.01]
SIG = [
    [7.8693177181e-01, 2.6803537860e-02, 5.6538149846e-02, -1.0901486761e-02, 3.2487564874e-06, 2.9919522273e-02, 8.0130282213e-03, 2.1544668156e-01, 3.9915837454e-02, -8.7715806493e-04, 2.1050625025e-02, 2.6609613869e-02, -5.3455522446e-02, 4.8313824939e-02, 1.0499183179e-01, -3.6500946465e-02, 2.4902557447e-02, 1.3222599613e-01],
    [2.6803537860e-02, 4.5928309389e-01, 1.9088877991e-02, -1.0061816532e-02, -1.7632197774e-05, 7.4229946741e-03, -7.0077694055e-03, 1.4311364548e-02, 4.2593072806e-02, 5.3360311065e-03, 2.0762686147e-03, 1.9391039637e-02, 2.1180310268e-02, 1.5037878828e-02, 2.0166003415e-02, 3.5629773995e-02, 2.9537667670e-02, -1.3758533343e-02],
    [5.6538149846e-02, 1.9088877991e-02, 2.5066126592e-01, 1.1813502943e-01, 9.6443250392e-05, 1.3240649054e-03, -1.0765482295e-02, 7.0614340589e-03, 6.0627095032e-02, 2.1869705589e-02, 3.7913336317e-02, 6.5549905806e-02, 1.5194369470e-02, 5.8940340094e-02, 9.5462362195e-02, 2.1598452995e-01, 2.6458674616e-01, 6.4507143109e-02],
    [-1.0901486761e-02, -1.0061816532e-02, 1.1813502943e-01, 6.0904867749e-01, 2.7241999281e-04, -1.1933723754e-02, -9.7657364319e-03, 1.3790643091e-03, 7.6301742916e-02, 2.0772719604e-02, 4.7275342882e-02, 1.3402884736e-01, 3.7153299320e-02, 1.3623821088e-01, 1.2386162419e-01, 2.1315465175e-01, 2.2927720652e-01, 6.8311932735e-02],
    [3.2487564874e-06, -1.7632197774e-05, 9.6443250392e-05, 2.7241999281e-04, 7.5685315635e-06, 3.0942343054e-05, -1.9767915538e-06, 4.8237201133e-05, -1.6353524879e-05, -6.7791802392e-05, 3.5668333897e-05, 4.6410160705e-05, 7.7073673639e-05, 1.0602503564e-04, 6.1340301552e-05, 1.5667590176e-04, 1.6094265546e-04, 1.0938845624e-05],
    [2.9919522273e-02, 7.4229946741e-03, 1.3240649054e-03, -1.1933723754e-02, 3.0942343054e-05, 3.6356132597e-02, 7.3390627470e-03, 1.3515940803e-02, -9.0856025023e-04, 1.9878094451e-03, -2.7543099646e-03, -1.0684186842e-02, -1.1816937025e-02, 2.0263475533e-03, 2.8071062208e-02, -6.0621001423e-03, 2.5661584954e-02, 2.1354342979e-02],
    [8.0130282213e-03, -7.0077694055e-03, -1.0765482295e-02, -9.7657364319e-03, -1.9767915538e-06, 7.3390627470e-03, 4.0976349236e-02, 1.0543056505e-02, 9.5639122061e-03, 3.3001355599e-03, 8.4537581703e-03, 1.9695009493e-03, -1.6412670252e-02, -8.3389210171e-03, 1.5335439692e-02, -1.1462993782e-02, -2.7969778114e-02, 1.2924825501e-02],
    [2.1544668156e-01, 1.4311364548e-02, 7.0614340589e-03, 1.3790643091e-03, 4.8237201133e-05, 1.3515940803e-02, 1.0543056505e-02, 1.4817381301e-01, 1.1582694026e-02, 1.7261760613e-03, 1.1402050587e-02, 1.2590886788e-02, -3.6625754020e-02, 1.4364919987e-02, 4.3143066606e-02, -2.2381650207e-02, 2.9376699243e-02, 4.6318187274e-02],
    [3.9915837454e-02, 4.2593072806e-02, 6.0627095032e-02, 7.6301742916e-02, -1.6353524879e-05, -9.0856025023e-04, 9.5639122061e-03, 1.1582694026e-02, 2.5772581003e-01, 1.4019697025e-02, 3.1909570880e-02, 9.1302221879e-02, 2.0810635168e-02, 5.1315070545e-02, 1.0541171559e-01, 1.1946151056e-01, 1.4960500545e-01, 3.4878112774e-02],
    [-8.7715806493e-04, 5.3360311065e-03, 2.1869705589e-02, 2.0772719604e-02, -6.7791802392e-05, 1.9878094451e-03, 3.3001355599e-03, 1.7261760613e-03, 1.4019697025e-02, 4.1651678600e-02, 1.5098335082e-02, 1.2791454191e-02, 1.6338995724e-02, 1.0370576683e-02, 1.5773922448e-02, 1.3630426101e-02, 3.2894027535e-02, 1.2690667367e-03],
    [2.1050625025e-02, 2.0762686147e-03, 3.7913336317e-02, 4.7275342882e-02, 3.5668333897e-05, -2.7543099646e-03, 8.4537581703e-03, 1.1402050587e-02, 3.1909570880e-02, 1.5098335082e-02, 6.2684950141e-02, 3.6932759499e-02, -5.8850168677e-04, 2.0251527105e-02, 3.9401137957e-02, 3.6294288556e-02, 4.9918556905e-02, 1.1330921374e-02],
    [2.6609613869e-02, 1.9391039637e-02, 6.5549905806e-02, 1.3402884736e-01, 4.6410160705e-05, -1.0684186842e-02, 1.9695009493e-03, 1.2590886788e-02, 9.1302221879e-02, 1.2791454191e-02, 3.6932759499e-02, 2.4372320134e-01, 6.7285867733e-02, 7.5349253616e-02, 1.0744939020e-01, 1.7602006115e-01, 1.6081480146e-01, 1.5054071340e-02],
    [-5.3455522446e-02, 2.1180310268e-02, 1.5194369470e-02, 3.7153299320e-02, 7.7073673639e-05, -1.1816937025e-02, -1.6412670252e-02, -3.6625754020e-02, 2.0810635168e-02, 1.6338995724e-02, -5.8850168677e-04, 6.7285867733e-02, 5.7209925474e-01, 2.8984962596e-02, 4.1329872305e-02, 4.8367745132e-02, 1.0054468065e-01, -1.2663102393e-02],
    [4.8313824939e-02, 1.5037878828e-02, 5.8940340094e-02, 1.3623821088e-01, 1.0602503564e-04, 2.0263475533e-03, -8.3389210171e-03, 1.4364919987e-02, 5.1315070545e-02, 1.0370576683e-02, 2.0251527105e-02, 7.5349253616e-02, 2.8984962596e-02, 1.8154987864e-01, 7.4519454632e-02, 1.1827749460e-01, 9.0697380314e-02, 4.7172627159e-02],
    [1.0499183179e-01, 2.0166003415e-02, 9.5462362195e-02, 1.2386162419e-01, 6.1340301552e-05, 2.8071062208e-02, 1.5335439692e-02, 4.3143066606e-02, 1.0541171559e-01, 1.5773922448e-02, 3.9401137957e-02, 1.0744939020e-01, 4.1329872305e-02, 7.4519454632e-02, 5.3192394146e-01, 1.8840523736e-01, 2.2317811290e-01, 1.1312375259e-01],
    [-3.6500946465e-02, 3.5629773995e-02, 2.1598452995e-01, 2.1315465175e-01, 1.5667590176e-04, -6.0621001423e-03, -1.1462993782e-02, -2.2381650207e-02, 1.1946151056e-01, 1.3630426101e-02, 3.6294288556e-02, 1.7602006115e-01, 4.8367745132e-02, 1.1827749460e-01, 1.8840523736e-01, 8.3973674794e-01, 4.6250714851e-01, 6.5406670102e-02],
    [2.4902557447e-02, 2.9537667670e-02, 2.6458674616e-01, 2.2927720652e-01, 1.6094265546e-04, 2.5661584954e-02, -2.7969778114e-02, 2.9376699243e-02, 1.4960500545e-01, 3.2894027535e-02, 4.9918556905e-02, 1.6081480146e-01, 1.0054468065e-01, 9.0697380314e-02, 2.2317811290e-01, 4.6250714851e-01, 1.3161522295e+00, 1.2823281199e-01],
    [1.3222599613e-01, -1.3758533343e-02, 6.4507143109e-02, 6.8311932735e-02, 1.0938845624e-05, 2.1354342979e-02, 1.2924825501e-02, 4.6318187274e-02, 3.4878112774e-02, 1.2690667367e-03, 1.1330921374e-02, 1.5054071340e-02, -1.2663102393e-02, 4.7172627159e-02, 1.1312375259e-01, 6.5406670102e-02, 1.2823281199e-01, 1.0492689215e+00],
]
# technology/growth stress pairs among the held assets
PAIRS = [(2, 3), (2, 13), (2, 15), (2, 16), (3, 13), (3, 15), (3, 16),
         (13, 15), (13, 16), (15, 16)]


def corr_from_cov(S):
    """Correlation matrix of a covariance."""
    d = [math.sqrt(S[i][i]) for i in range(N)]
    return [[S[i][j] / (d[i] * d[j]) for j in range(N)] for i in range(N)]


def cov_from_corr(S, C):
    """Covariance from a correlation matrix and the variances of S."""
    d = [math.sqrt(S[i][i]) for i in range(N)]
    return [[d[i] * d[j] * C[i][j] for j in range(N)] for i in range(N)]


def chol(S):
    """Cholesky factor G with S = G G'; returns (ok, G)."""
    G = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1):
            s = S[i][j]
            for k in range(j):
                s -= G[i][k] * G[j][k]
            if i == j:
                if s <= 0.0:
                    return False, G
                G[i][j] = math.sqrt(s)
            else:
                G[i][j] = s / G[j][j]
    return True, G


def port_var(x, S):
    """Portfolio variance x'S x."""
    v = 0.0
    for i in range(N):
        t = 0.0
        for j in range(N):
            t += S[i][j] * x[j]
        v += x[i] * t
    return v


def min_risk(G, target):
    """Min risk at a target return (SOCP): vars x, c, t, y = G'x; returns (ok, vol)."""
    with Model(N + 2, 2 * N + 2) as m:
        for j in range(N + 1):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.var_bounds(N + 1, BK.LO, 0.0, math.inf)
        for i in range(N):
            m.var_bounds(N + 2 + i, BK.FR, -math.inf, math.inf)
        m.cj(N + 1, 1.0)
        m.a_row(0, list(range(N + 1)), [1.0] * (N + 1))
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, list(range(N)), MU)
        m.con_bounds(1, BK.LO, target, math.inf)
        for i in range(N):
            m.a_row(2 + i, list(range(N)) + [N + 2 + i], [-G[j][i] for j in range(N)] + [1.0])
            m.con_bounds(2 + i, BK.FX, 0.0, 0.0)
        m.cone(CT.QUAD, [N + 1] + list(range(N + 2, 2 * N + 2)))
        r = m.solve()
    return r.rc == 0, r.objective


def max_return(G, risk):
    """Max return at a risk limit (SOCP): vars x, c, y = G'x, rb; returns (ok, return)."""
    with Model(N + 1, 2 * N + 2) as m:
        for j in range(N + 1):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        for i in range(N):
            m.var_bounds(N + 1 + i, BK.FR, -math.inf, math.inf)
        m.var_bounds(2 * N + 1, BK.FX, risk, risk)
        m.sense(SENSE.MAX)
        for j in range(N):
            m.cj(j, MU[j])
        m.a_row(0, list(range(N + 1)), [1.0] * (N + 1))
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        for i in range(N):
            m.a_row(1 + i, list(range(N)) + [N + 1 + i], [-G[j][i] for j in range(N)] + [1.0])
            m.con_bounds(1 + i, BK.FX, 0.0, 0.0)
        m.cone(CT.QUAD, [2 * N + 1] + list(range(N + 1, 2 * N + 1)))
        r = m.solve()
    return r.rc == 0, r.objective


def stress_corr(C):
    """Nearest valid stressed correlation matrix (SDP); returns (ok, C*)."""
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    NP = len(pairs)
    isgrp = [(i, j) in PAIRS or (j, i) in PAIRS for (i, j) in pairs]
    with Model() as m:
        m.add_vars(2 * NP + 1)                      # vars C* (NP), t, d (NP)
        for k in range(NP):
            if isgrp[k]:
                m.var_bounds(k, BK.LO, RHO, math.inf)
            else:
                m.var_bounds(k, BK.RA, -1.0, 1.0)
        m.var_bounds(NP, BK.LO, 0.0, math.inf)
        for k in range(NP):
            m.var_bounds(NP + 1 + k, BK.FR, -math.inf, math.inf)
        m.cj(NP, 1.0)
        esym = [m.sparsesymmat(N, [(i, i, 1.0)]) for i in range(N)]
        psym = [m.sparsesymmat(N, [(i, j, 1.0)]) for (i, j) in pairs]
        m.barvars([N])
        m.add_cons(NP + N + NP)
        for k, (i, j) in enumerate(pairs):          # d_k - sqrt2 C*_k = -sqrt2 C_k
            m.a_row(k, [NP + 1 + k, k], [1.0, -math.sqrt(2.0)])
            m.con_bounds(k, BK.FX, -math.sqrt(2.0) * C[i][j], -math.sqrt(2.0) * C[i][j])
        for i in range(N):                          # <E_ii,X> = 1
            m.baraij(NP + i, 0, [(esym[i], 1.0)])
            m.con_bounds(NP + i, BK.FX, 1.0, 1.0)
        for k in range(NP):                         # <E_ij,X> - 2 C*_k = 0
            m.baraij(NP + N + k, 0, [(psym[k], 1.0)])
            m.a_row(NP + N + k, [k], [-2.0])
            m.con_bounds(NP + N + k, BK.FX, 0.0, 0.0)
        m.cone(CT.QUAD, [NP] + list(range(NP + 1, 2 * NP + 1)))
        r = m.solve()
        if r.rc != 0:
            return False, [[0.0] * N for _ in range(N)]
        X = m.getbarxj(0)
    return True, [[X[i * N + j] for j in range(N)] for i in range(N)]


def main() -> int:
    ok = True
    R0 = 0.0
    for j in range(N):
        R0 += MU[j] * X0[j]
    var0 = port_var(X0, SIG)
    vol0 = math.sqrt(var0)
    g, G = chol(SIG); ok &= g
    g, vols_base = min_risk(G, R0); ok &= g
    g, Rrisk = max_return(G, vol0); ok &= g
    C = corr_from_cov(SIG)
    g, Cs = stress_corr(C); ok &= g
    Ss = cov_from_corr(SIG, Cs)
    volstr = math.sqrt(port_var(X0, Ss))
    mult = port_var(X0, Ss) / var0
    g, Gs = chol(Ss); ok &= g
    g, vols_str = min_risk(Gs, R0); ok &= g
    if ok:
        ok_c = all(abs(Cs[a][b] - RHO) <= 1e-3 for (a, b) in PAIRS)
        ok = (ok_c and abs(vol0 - 0.229369) < 1e-4 and abs(mult - 1.255767) < 2e-3 and abs(volstr - 0.257033) < 2e-3
              and abs(vols_base - 0.135508) < 2e-3 and abs(vols_str - 0.137191) < 2e-3 and abs(Rrisk - 0.400353) < 2e-3)
        print("big_portfolio  N=%d (%s..%s)  R0=%.6f vol0=%.6f (case 0.229369)" % (N, ASSET[0], ASSET[N - 1], R0, vol0))
        print("               group C* entries=0.85  multiplier=%.6f (case 1.255767) stressed vol=%.6f (case 0.257033)"
              % (mult, volstr))
        print("               min-risk@R0 base=%.6f stressed=%.6f (case 0.135508 / 0.137191)" % (vols_base, vols_str))
        print("               max-return@vol0=%.6f (case 0.400353)  %s" % (Rrisk, "OK" if ok else "FAIL"))
    else:
        print("big_portfolio  solve FAIL (rc)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
