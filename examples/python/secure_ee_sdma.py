#!/usr/bin/env python3
"""Secure energy efficiency of a downlink SDMA beamformer (rezarhp/Secure-EE-SDMA,
Hashempour et al., ICC Workshops 2026, arXiv:2603.21889).

The repository solves the transmit-precoder subproblem with an OUTER Dinkelbach
loop on the secure energy efficiency SEE = zeta / (P0 + ||p||^2), zeta the minimum
secrecy rate, and an INNER successive-convex-approximation (SCA) loop for the two
non-convex ratios. This sample is a faithful deterministic instance: one user, one
untrusted energy-harvesting eavesdropper, a two-antenna BS, real channels, the RIS at
the zero-phase reflection. The SCA surrogates are affine; the exact cones are
|h_e^H p|^2 <= sigma^2 rho_e (RQUAD), 1 + rho_u >= 2^f_p (PEXP), ||p||^2 <= Pmax and
the harvesting constraint |h_e^H p|^2 >= E_req. The solution is validated against a
dense brute-force search over (p0,p1) of the TRUE objective.
Port of examples/c/secure_ee_sdma.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

HU = [1.0, 0.5]                                     # legitimate-user channel
HE = [0.3, 1.0]                                     # eavesdropper/harvester channel
SIGMA2 = 1.0
PMAX = 2.0
PCIRC = 1.0                                         # circuit power P0
EH = 0.01                                           # harvested DC energy [W]
(PV0, PV1, Y, Z, FP, FE, RU, RE, OPRU, LFP, ZE, PW, HALF, ONE, TPS, HALFSIG, NVAR) = range(17)
(R_ZE, R_OBJ, R_SEC, R_EXP, R_GAM, R_USR, R_HARV, R_LFP, NR) = range(9)


def sai_inverse(x):
    """The repository's inverse of the non-linear harvesting map."""
    b0, b1, phi, k1 = 150.0, 0.014, 0.024, 1.0
    k2 = phi / (k1 * (1.0 + math.exp(b0 * b1)))
    return b1 - math.log(phi / (k1 * (x + k2)) - 1.0) / b0


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def evaluate(p):
    """True secrecy rate and energy efficiency at a precoder p."""
    hu, he = dot(HU, p), dot(HE, p)
    gu, ge = hu * hu / SIGMA2, he * he / SIGMA2
    r = math.log(1.0 + gu) / math.log(2.0) - math.log(1.0 + ge) / math.log(2.0)
    return r, r / (PCIRC + dot(p, p))


def solve_sca(p_old, rho_u_old, f_e_old, eta):
    """One SCA step around the previous point; returns (ok, p, zeta)."""
    with Model(maxcon=NR, maxvar=NVAR) as m:
        for j in range(NVAR):
            m.var_bounds(j, BK.FR, -math.inf, math.inf)
        for j in (Z, FP, FE, PW):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.var_bounds(RU, BK.LO, 1e-9, math.inf)
        m.var_bounds(RE, BK.LO, 1e-9, math.inf)
        m.var_bounds(HALF, BK.FX, 0.5, 0.5)
        m.var_bounds(ONE, BK.FX, 1.0, 1.0)
        m.var_bounds(TPS, BK.FX, math.sqrt(PMAX), math.sqrt(PMAX))
        m.var_bounds(HALFSIG, BK.FX, 0.5 * SIGMA2, 0.5 * SIGMA2)
        hu_old, he_old = dot(HU, p_old), dot(HE, p_old)
        a, ln2 = math.pow(2.0, f_e_old), math.log(2.0)
        m.a_row(R_ZE, [ZE, PV0, PV1], [1.0, -HE[0], -HE[1]])          # z_e = h_e^T p
        m.con_bounds(R_ZE, BK.FX, 0.0, 0.0)
        m.a_row(R_OBJ, [Y, Z, PW], [1.0, -1.0, eta])                   # EE surrogate, pw >= ||p||^2
        m.con_bounds(R_OBJ, BK.UP, -math.inf, -eta * PCIRC)
        m.a_row(R_SEC, [Z, FP, FE], [1.0, -1.0, 1.0])                  # zeta: minimum secrecy rate
        m.con_bounds(R_SEC, BK.UP, -math.inf, 0.0)
        m.a_row(R_EXP, [OPRU, RU], [1.0, -1.0])                        # onepru = 1 + rho_u
        m.con_bounds(R_EXP, BK.FX, 1.0, 1.0)
        # 1 + rho_e <= A(1 + ln2 (f_e - f_e_old))
        m.a_row(R_GAM, [RE, FE], [1.0, -a * ln2])
        m.con_bounds(R_GAM, BK.UP, -math.inf, a - 1.0 - a * ln2 * f_e_old)
        # user SCA: 2 hu_old (h_u^T p)/rho_u_old - (hu_old^2/rho_u_old^2) rho_u >= sigma^2
        m.a_row(R_USR, [PV0, PV1, RU], [2.0 * hu_old * HU[0] / rho_u_old, 2.0 * hu_old * HU[1] / rho_u_old,
                                        -hu_old * hu_old / (rho_u_old * rho_u_old)])
        m.con_bounds(R_USR, BK.LO, SIGMA2, math.inf)
        # harvesting SCA: 2 he_old (h_e^T p) - he_old^2 >= E_req
        m.a_row(R_HARV, [PV0, PV1], [2.0 * he_old * HE[0], 2.0 * he_old * HE[1]])
        m.con_bounds(R_HARV, BK.LO, sai_inverse(EH) + he_old * he_old, math.inf)
        m.a_row(R_LFP, [LFP, FP], [1.0, -ln2])                         # exponential-cone argument
        m.con_bounds(R_LFP, BK.FX, 0.0, 0.0)
        m.cone(CT.PEXP, [OPRU, ONE, LFP])
        m.cone(CT.RQUAD, [RE, HALFSIG, ZE])
        m.cone(CT.RQUAD, [PW, HALF, PV0, PV1])
        m.cone(CT.QUAD, [TPS, PV0, PV1])
        m.cj(Y, 1.0)
        m.sense(SENSE.MAX)
        r = m.solve()
    if r.rc != 0:
        return False, None, 0.0
    return True, [r.x[PV0], r.x[PV1]], r.x[Z]


def main() -> int:
    ereq = sai_inverse(EH)
    nrm = math.sqrt(dot(HU, HU))                    # MRT initialization toward the user
    p = [math.sqrt(PMAX) * HU[0] / nrm, math.sqrt(PMAX) * HU[1] / nrm]
    eta = ee_actual = rate_actual = 0.0             # outer Dinkelbach, inner SCA
    outer = inner = 0
    for outer in range(50):
        y_prev = -math.inf
        for inner in range(50):
            hu, he = dot(HU, p), dot(HE, p)
            rho_u_old = hu * hu / SIGMA2
            f_e_old = math.log(1.0 + he * he / SIGMA2) / math.log(2.0)
            ok, pn, zeta = solve_sca(p, rho_u_old, f_e_old, eta)
            if not ok:
                print("secure_ee_sdma  SCA subproblem rc!=OK FAIL")
                return 1
            p = pn
            y = zeta - eta * (PCIRC + dot(p, p))
            if abs(y - y_prev) < 1e-9:
                break
            y_prev = y
        rate_actual, ee_actual = evaluate(p)
        if abs(ee_actual - eta) < 1e-9:
            eta = ee_actual
            break
        eta = ee_actual
    power = dot(p, p)
    harvest = dot(HE, p) * dot(HE, p)

    best_ee, best_p0, best_p1 = -1e300, 0.0, 0.0    # dense brute force of the TRUE objective
    span, ng = math.sqrt(PMAX), 1200
    for a in range(ng + 1):
        p0 = -span + 2.0 * span * a / ng
        for b in range(ng + 1):
            p1 = -span + 2.0 * span * b / ng
            q = [p0, p1]
            if dot(q, q) > PMAX:
                continue
            he = dot(HE, q)
            if he * he < ereq:
                continue
            ee = evaluate(q)[1]
            if ee > best_ee:
                best_ee, best_p0, best_p1 = ee, p0, p1
    ok = abs(ee_actual - best_ee) < 1e-3 * (1.0 + abs(best_ee)) and power <= PMAX + 1e-7 and harvest >= ereq - 1e-7
    print("secure_ee_sdma  K=1 J=1 Nt=2")
    print(f"  E_req = sai_inverse({EH:.3f}) = {ereq:.6f}")
    print(f"  p = ({p[0]:.6f}, {p[1]:.6f})  ||p||^2 = {power:.6f} <= {PMAX:.3f}")
    print(f"  secrecy rate = {rate_actual:.6f} bit/s/Hz   harvested = {harvest:.6f} >= {ereq:.6f}")
    print(f"  SEE = {ee_actual:.6f} bit/J/Hz   [brute force {best_ee:.6f} at ({best_p0:.4f},{best_p1:.4f})]")
    print(f"  outer={outer + 1} inner={inner + 1}")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
