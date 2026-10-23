#!/usr/bin/env python3
"""Steering robustness of the two-qubit Werner state
(slmaquedano/semidefinite_programming_workbench, steering_robustness/src).

Given a state rho_AB and measurements {M_{a|x}} on A, the assemblage is
sigma_{a|x} = tr_A[(M_{a|x} (x) I) rho_AB]. The steering ROBUSTNESS is the minimal t
with
    tau~_{a|x} = sum_lambda D[a,x,lambda] tau~_lambda,   tau~_lambda >= 0,
    pi~_{a|x}  = tau~_{a|x} - sigma_{a|x} >= 0,          t = sum_lambda tr(tau~_lambda) - 1.
Each 2x2 Hermitian PSD variable H = [[a, c+id],[c-id, b]] is written EXACTLY as the
rotated cone (a, b, sqrt2 c, sqrt2 d) in Qr^4, so the SDP is an SOCP.

Two declared deviations from the reference (both leave the model equivalent): the
redundant no-signaling rows are omitted, and the objective is the explicit
expression sum(alpha_l + beta_l) - 1. The optimum matches the reference
dat/SR_werner_state.dat. Port of examples/c/steering_robustness.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

OA = 2
MA = 3
L = 8


def meas(x, a):
    """Measurement vector: the three mutually unbiased bases (x, a are 1-based)."""
    s = 1.0 / math.sqrt(2.0)
    if x == 1:
        return [complex(s, 0), complex(s if a == 1 else -s, 0)]
    if x == 2:
        return [complex(s, 0), complex(0, s if a == 1 else -s)]
    return [complex(1 if a == 1 else 0, 0), complex(1 if a == 2 else 0, 0)]


def werner(w):
    """Werner state p|Phi+><Phi+| + (1-p) I/4 with p = w."""
    r = [[0j] * 4 for _ in range(4)]
    r[0][0] = r[3][3] = complex((1.0 - w) / 4, 0)
    r[1][1] = r[2][2] = complex((1.0 + w) / 4, 0)
    r[1][2] = r[2][1] = complex(-w / 2, 0)
    return r


def assemblage(w, x, a):
    """sigma_{a|x} = tr_A[(M M^dag (x) I) rho]."""
    rho, m = werner(w), meas(x + 1, a + 1)
    p = [[m[i] * m[j].conjugate() for j in range(2)] for i in range(2)]
    sig = [[0j, 0j], [0j, 0j]]
    for ib in range(2):
        for jb in range(2):
            s = 0j
            for ia in range(2):
                for ma in range(2):
                    s += p[ia][ma] * rho[2 * ma + ib][2 * ia + jb]
            sig[ib][jb] = s
    return sig


def dstrategy(a, x, l):
    """True when deterministic strategy l answers a to setting x."""
    return ((l >> x) & 1) == a


def sr(w):
    """Steering robustness at Werner parameter w, as an SOCP over Qr^4; returns (ok, value)."""
    sig = {}                                        # constants as (a, b, sqrt2 Re c, sqrt2 Im c)
    for x in range(MA):
        for a in range(OA):
            s = assemblage(w, x, a)
            sig[(a, x)] = [s[0][0].real, s[1][1].real, math.sqrt(2.0) * s[0][1].real,
                           math.sqrt(2.0) * s[0][1].imag]
    tau, pi = 0, 4 * L
    nv = pi + 4 * OA * MA
    with Model() as m:
        m.add_vars(nv)
        m.sense(SENSE.MIN)
        m.add_cons(4 * OA * MA)                     # pi link rows only
        for l in range(L):
            m.var_bounds(tau + 4 * l, BK.LO, 0.0, math.inf)
        for i in range(4 * OA * MA):
            m.var_bounds(pi + i, BK.FR, -math.inf, math.inf)
        r = 0
        for x in range(MA):
            for a in range(OA):
                piv = 4 * (a * MA + x)
                for cp in range(4):
                    ls = [l for l in range(L) if dstrategy(a, x, l)]
                    m.a_row(r, [pi + piv + cp] + [tau + 4 * l + cp for l in ls], [1.0] + [-1.0] * len(ls))
                    m.con_bounds(r, BK.FX, -sig[(a, x)][cp], -sig[(a, x)][cp])
                    r += 1
        for x in range(MA):
            for a in range(OA):
                base = pi + 4 * (a * MA + x)
                m.cone(CT.RQUAD, [base, base + 1, base + 2, base + 3])
        for l in range(L):
            m.cone(CT.RQUAD, [tau + 4 * l, tau + 4 * l + 1, tau + 4 * l + 2, tau + 4 * l + 3])
        for l in range(L):
            m.cj(tau + 4 * l, 1.0)
            m.cj(tau + 4 * l + 1, 1.0)
        m.cfix(-1.0)
        res = m.solve()
    return (res.rc == 0), (res.objective if res.rc == 0 else -1.0)


def main() -> int:
    print("steering_robustness (Werner state)")
    ws = [0.0, 0.4, 0.5, 0.58, 0.8, 0.9, 1.0]
    want = [0.0000, 0.0000, 0.0000, 0.0017, 0.1412, 0.2046, 0.2679]
    bad = 0
    for w, ref in zip(ws, want):
        ok, v = sr(w)
        good = ok and v > -1e-6 and abs(v - ref) < 1e-3
        if not good:
            bad += 1
        print(f"  w={w:.2f}  SR={v:.4f}  (ref {ref:.4f})  {'OK' if good else 'FAIL'}")
    print("OK" if bad == 0 else "FAIL")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
