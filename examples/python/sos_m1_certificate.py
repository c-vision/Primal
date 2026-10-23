#!/usr/bin/env python3
"""Exact rational SOS certificate of the axisymmetric sub-case of the "KKT-infused
flag algebra" P2 project (https://github.com/jcpaik/p2-kkt-flag-sos, MIT).

On the axisymmetric branch the energy is the bicubic C(s,t) = sum M0[i,j] s^i t^j on
[0,1]^2, and the committed rational certificate proves int int C dtau dtau >= 0 for
every probability measure tau on [0,1] via the identity
    C(s,t) = Phi(s)^T G Phi(t) + sum_beta lambda_beta B_beta(s,t),
    G = P W P^T,  Phi = (1,s,s^2,s^3),
with W >= 0 (3x3), G v* = 0 for v* = (1,1/3,1/3,1/3), lambda >= 0, and B_beta the
symmetrized corner-vanishing Handelman products s^i(1-s)^j t^k(1-t)^l.

This sample reproduces the repository's `cylinder_cert.py m1 --verify` (6/6): the
polynomial identity coefficient by coefficient, W PSD, G v* = 0, lambda >= 0 and the
corner vanishing, and reports the certified margin min(lambda_min(W), min lambda).
Like the C example it is pure arithmetic: no solver call.
Port of examples/c/sos_m1_certificate.c.
"""

import math

M0 = [[2.0 / 3, -4, 12, -10], [-4, 84, -270, 210], [12, -270, 840, -630], [-10, 210, -630, 462]]
FACE_P = [[1, 0, 0], [-3, 1, 0], [0, -1, 1], [0, 0, -1]]   # columns span v*^perp
W = [[0.666666666667, -4.634368181351, 12.634368181351],   # committed rational witness
     [-4.634368181351, 68.817301074893, -175.449591239948],
     [12.634368181351, -175.449591239948, 450.081881405003]]
HL = [(0, 5, 1, 2), (1, 2, 1, 6), (1, 2, 2, 6), (1, 2, 4, 0),   # Handelman terms (i,j),(k,l)
      (1, 2, 5, 0), (1, 2, 5, 1), (1, 2, 5, 2), (1, 2, 5, 3)]
LAM = [5.268736362702, 7.720171650511, 8.532848755941, 0.067019857594,
       13.395507417002, 20.316405933601, 17.878374617311, 8.532848755941]


def binom(n, k):
    """Binomial coefficient n choose k as a float."""
    if k < 0 or k > n:
        return 0.0
    r = 1.0
    for i in range(k):
        r = r * (n - i) / (i + 1)
    return r


def s_coef(a, j, x):
    """Coefficient of s^x in s^a (1-s)^j."""
    e = x - a
    if e < 0 or e > j:
        return 0.0
    s = binom(j, e)
    return -s if e % 2 else s


def handelman(p, m, n):
    """Coefficient of s^m t^n in the symmetrized Handelman product p."""
    f = s_coef(p[0], p[1], m) * s_coef(p[2], p[3], n)
    g = s_coef(p[2], p[3], m) * s_coef(p[0], p[1], n)
    return 0.5 * (f + g)


def handelman_value(p, s, t):
    """Value of the symmetrized Handelman basis at (s,t)."""
    f = math.pow(s, p[0]) * math.pow(1 - s, p[1]) * math.pow(t, p[2]) * math.pow(1 - t, p[3])
    g = math.pow(s, p[2]) * math.pow(1 - s, p[3]) * math.pow(t, p[0]) * math.pow(1 - t, p[1])
    return 0.5 * (f + g)


def eig3(mat):
    """Eigenvalues of a symmetric 3x3 matrix (cyclic Jacobi, as the C example)."""
    a = [row[:] for row in mat]
    for _ in range(100):
        off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2]
        if off < 1e-30:
            break
        for p in range(2):
            for q in range(p + 1, 3):
                if abs(a[p][q]) < 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(3):
                    ap, aq = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * ap - s * aq, s * ap + c * aq
                for k in range(3):
                    ap, aq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * ap - s * aq, s * ap + c * aq
    return [a[i][i] for i in range(3)]


def main() -> int:
    vstar = [1.0, 1.0 / 3, 1.0 / 3, 1.0 / 3]
    g = [[0.0] * 4 for _ in range(4)]               # G = P W P^T
    for p in range(4):
        for q in range(4):
            s = 0.0
            for r in range(3):
                for t in range(3):
                    s += FACE_P[p][r] * W[r][t] * FACE_P[q][t]
            g[p][q] = s
    nres, maxres = 0, 0.0
    for m in range(9):
        for n in range(m, 9):
            gram = g[m][n] if m <= 3 and n <= 3 else 0.0
            hand = 0.0
            for b in range(8):
                hand += LAM[b] * handelman(HL[b], m, n)
            c = M0[m][n] if m <= 3 and n <= 3 else 0.0
            maxres = max(maxres, abs(gram + hand - c))
            nres += 1
    ev = eig3(W)
    lam_min = min(LAM)
    evmin = min(ev)
    margin = evmin if evmin < lam_min else lam_min
    gv = []
    for i in range(4):
        s = 0.0
        for j in range(4):
            s += g[i][j] * vstar[j]
        gv.append(s)
    gvmax = max(abs(v) for v in gv)
    corner_ok = all(abs(handelman_value(p, cs, cd)) <= 1e-12
                    for p in HL for cs in (0.0, 1.0) for cd in (0.0, 1.0))
    lam_ok = all(v >= 0.0 for v in LAM)
    id_ok, w_ok, gv_ok = maxres < 1e-6, evmin > -1e-9, gvmax < 1e-6
    ok = id_ok and w_ok and gv_ok and lam_ok and corner_ok
    yes = lambda f: "OK" if f else "FAIL"           # noqa: E731
    print(f"sos_m1_certificate (deg 8, {nres} monomials)")
    print(f"  (i)   C = Phi^T G Phi + sum lam*B        max|res|={maxres:.3e}  {yes(id_ok)}")
    print(f"  (ii)  W PSD                              lambda_min={evmin:.6f}  {yes(w_ok)}")
    print(f"  (iii) G v* = 0                           max|={gvmax:.3e}  {yes(gv_ok)}")
    print(f"  (iv)  lambda >= 0                        min={lam_min:.6f}  {yes(lam_ok)}")
    print(f"  (v)   Handelman corner-vanishing         {yes(corner_ok)}")
    print(f"  certified margin min(lmin(W), min lam) = {margin:.6f}  ->  int int C dtau dtau >= 0  {yes(ok)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
