#!/usr/bin/env python3
"""PPT and CCNR criteria for bipartite entanglement (y1-zhu/quantum-correlations,
"Certifying quantum separability with adaptive polytopes", SciPost Phys. 16, 063).

  - PPT (Peres-Horodecki): t*rho + (1-t)/d^2 I stays PPT while
    t*rho^G + (1-t)/d^2 I >= 0 (rho^G the partial transpose). The largest such t is
    the SDP  max t  s.t.  (1/d^2) I + t C >= 0,  C = rho^G - (1/d^2) I, equal to the
    closed form t* = 1 / (1 - d^2 * lambda_min(rho^G)).
  - CCNR (realignment): for a separable state the realigned matrix R(rho) has
    nuclear norm <= 1, so the white-noise robustness is where ||R(mixture)||_1 = 1.

Instance: the qutrit isotropic state rho_p = p |Phi+><Phi+| + (1-p) I/9: both
robustnesses are 1/(4p), and at p = 1/4 both criteria give exactly 1. The SDP uses
one 9x9 semidefinite bar variable; every number is cross-checked against the closed
forms and Jacobi eigenvalues (the library's routine, as in the C example).
Port of examples/c/quantum_separability.c.
"""

import ctypes
import math

import primalsolver as ps
from primalsolver import Model, BK, SENSE

D = 3
N = D * D                                           # 9
UPLO_UP = 1


def build_state(p):
    """rho_p and its partial transpose on B: rho^G[(i,k),(j,l)] = rho[(i,l),(j,k)]."""
    rho = [0.0] * (N * N)
    for a in range(D):
        for b in range(D):
            for c in range(D):
                for d in range(D):
                    v = (1.0 - p) / (D * D) if (a == c and b == d) else 0.0
                    if a == b and c == d:
                        v += p / D
                    rho[(a * D + b) * N + (c * D + d)] = v
    return rho, partial_transpose(rho)


def partial_transpose(mat):
    out = [0.0] * (N * N)
    for a in range(D):
        for k in range(D):
            for j in range(D):
                for l in range(D):
                    out[(a * D + k) * N + (j * D + l)] = mat[(a * D + l) * N + (j * D + k)]
    return out


def eigenvalues(mat):
    """Eigenvalues of a real symmetric N x N matrix, by the library's Jacobi routine."""
    ev = (ctypes.c_double * N)()
    buf = (ctypes.c_double * (N * N))(*mat)
    try:
        ps.LIB.dmat_eig_jacobi(ctypes.c_int(N), buf, ev, (ctypes.c_double * (N * N))())
    except AttributeError:                          # a build that hides internal symbols
        ps.PRIMAL_syeig(None, UPLO_UP, N, buf, ev)
    return list(ev)


def lambda_min(mat):
    return min(eigenvalues(mat))


def nuclear_norm(a):
    """Sum of singular values of a real N x N matrix, via the eigenvalues of A A^T."""
    m = [0.0] * (N * N)
    for i in range(N):
        for j in range(N):
            s = 0.0
            for k in range(N):
                s += a[i * N + k] * a[j * N + k]
            m[i * N + j] = s
    s = 0.0
    for e in eigenvalues(m):
        s += math.sqrt(e if e > 0.0 else 0.0)
    return s


def realign(rho):
    """CCNR: realigned state R[(i,j),(k,l)] = rho[(i,k),(j,l)]."""
    out = [0.0] * (N * N)
    for i in range(D):
        for j in range(D):
            for k in range(D):
                for l in range(D):
                    out[(i * D + j) * N + (k * D + l)] = rho[(i * D + k) * N + (j * D + l)]
    return out


def ppt_robustness_sdp(rhot):
    """max t  s.t.  S = (1/9) I + t C >= 0, one equality row per lower-triangle entry."""
    with Model() as m:
        m.add_vars(1)                               # scalar t
        m.var_bounds(0, BK.RA, 0.0, 1.0)
        m.barvars([N])
        m.add_cons(N * (N + 1) // 2)
        row = 0
        for l1 in range(N):
            for l2 in range(l1 + 1):
                c = rhot[l1 * N + l2] - (1.0 / (D * D) if l1 == l2 else 0.0)
                sym = m.sparsesymmat(N, [(l1, l2, 1.0 if l1 == l2 else 0.5)])   # <m,S> = S_l1l2
                m.baraij(row, 0, [(sym, 1.0)])
                m.a_row(row, [0], [-c])
                rhs = 1.0 / (D * D) if l1 == l2 else 0.0
                m.con_bounds(row, BK.FX, rhs, rhs)
                row += 1
        m.cj(0, 1.0)
        m.sense(SENSE.MAX)
        r = m.solve()
    return (r.rc == 0), (r.x[0] if r.rc == 0 else 0.0)


def main() -> int:
    p, pct = 0.5, 0.25                              # entangled state, PPT separability threshold
    rho, rhot = build_state(p)
    lmin = lambda_min(rhot)                         # eigenvalues of rho^G: closed form vs Jacobi
    lmin_cf = (1.0 - 4.0 * p) / (D * D)
    ok_sdp, t_sdp = ppt_robustness_sdp(rhot)        # PPT robustness: SDP vs closed form
    t_cf = 1.0 / (1.0 - (D * D) * lmin)
    rmat = realign(rho)                             # CCNR: realignment norm and robustness
    ccnr = nuclear_norm(rmat)
    ccnr_cf = (1.0 + 8.0 * p) / 3.0
    # R(t rho + (1-t) I/9) = t Rmat + (1-t)/9 e e^T, e e^T[(i,j),(k,l)] = delta_ij delta_kl
    eet = [1.0 if (i == j and k == l) else 0.0 for i in range(D) for j in range(D)
           for k in range(D) for l in range(D)]
    t_ccnr = 1.0
    for k in range(10001):
        t = k * 1e-4
        rm = [t * rmat[q] + (1.0 - t) * (1.0 / (D * D)) * eet[q] for q in range(N * N)]
        if nuclear_norm(rm) > 1.0:
            break
        t_ccnr = t
    # witness W = I - |Phi~><Phi~|: W^G = I - SWAP >= 0 and tr(W rho) = 2/3 - 8p/3 < 0
    w = [(1.0 if (a == c and b == d) else 0.0) - (1.0 if (a == b and c == d) else 0.0)
         for a in range(D) for b in range(D) for c in range(D) for d in range(D)]
    wmin = lambda_min(partial_transpose(w))
    wtr = 0.0
    for a in range(N):
        for b in range(N):
            wtr += w[a * N + b] * rho[b * N + a]
    wtr_cf = 2.0 / 3.0 - 8.0 * p / 3.0
    rho_t, rhot_t = build_state(pct)                # thresholds: at p=1/4 both criteria are 1
    t_thr = 1.0 / (1.0 - (D * D) * lambda_min(rhot_t))
    ccnr_t = nuclear_norm(realign(rho_t))
    ok = (ok_sdp and abs(t_sdp - t_cf) < 1e-5 and abs(lmin - lmin_cf) < 1e-9 and abs(ccnr - ccnr_cf) < 1e-8
          and abs(t_ccnr - t_cf) < 1e-3 and wmin > -1e-9 and wtr < -1e-6 and abs(wtr - wtr_cf) < 1e-9
          and abs(t_thr - 1.0) < 1e-8 and abs(ccnr_t - 1.0) < 1e-8)
    print(f"quantum_separability  qutrit isotropic  p={p:.2f}  (p_thr={pct:.2f})")
    print(f"  lambda_min(rho^T) = {lmin:.10f}   (closed form {lmin_cf:.10f})")
    print(f"  PPT robustness: SDP t*={t_sdp:.10f}   closed form 1/(4p)={t_cf:.10f}   ok={int(ok_sdp)}")
    print(f"  CCNR ||R(rho)||_1 = {ccnr:.10f}   (closed form (1+8p)/3={ccnr_cf:.10f})")
    print(f"  CCNR robustness t*={t_ccnr:.10f}   PPT robustness t*={t_cf:.10f}")
    print(f"  witness W=I-|Phi~><Phi~|: lambda_min(W^T)={wmin:.3e}, tr(W rho)={wtr:.10f} "
          f"(closed form {wtr_cf:.10f})")
    print(f"  threshold p=1/4: PPT t*={t_thr:.10f}, CCNR ||R||={ccnr_t:.10f} (both = 1)")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
