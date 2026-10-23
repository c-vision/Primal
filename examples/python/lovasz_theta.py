#!/usr/bin/env python3
"""Maximum-clique upper bound via the Lovasz theta SDP
(Rudolfovoorg/Improving_Upper_Bounds_of_MCP_using_Reduction_Rules).

The SDP bound is the Lovasz theta number of the COMPLEMENT,

    theta(Gbar) = max <J, X>   s.t.  tr(X) = 1,  X_ij = 0 for every non-edge (i,j),  X >= 0,

an upper bound on the clique number: omega(G) <= theta(Gbar) <= chi(G).
Hand-derived instances: C5 -> sqrt(5) (omega 2), K3 -> 3, E5 (no edges) -> 1.
The sample checks the value, the primal X (tr = 1, X PSD, X_ij = 0 on non-edges)
and the integer bound floor(theta) against omega. Port of examples/c/lovasz_theta.c.
"""

import ctypes
import math

import primalsolver as ps
from primalsolver import Model, BK, SENSE

UPLO_UP = 1


def theta_sdp(n, adj):
    """theta(Gbar) for a graph on n vertices; returns (ok, theta, X row-major)."""
    nonedges = [(i, j) for i in range(n) for j in range(i + 1, n) if not adj[i][j]]
    with Model() as m:
        m.barvars([n])
        m.add_cons(1 + len(nonedges))
        jterms = []                                 # objective matrix J (all ones), lower triangle
        for i in range(n):
            jterms.append((i, i, 1.0))
            jterms += [(i, j, 1.0) for j in range(i)]
        m.barcj(0, [(m.sparsesymmat(n, jterms), 1.0)])
        tr = m.sparsesymmat(n, [(i, i, 1.0) for i in range(n)])   # row 0: tr(X) = 1
        m.baraij(0, 0, [(tr, 1.0)])
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        for row, (i, j) in enumerate(nonedges, start=1):
            e = m.sparsesymmat(n, [(j, i, 0.5)])    # <E_ij,X> = X_ij
            m.baraij(row, 0, [(e, 1.0)])
            m.con_bounds(row, BK.FX, 0.0, 0.0)
        m.sense(SENSE.MAX)
        r = m.solve()
        if r.rc != 0:
            return False, 0.0, None
        return True, r.objective, m.getbarxj(0)


def min_eig(n, x):
    """Minimum eigenvalue of the n x n solution matrix (PSD check), with the C
    example's Jacobi routine on X as returned (PRIMAL_syeig if it is hidden)."""
    ev = (ctypes.c_double * n)()
    xbuf = (ctypes.c_double * (n * n))(*x)
    try:
        ps.LIB.dmat_eig_jacobi(ctypes.c_int(n), xbuf, ev, (ctypes.c_double * (n * n))())
    except AttributeError:
        ps.PRIMAL_syeig(None, UPLO_UP, n, xbuf, ev)
    return min(ev)


def check_x(n, adj, x, tol):
    """tr(X) = 1, X PSD, X_ij = 0 on non-edges."""
    if abs(sum(x[i * n + i] for i in range(n)) - 1.0) > 1e-6:
        return False
    if min_eig(n, x) < -tol:
        return False
    return all(adj[i][j] or abs(x[i * n + j]) <= tol for i in range(n) for j in range(i + 1, n))


def main() -> int:
    c5 = [[1 if (j - i) % 5 in (1, 4) else 0 for j in range(5)] for i in range(5)]
    k3 = [[1 if i != j else 0 for j in range(3)] for i in range(3)]
    e5 = [[0] * 5 for _ in range(5)]
    ok5, th5, x = theta_sdp(5, c5)
    x5 = int(ok5 and check_x(5, c5, x, 1e-6))
    ok3, th3, x = theta_sdp(3, k3)
    x3 = int(ok3 and check_x(3, k3, x, 1e-6))
    oke, the, x = theta_sdp(5, e5)
    xe = int(oke and check_x(5, e5, x, 1e-6))
    fl = [int(math.floor(v + 1e-6)) for v in (th5, th3, the)]
    ok = (x5 and x3 and xe and abs(th5 - math.sqrt(5.0)) < 1e-6 and abs(th3 - 3.0) < 1e-6
          and abs(the - 1.0) < 1e-6 and fl == [2, 3, 1])
    print("lovasz_theta  max-clique upper bound (theta of the complement)")
    print(f"  C5 (omega=2): theta={th5:.10f}   expected sqrt(5)={math.sqrt(5.0):.10f}   X ok={x5}")
    print(f"  K3 (omega=3): theta={th3:.10f}   expected 3              X ok={x3}")
    print(f"  E5 (omega=1): theta={the:.10f}   expected 1              X ok={xe}")
    print(f"  integer bound floor(theta): C5->{fl[0]}, K3->{fl[1]}, E5->{fl[2]}")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
