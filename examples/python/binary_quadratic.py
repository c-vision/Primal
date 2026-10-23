#!/usr/bin/env python3
"""Binary quadratic problem via its Shor SDP relaxation (MOSEK/Tutorials,
binary-quadratic/binquad.py).

    minimize  x'Qx + c'x + R      over x in {0,1}^n

The Shor relaxation lifts X = x x' to an (n+1)x(n+1) matrix
    Z = [ X  x ; x' 1 ] >= 0,   diag(X) = x,   Z[n,n] = 1
and minimizes R + c'x + <Q,X>. The sample checks it is a valid lower bound against
a brute force over the 2^n binary points, and that Z is PSD (its eigenvalues come
from the library's Jacobi eigensolver, as in the C example).

Instance n=4: Q = tridiag(-2, 2, -2), c = 0, R = 0 -> optimum x = (1,1,1,1), -4.
Port of examples/c/binary_quadratic.c.
"""

import ctypes
import itertools

import primalsolver as ps
from primalsolver import Model, BK, SENSE

N = 4
Q = [[2, -2, 0, 0], [-2, 2, -2, 0], [0, -2, 2, -2], [0, 0, -2, 2]]
C = [0, 0, 0, 0]
UPLO_UP = 1


def fval(x):
    """Objective value of the binary quadratic at the binary point x."""
    return sum(Q[i][j] * x[i] * x[j] for i in range(N) for j in range(N)) + sum(C[i] * x[i] for i in range(N))


def shor():
    """Shor SDP relaxation; returns (ok, value, Z row-major, min eigenvalue of Z)."""
    dim = N + 1
    with Model() as m:
        m.barvars([dim])
        m.add_cons(dim)                             # N diag = last column, 1 last diag = 1
        # objective symmat: Q on the top-left block, c/2 on the (i,n) entries
        terms = [(i, i, Q[i][i]) for i in range(N)]
        terms += [(i, j, Q[i][j]) for i in range(N) for j in range(i + 1, N)]
        terms += [(i, N, 0.5 * C[i]) for i in range(N)]
        m.barcj(0, [(m.sparsesymmat(dim, terms), 1.0)])
        for i in range(N):                          # Z_ii - Z_iN = 0
            m1 = m.sparsesymmat(dim, [(i, i, 1.0)])
            m2 = m.sparsesymmat(dim, [(i, N, 0.5)])
            m.baraij(i, 0, [(m1, 1.0), (m2, -1.0)])
            m.con_bounds(i, BK.FX, 0.0, 0.0)
        mn = m.sparsesymmat(dim, [(N, N, 1.0)])
        m.baraij(N, 0, [(mn, 1.0)])
        m.con_bounds(N, BK.FX, 1.0, 1.0)
        m.sense(SENSE.MIN)
        r = m.solve()
        if r.rc != 0:
            return False, 0.0, [0.0] * (dim * dim), 0.0
        z = m.getbarxj(0)
        ev = (ctypes.c_double * dim)()
        zbuf = (ctypes.c_double * (dim * dim))(*z)
        try:
            # the C example's own routine (linalg.h), on Z exactly as returned: the
            # solved Z is symmetric only to ~1e-9, so the ~1e-9 minimum eigenvalue
            # depends on the triangle a symmetrizing eigensolver would read
            ps.LIB.dmat_eig_jacobi(ctypes.c_int(dim), zbuf, ev, (ctypes.c_double * (dim * dim))())
        except AttributeError:                      # a build that hides internal symbols
            ps.PRIMAL_syeig(m.env, UPLO_UP, dim, zbuf, ev)
    return True, r.objective, z, min(ev)


def main() -> int:
    ok, sdp, _, mineig = shor()
    best, bx = 1e300, [0] * N                       # exact optimum by brute force over 2^N
    for bits in itertools.product((0, 1), repeat=N):
        x = list(reversed(bits))                    # same visiting order as the C bit mask
        f = fval(x)
        if f < best:
            best, bx = f, x
    okall = ok and sdp <= best + 1e-6 and mineig > -1e-7
    print(f"binary_quadratic  n={N}")
    print(f"  Shor SDP lower bound = {sdp:.8f}   (Z PSD, min_eig={mineig:.2e})")
    print(f"  exact optimum        = {best:.8f}   at x=({bx[0]},{bx[1]},{bx[2]},{bx[3]})")
    print(f"  gap = {best - sdp:.8f}")
    print("OK" if okall else "FAIL")
    return 0 if okall else 1


if __name__ == "__main__":
    raise SystemExit(main())
