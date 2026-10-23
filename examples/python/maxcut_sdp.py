#!/usr/bin/env python3
"""Max-cut SDP relaxation on a complete bipartite graph.

    min (1/4) sum_{(i,j) in E} X_ij   s.t.  X_ii = 1, X >= 0

For a bipartite graph the relaxation is exact, so the optimum equals -|E|/4 (all
edges cut): -p*q/4 for K_{p,q}.

    python maxcut_sdp.py [p] [q]   (default 4 4)

Port of examples/c/maxcut_sdp.c.
"""

import sys
import time

from primalsolver import Model, BK


def main() -> int:
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    q = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    if p < 1 or q < 1:
        return 2
    n, e = p + q, p * q
    expected = -0.25 * e
    with Model() as m:
        m.add_vars(1)
        m.var_bounds(0, BK.FX, 0.0, 0.0)            # dummy scalar
        m.barvars([n])
        m.add_cons(n)                               # X_ii = 1
        # objective: coefficient 1/4 on each edge X_ij (i<p<=j): <E_ij,X> = 2 X_ij,
        # so provide val = (1/4)/2 = 1/8
        mc = m.sparsesymmat(n, [(i, j, 0.125) for i in range(p) for j in range(p, n)])
        m.barcj(0, [(mc, 1.0)])
        for i in range(n):
            me = m.sparsesymmat(n, [(i, i, 1.0)])
            m.baraij(i, 0, [(me, 1.0)])
            m.con_bounds(i, BK.FX, 1.0, 1.0)
        # valid inequalities |X_ij| <= 1 on the edges (implied by PSD + diag=1): they
        # tighten the outer LP relaxation substantially. <E_ij,X> = 2 X_ij.
        row = n
        m.add_cons(e)
        for i in range(p):
            for j in range(p, n):
                mo = m.sparsesymmat(n, [(i, j, 1.0)])
                m.baraij(row, 0, [(mo, 1.0)])
                m.con_bounds(row, BK.RA, -2.0, 2.0)
                row += 1
        t0 = time.perf_counter()
        r = m.solve()
        sec = time.perf_counter() - t0
    obj = 0.0
    ok = r.rc == 0
    if ok:
        obj = r.objective
        ok = abs(obj - expected) < 1e-4 * (1.0 + abs(expected))
    print(f"maxcut_sdp p={p} q={q} n={n} edges={e} obj={obj:.6f} ref={expected:.6f} rc={r.rc} "
          f"t={sec:.4f}s {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
