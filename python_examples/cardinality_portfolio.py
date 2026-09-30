#!/usr/bin/env python3
"""PrimalSolver - minimum variance with a cardinality constraint (MIQP).

Port of c_examples/finance/cardinality_portfolio.c:

    min w'Sigma w  s.t. e'w = 1, w >= 0, sum z <= K, w_j <= z_j, z binary.

Hand case: Sigma = I, N = 3, K = 2 -> optimum (1/2,1/2,0), value 1/2.

    python python_examples/cardinality_portfolio.py
"""

from primalsolver import Model, BK, VAR_TYPE, SOLSTA

N, K = 3, 2


def main() -> int:
    with Model(maxcon=5, maxvar=6) as m:     # 0..2 w, 3..5 z
        for j in range(N):
            m.var_bounds(j, BK.LO, 0.0, 1.0)
            m.var_bounds(3 + j, BK.RA, 0.0, 1.0)
            m.var_type(3 + j, VAR_TYPE.INT_BIN)
            m.a_row(j, [j, 3 + j], [1.0, -1.0])          # w_j - z_j <= 0
            m.con_bounds(j, BK.UP, up=0.0)
        m.a_row(3, [0, 1, 2], [1.0, 1.0, 1.0]); m.con_bounds(3, BK.FX, 1.0, 1.0)
        m.a_row(4, [3, 4, 5], [1.0, 1.0, 1.0]); m.con_bounds(4, BK.UP, up=float(K))
        m.qobj([(j, j, 2.0) for j in range(N)])          # Q = 2I
        r = m.solve()

    x = r.x
    used = sum(1 for j in range(N) if x[j] > 1e-6)
    quad = sum(x[j] * x[j] for j in range(N))
    ok = (r.solsta in (SOLSTA.OPTIMAL, SOLSTA.INTEGER_OPTIMAL)
          and abs(r.objective - 0.5) < 1e-6 and used <= K
          and abs(sum(x[:N]) - 1.0) < 1e-6 and abs(quad - r.objective) < 1e-6)
    print(f"w = ({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}) active={used} "
          f"obj = {r.objective:.6f} (0.5) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
