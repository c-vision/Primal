#!/usr/bin/env python3
"""PrimalSolver - minimum-variance portfolio under a factor risk model (QP).

Port of c_examples/finance/factor_model.c. Sigma = F F' + diag(D) is built
explicitly and the objective is the QP 1/2 w'Qw with Q = 2 Sigma.

Hand case: N=3, F=(1,1,0), D=(0,0,1) -> Sigma=[[1,1,0],[1,1,0],[0,0,1]];
min w'Sigma w s.t. e'w=1 is 1/2 at w2=1/2 (any split of w0+w1=1/2).

    python python_examples/factor_model.py
"""

from primalsolver import Model, BK, SOLSTA

F = [1.0, 1.0, 0.0]
D = [0.0, 0.0, 1.0]
N = 3
SIGMA = [[F[i] * F[j] + (D[i] if i == j else 0.0) for j in range(N)] for i in range(N)]


def main() -> int:
    with Model(maxcon=1, maxvar=N) as m:
        for j in range(N):
            m.var_bounds(j, BK.FR)
        m.a_row(0, list(range(N)), [1.0] * N)
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.qobj([(i, j, 2.0 * SIGMA[i][j]) for i in range(N) for j in range(i, N)
                if SIGMA[i][j] != 0.0])
        r = m.solve()

    w = r.x
    quad = sum(w[i] * SIGMA[i][j] * w[j] for i in range(N) for j in range(N))
    ok = (r.solsta == SOLSTA.OPTIMAL and abs(r.objective - 0.5) < 1e-7
          and abs(w[2] - 0.5) < 1e-6 and abs(w[0] + w[1] - 0.5) < 1e-6
          and abs(sum(w) - 1.0) < 1e-6 and abs(quad - r.objective) < 1e-7)
    print(f"w = ({w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}) obj = {r.objective:.6f} "
          f"(0.5), w'Sigma w = {quad:.6f} {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
