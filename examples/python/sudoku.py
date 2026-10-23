#!/usr/bin/env python3
"""MILP Sudoku solver (MOSEK Cookbook 11.7, sudoku.cc).

Binary X[i][j][k] = 1 iff cell (i,j) holds digit k+1. Constraints: exactly one digit
per cell, each digit once per row, once per column, once per block, plus the givens.
Pure feasibility (objective 0).

The MOSEK example is 9x9 (729 binaries), which does not close with this library's
branch-and-bound today, so the same model is solved on a 4x4 board (M=2, 64
binaries). The check: the returned grid is a valid Sudoku and respects every given.
Port of examples/c/sudoku.c.
"""

from primalsolver import Model, BK, SENSE, VAR_TYPE

M = 2
N = M * M                                           # 4
NVAR = N * N * N                                    # 64
# givens (1-based) (row, col, digit)
HR = [(1, 1, 1), (1, 3, 3), (2, 1, 3), (2, 4, 2), (3, 1, 2), (3, 4, 3), (4, 2, 3), (4, 4, 1)]


def idx(i, j, k):
    return (i * N + j) * N + k


def valid(digits):
    return sorted(digits) == list(range(1, N + 1))


def main() -> int:
    nrow = 4 * N * N + len(HR)                      # 64 + 8 = 72
    rows = []
    rows += [[idx(i, j, k) for k in range(N)] for i in range(N) for j in range(N)]   # one digit per cell
    rows += [[idx(i, j, k) for j in range(N)] for i in range(N) for k in range(N)]   # digit once per row
    rows += [[idx(i, j, k) for i in range(N)] for j in range(N) for k in range(N)]   # digit once per column
    rows += [[idx(bi * M + di, bj * M + dj, k) for di in range(M) for dj in range(M)]
             for bi in range(M) for bj in range(M) for k in range(N)]                # digit once per block
    rows += [[idx(i - 1, j - 1, k - 1)] for i, j, k in HR]                           # givens
    with Model(maxcon=nrow, maxvar=NVAR) as m:
        for v in range(NVAR):
            m.var_bounds(v, BK.RA, 0.0, 1.0)
            m.var_type(v, VAR_TYPE.INT_BIN)
            m.cj(v, 0.0)
        m.sense(SENSE.MIN)
        for row, sub in enumerate(rows):
            m.a_row(row, sub, [1.0] * len(sub))
            m.con_bounds(row, BK.FX, 1.0, 1.0)
        r = m.solve()
    grid = [[0] * N for _ in range(N)]
    good = r.rc == 0
    if good:
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if r.x[idx(i, j, k)] > 0.5:
                        grid[i][j] = k + 1
        good = (all(valid(grid[i]) for i in range(N))
                and all(valid([grid[i][j] for i in range(N)]) for j in range(N))
                and all(valid([grid[bi * M + di][bj * M + dj] for di in range(M) for dj in range(M)])
                        for bi in range(M) for bj in range(M))
                and all(grid[i - 1][j - 1] == k for i, j, k in HR))
    print(f"sudoku: rc={r.rc}  grid {'valid and givens respected' if good else 'NOT valid'}")
    if good:
        for i in range(N):
            print("  " + " ".join(str(d) for d in grid[i]))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
