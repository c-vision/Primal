#!/usr/bin/env julia
# sudoku.jl — MILP Sudoku solver (MOSEK Cookbook 11.7). Binary X[i][j][k] = 1
# iff cell (i,j) holds digit k+1: one digit per cell, each digit once per row,
# column and block, plus the givens (pure feasibility). Solved on a 4x4 board
# (M = 2, 64 binaries); the returned grid is checked to be a valid Sudoku that
# respects every given. Port of examples/c/sudoku.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const M = 2
const N = M * M                                     # 4
const NVAR = N * N * N                              # 64
IDX(i, j, k) = (i * N + j) * N + k
# givens (1-based) {row, col, digit}
const HR = [(1, 1, 1), (1, 3, 3), (2, 1, 3), (2, 4, 2), (3, 1, 2), (3, 4, 3), (4, 2, 3), (4, 4, 1)]

function main()
    nrow = 4 * N * N + length(HR)
    m = Model(nrow, NVAR)
    r = try
        for v in 0:NVAR-1
            var_bounds(m, v, BK_RA; lo=0.0, up=1.0)
            var_type(m, v, VT_INT_BIN)
            cj(m, v, 0.0)
        end
        sense(m, SENSE_MIN)
        row = 0
        addrow(sub) = (a_row(m, row, sub, ones(length(sub))); con_bounds(m, row, BK_FX; lo=1.0, up=1.0); row += 1)
        for i in 0:N-1, j in 0:N-1                  # one digit per cell
            addrow([IDX(i, j, k) for k in 0:N-1])
        end
        for i in 0:N-1, k in 0:N-1                  # digit once per row
            addrow([IDX(i, j, k) for j in 0:N-1])
        end
        for j in 0:N-1, k in 0:N-1                  # digit once per column
            addrow([IDX(i, j, k) for i in 0:N-1])
        end
        for bi in 0:M-1, bj in 0:M-1, k in 0:N-1    # digit once per block
            addrow([IDX(bi * M + di, bj * M + dj, k) for di in 0:M-1 for dj in 0:M-1])
        end
        for (i, j, k) in HR                         # givens
            addrow([IDX(i - 1, j - 1, k - 1)])
        end
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    grid = zeros(Int, N, N)
    if ok
        for i in 0:N-1, j in 0:N-1, k in 0:N-1
            r.x[IDX(i, j, k)+1] > 0.5 && (grid[i+1, j+1] = k + 1)
        end
    end
    valid(cells) = sort(cells) == collect(1:N)
    good = ok
    good = good && all(i -> valid(grid[i, :]), 1:N)                   # rows
    good = good && all(j -> valid(grid[:, j]), 1:N)                   # columns
    good = good && all(valid(vec(grid[bi*M+1:bi*M+M, bj*M+1:bj*M+M])) for bi in 0:M-1 for bj in 0:M-1)   # blocks
    good = good && all(grid[i, j] == d for (i, j, d) in HR)          # givens
    @printf("sudoku: rc=%d  grid %s\n", r.rc, good ? "valid and givens respected" : "NOT valid")
    if good
        for i in 1:N
            println("  ", join(grid[i, :], " "))
        end
    end
    return good ? 0 : 1
end

exit(main())
