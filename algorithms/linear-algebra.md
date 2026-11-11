# Linear algebra: kernel inventory

**Source:** `linalg.c` / `linalg.h`. **Verified against** revision `1f2ddee`.

Everything PrimalSolver factors is in this file. There is no BLAS, no LAPACK and no
external library. The teaching chapter is [linear algebra for Newton and basis
systems](linear_algebra.md); this page is the reference list of the kernels, what they
assume, and who calls them.

## Kernels

| Kernel | Algorithm | Fails when | Production callers |
|---|---|---|---|
| `dmat_lu_factor` / `dmat_lu_solve` | dense LU, partial pivoting, reusable factor | pivot below $10^{-300}$ (exactly singular): returns `NULL` | `ipm.c` (dense augmented system, QP reduction), `simplex.c` (duals of the dual simplex), `socp.c`, `sdp.c`, `primal_verdict.c`, `primal_misc.c`, `primal_quad.c` |
| `dmat_lu_solve_comp` | same factor, Neumaier-compensated substitutions | as above | the degenerate PSD/SOC path of `sdp.c` |
| `dmat_solve_lu` | one-shot LU, destroys its input | pivot below $10^{-14}$ (absolute) | legacy API |
| `dmat_cholesky` / `dmat_chol_solve` | dense Cholesky | non-positive pivot | `expcone_corrector.c` |
| `spchol_factor` | sparse left-looking Cholesky, natural order | non-positive pivot: returns `NULL` | `ipm.c`, `socp.c`, `primal_misc.c` |
| `spchol_factor_ord` | the same with an AMD ordering applied inside | as above | `ipm.c`, `socp.c`, `primal_misc.c` |
| `spchol_order` + `spchol_factor_perm` | ordering computed once, reused for a fixed pattern | as above | `ipm.c`, `socp.c` |
| `spchol_factor_fail` | like `spchol_factor`; on a bad pivot also returns the shift that would repair it | as above | `socp.c` |
| `spchol_factor_dense` | Cholesky of a matrix known to have a full lower triangle | as above | `ipm.c` |
| `spchol_solve`, `spchol_solve_all` | natural-order solve, one or many right-hand sides | **refuse** an ordered factor | `ipm.c` |
| `spchol_solve_ord` | solve through any factor, undoing its ordering | NULL input | `ipm.c`, `socp.c` |
| `splu_factor` / `splu_factor_ord` | sparse LU, row pivoting, column ordering | pivot below $10^{-300}$ or a row that cannot grow; needs distinct row indices in a column | `socp.c`, `sdp.c` |
| `sym_amd_order` | approximate minimum degree ordering | allocation failure: `NULL` | `sdp.c` |
| `spqd_analyze` / `spqd_factor` / `spqd_solve` | LDL of a quasi-definite matrix with regularization | factor refuses | **experimental**, `socp.c` with `GMB_SOCP_LDL` |
| `spldl_factor` / `spldl_solve` | sparse LDL with static 1×1 and 2×2 pivots | a block does not factor | no production caller; tests only |
| `dmat_ldl_bk` | dense Bunch–Kaufman LDL | numerically singular | no production caller; tests only |
| `sym_rcm_order` | reverse Cuthill–McKee ordering | | no production caller |
| `dmat_eig_jacobi` | cyclic Jacobi, one-sided update; `evec` column $k$ is eigenvector $k$ | | QCQP encoding, PSD checks, the 3×3 metrics of the exp/power cones |
| `dmat_eig_jacobi_twosided` | the same rotations with the two-sided update | | the experimental HSD path in `sdp.c` only |

An ordered `SpChol` carries its permutation. `spchol_solve` and `spchol_solve_all` return
an error for it; `spchol_solve_ord` is the safe call.

## Eigenvalues

`dmat_eig_jacobi` stops when the off-diagonal sum of squares falls below
$10^{-28}\|A\|_F^2$ (about $10^{-14}$ of the matrix), so the tolerance is relative. It is
slow for large $n$ by design.

## Limits

- No iterative refinement; no supernodal or multifrontal variants.
- The sparse Cholesky assumes positive definiteness; the regularizing shift is the
  caller's responsibility.
- `dmat_eig_jacobi` is not competitive with a tridiagonal reduction for large $n$.
