# Linear algebra

**Source:** `linalg.c` / `linalg.h`

Everything PrimalSolver factors is in this file. There is no BLAS, no LAPACK, no
external library: these are the kernels the whole solver stands on. They are
small, dense/sparse symmetric, and written for the shapes the solver actually
needs rather than for generality.

## What is available

| Kernel | Type | Used for |
|---|---|---|
| `dmat_lu` | dense, partial pivoting | simplex basis, dense augmented systems |
| `smat_lu` | sparse, partial pivoting | sparse fallback paths |
| `dmat_chol` | dense SPD | normal equations (QP, SOCP) |
| `spchol_factor_ord` | sparse SPD, AMD ordering | normal equations for large sparse LPs and QPs |
| `spldl` | sparse LDLᵀ, 1×1 and 2×2 pivots | indefinite KKT blocks with structurally zero diagonal |
| `dmat_ldl_bk` | dense LDLᵀ, Bunch–Kaufman | indefinite dense systems |
| `dmat_eig_jacobi` | symmetric eigenvalues + eigenvectors | QCQP encoding, smallest-eigenvalue checks |

## The dense LU

Partial pivoting on the largest available column. The allocation is explicit
(`a` is caller-owned, `piv` and the scratch are returned), so no kernel decides
on its own how much memory to take.

## Sparse Cholesky

Left-looking, with a fill-reducing **minimum-degree** ordering
(`sym_amd`) and the permutation applied on solve. This is the path that makes
large sparse LPs and QPs tractable: the interior-point method forms normal
equations `K = A·M⁻¹·Aᵀ + δI` and factors `K` here, never the full KKT.

Measured (`bench/`, sparse route forced on): a 400×200 LP solves in 0.57 s
through this path versus 1.63 s dense — forcing the sparse route closes about
2.9× of the gap to HiGHS on that size, the rest being algorithmic.

## Sparse LDLᵀ with 2×2 pivots

`spldl` exists because a KKT matrix has rows with **structurally zero
diagonal** (an equality row contributes a zero on the diagonal block). Those
rows need a 2×2 pivot pairing them with a neighbour. The 2×2 branch switches on
only when the diagonal is near zero; a quasi-definite matrix stays on the 1×1
path, so the common case is unchanged.

Pivots are matched greedily on off-diagonal nonzeros, and `spldl_solve` applies
the resulting permutation. The permutation is part of the factorization, not
something the caller has to remember: forgetting it is a silent wrong answer,
not a crash, so it lives inside the same file as the factor.

Validated by a standalone oracle — backward error `‖Kx−b‖/(‖K‖·‖x‖+‖b‖) < 1e-9`
on random indefinite matrices with structurally zero diagonals, and on the
KKT family `[[H, Bᵀ],[B, 0]]` — under AddressSanitizer.

## Eigenvalues

`dmat_eig_jacobi` is the cyclic Jacobi method: `evec` is `n × n` row-major and
**column k is eigenvector k**. It is used to encode QCQP constraints exactly
([QCQP encoding](qcqp-encoding.md)) and to read the smallest eigenvalue of a PSD
block. It is slow for large `n` by design — correctness over speed, at a size
where the callers do not care.

## Limits

- No iterative refinement, no supernodal/multifrontal variants, no
  rectangular or rank-deficient paths.
- The sparse Cholesky assumes positive definiteness; the regularizing shift
  `δI` that makes this true is the caller's responsibility.
- `dmat_eig_jacobi` is not competitive with a tridiagonal reduction for
  `n > ~500`; no such caller exists yet.
