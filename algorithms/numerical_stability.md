# Numerical stability, scaling and stopping criteria

Numerical optimization solvers operate in finite-precision IEEE 754 floating-point
arithmetic. Without careful preconditioning, pivot safeguards, and normalized
stopping tests, ill-conditioned models produce divergence, false infeasibility
claims, or stalled progress. This chapter explains PRIMAL's numerical safeguards:
exact power-of-2 Ruiz scaling, Jacobi equilibration, iterative refinement with
Wilkinson's stopping test, dynamic regularization, and normalized convergence criteria.

## Equilibration with powers of two

Raw models often feature constraint coefficients spanning twelve orders of magnitude
(e.g. electrical networks or penalty reformulations).
To balance the linear system without altering the significand (mantissa),
`scaling.c` implements geometric equilibration where every scaling factor is rounded
to the nearest power of two.

### The Ruiz power-of-two equilibration

Let $A \in \mathbb{R}^{m\times n}$ be the constraint matrix.
Define diagonal matrices $R = \operatorname{diag}(r_1,\dots,r_m)$ and
$D = \operatorname{diag}(d_1,\dots,d_n)$.
The code performs 4 alternating passes:
1. For each row $i$, compute the root-mean-square norm:
   $$
   \rho_i = \sqrt{\frac{1}{n_i} \sum_{j \in \operatorname{supp}(A_{i,:})} A_{ij}^2}.
   $$
2. Compute $r_i = \operatorname{pow2\_round}(1 / \rho_i)$, where
   $\operatorname{pow2\_round}(x) = 2^{\lfloor \log_2(x) + 0.5 \rfloor}$, clamped to $[2^{-24}, 2^{24}]$.
3. Scale row $i$ of $A$ and its bounds $l_c, u_c$ by $r_i$.
4. Repeat the procedure along the columns, computing column scaling factors $d_j$
   and scaling column $j$ of $A$, variable bounds $l_x, u_x$, and objective vector $c$.
5. If a quadratic matrix $Q$ is present, it is scaled symmetrically as $Q \leftarrow D Q D$.

```mermaid
flowchart LR
    A[Unscaled Model] --> B[Pass 1: Row RMS scaling to 2^k]
    B --> C[Pass 2: Column RMS scaling to 2^k]
    C --> D[Repeat 4 passes]
    D --> E[Zero-mantissa roundoff scaled model]
    E --> Sol[IPM / Simplex Solve]
    Sol --> F[Unscale x = D * x_scaled, y = R * y_scaled]
```

Because scaling factors are powers of two, the operation $x \cdot 2^k$ simply increments
the IEEE 754 exponent field by $k$. It produces **zero roundoff error** in the mantissa.

## Residual normalization and stopping criteria

Primal and dual residuals depend on problem scale. PRIMAL measures dimensionless
relative residuals:

$$
\text{rel\_pri} = \frac{\|r_p\|_\infty}{\max(1, \|b\|_\infty)}, \qquad
\text{rel\_dual} = \frac{\|r_d\|_\infty}{\max(1, \|c\|_\infty)}, \qquad
\text{rel\_gap} = \frac{|c^T x - b^T y|}{1 + \frac{1}{2}(|c^T x| + |b^T y|)}.
$$

An iterate is declared `OPTIMAL` when:

$$
\text{rel\_pri} \le \epsilon_{\mathrm{pfeas}}, \qquad
\text{rel\_dual} \le \epsilon_{\mathrm{dfeas}}, \qquad
\text{rel\_gap} \le \epsilon_{\mathrm{relgap}}.
$$

Defaults for continuous problems are typically $10^{-8}$.

### Near-optimal acceptance

When an interior-point iterate stalls near the boundary of the cone due to
extreme conditioning ($\kappa \approx 10^{15}$), further Newton steps can introduce
cancellation without reducing residuals. If no fallback route exists (such as
in pure SDP), PRIMAL allows a relaxed acceptance factor: if residuals satisfy
$1000 \times \epsilon_{\mathrm{tol}}$, the solver publishes `OPTIMAL` with near-optimal
tolerances rather than failing.

## Augmented system equilibration and iterative refinement

In the unified conic solver (`sdp.c`), the augmented KKT matrix:

$$
K_{\mathrm{aug}} = \begin{bmatrix}
K_{pp} & E^T \\
E & -\Theta
\end{bmatrix}
$$

mixes second-order cone arrow blocks, PSD Schur blocks, exponential Hessian blocks,
and scalar slacks.

### Jacobi diagonal equilibration

Before factorization, `sdp.c` runs a two-sided Jacobi diagonal scaling:

$$
S = \operatorname{diag}(s_1,\dots,s_N), \quad s_i = \frac{1}{\sqrt{\max_j |(K_{\mathrm{aug}})_{ij}|}}, \qquad
\widetilde{K}_{\mathrm{aug}} = S K_{\mathrm{aug}} S.
$$

This balances entries across distinct cone orders, preventing the dense LU factorization
from prematurely declaring rank deficiency.

### Iterative refinement with Wilkinson stopping criterion

During the solution polish phase, `sdp.c` refines the solution $\Delta v$ to $K_{\mathrm{aug}} \Delta v = r$:
1. Solve $K_{\mathrm{aug}} \Delta v^{(0)} = r$ using the computed LU factors.
2. At step $k$, compute the high-precision residual $d^{(k)} = r - K_{\mathrm{aug}} \Delta v^{(k)}$.
3. Solve $K_{\mathrm{aug}} \delta^{(k)} = d^{(k)}$ and update $\Delta v^{(k+1)} = \Delta v^{(k)} + \delta^{(k)}$.
4. **Wilkinson's stopping test:** Terminate when the norm $\|d^{(k)}\|_\infty$ fails
   to decrease by a factor of at least 2, or when machine precision is reached.
   This avoids iterating when numerical noise dominates.

## Regularization and pivot floors

1. **Normal equations shift:** In `ipm.c`, the Cholesky factorization of $A \Theta A^T$
   adds a dynamic diagonal shift $\delta I$. If Cholesky fails, $\delta$ escalates
   geometrically:
   $$
   \delta \leftarrow 10 \cdot \delta, \quad \delta \in [10^{-6}, 10^{-1}].
   $$
2. **Dense LU pivoting:** In `linalg.c`, `dmat_solve_lu` employs partial pivoting.
   A pivot element is accepted only if $|P_{ii}| > 10^{-14} \max_k |P_{ki}|$;
   otherwise, a diagonal perturbation is applied to maintain solvability.

## References

- D. Ruiz, [A scaling algorithm to equilibrate both rows and columns norms in matrices](https://www.numerical.rl.ac.uk/reports/drRAL2001034.pdf), Rutherford Appleton Laboratory, 2001.
- J. H. Wilkinson, [Rounding Errors in Algebraic Processes](https://archive.org/details/roundingerrorsin0000wilk), Prentice-Hall, 1963.
- N. J. Higham, [Accuracy and Stability of Numerical Algorithms](https://doi.org/10.1137/1.9780898718027), SIAM, 2nd ed., 2002.
