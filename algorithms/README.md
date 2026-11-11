# PRIMAL algorithms

Use this guide to learn numerical optimization and trace each method into
the C solver. The [teaching documentation plan](PLAN.md) outlines the
curriculum structure, verified source audits, and completion criteria across all
four phases.

Start with linear algebra and multivariable calculus. Follow the topics in order:

## Curriculum overview

| Phase | Topics | Core Chapters |
|---|---|---|
| **Phase 1: Foundations & Modeling** | Convexity, Duality, Standard Forms, Presolve | [Mathematical Foundations](mathematical_foundations.md) · [Problem Formulations](problem_formulations.md) · [Presolve](presolve.md) · [Standard Form](standard-form.md) · [QCQP Encoding](qcqp-encoding.md) |
| **Phase 2: Linear & Quadratic** | Simplex, Factorizations, Predictor-Corrector, QP | [Simplex Method](simplex.md) · [Linear Algebra](linear_algebra.md) · [Interior Point](interior_point.md) · [Quadratic Programming](quadratic_programming.md) |
| **Phase 3: Conic Optimization** | Lorentz, PSD, Exponential, and Power Cones | [Second-Order Cones (SOCP)](conic-socp.md) · [Semidefinite Programming (SDP)](semidefinite_programming.md) · [Exponential Cone](exponential_cone.md) · [Power Cones](power_cone.md) · [Tangent Cuts](tangent-cuts.md) |
| **Phase 4: Discrete & Systems** | Branch & Bound, Infeasibility Rays, Stability, Pipeline | [Mixed-Integer Optimization](mixed_integer_optimization.md) · [Disjunctive Constraints](disjunctive-constraints.md) · [Certificates & Rays](certificates.md) · [Numerical Stability](numerical_stability.md) · [Solver Architecture](architecture.md) |

## Implementation and experimental notes

Detailed engineering and specialized backend notes:
- **SOCP:** [SOCP fixed-coordinate reduction](socp-fixed-reduction.md), [SOCP second-order corrector](socp-mehrotra-correction.md), and [symmetric SOCP KKT backend](socp-symmetric-kkt.md) (experimental).
- **Exponential & Power:** [Higher-order PEXP correction](pexp-higher-correction.md) and [legacy exp-power notes](exp-power-cones.md).
- **Semidefinite & Conic:** [Legacy unified SDP notes](sdp.md).
- **Embedding:** [Homogeneous self-dual embedding (HSD)](hsd.md) (experimental).
- **API & Benchmarking:** [Public C API](../docs/api.md) and [benchmark methodology](../docs/benchmarks.md).
