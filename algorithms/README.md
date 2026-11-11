# PRIMAL algorithms

Use this guide to learn numerical optimization and trace each method into
the C solver. The [teaching documentation plan](PLAN.md) lists the proposed
chapters, source audit and completion criteria.

Start with linear algebra and multivariable calculus. Read the following
topics in order, or use the links to follow a particular solve path.

| Step | Reading |
|---|---|
| Model and transform | [Standard form](standard-form.md), [presolve](presolve.md), [scaling](scaling.md), [QCQP encoding](qcqp-encoding.md) |
| Solve LP and QP | [Simplex](simplex.md), [interior point](interior-point.md), [linear algebra](linear-algebra.md) |
| Solve conic models | [SOCP](conic-socp.md), [semidefinite programming](sdp.md), [exponential and power cones](exp-power-cones.md), [tangent cuts](tangent-cuts.md) |
| Solve discrete models | [Branch and bound](branch-and-bound.md), [affine and disjunctive constraints](disjunctive-constraints.md) |
| Interpret results | [Certificates](certificates.md), [public API](../docs/api.md), [benchmark methodology](../docs/benchmarks.md) |

Implementation notes cover [SOCP fixed-coordinate reduction](socp-fixed-reduction.md),
[the SOCP second-order corrector](socp-mehrotra-correction.md), and
[higher-order PEXP correction](pexp-higher-correction.md). Experimental
notes cover [homogeneous self-dual embedding](hsd.md) and
[the symmetric SOCP KKT backend](socp-symmetric-kkt.md).
