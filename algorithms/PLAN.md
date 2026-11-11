# PRIMAL teaching documentation plan

## Audience and scope

Write in English for students who have taken linear algebra and multivariable
calculus, and for developers who need to audit PRIMAL's C implementation. A
student should be able to reproduce a worked example and explain its solver
status. A developer should be able to trace a formula from the published model
through the numerical kernel to the reported solution.

Publish the course in `algorithms/`. Use [the index](README.md) as the learning
path. The existing notes provide source material and preserve technical detail
during the rewrite. This plan does not require changes to the solver.

## Chapter inventory and production order

Proposed filenames below describe the finished course. Keep a link to the
existing notes until the replacement covers the same verified material.

| Phase | Chapter | Source review and teaching deliverable | Status |
|---|---|---|---|
| 1 | [`mathematical_foundations.md`](mathematical_foundations.md) | Introduce norms, quadratic forms, PSD matrices, convexity, dual cones, weak and strong duality, and KKT under stated assumptions. Draw a supporting hyperplane and work a primal–dual LP pair. | Completed — verified at `1f2ddee` |
| 1 | [`problem_formulations.md`](problem_formulations.md) | Map LP, QP, QCQP, SOCP, SDP, EXP, POW and MIP to their mathematical forms. Trace bounds, signs and slack variables through `stdform.c` and `primal_quad.c`. Include a modeling example that goes from its original inequalities to canonical form and back. | Completed — verified at `1f2ddee` |
| 1 | [`presolve.md`](presolve.md) | Audit `presolve.c`, `stdform.c` and the dispatch conditions. Derive singleton elimination, show reverse postsolve and homogeneous ray lifting, and draw the row/column changes. | Completed — verified at `1f2ddee` |
| 2 | [`simplex.md`](simplex.md) | Use `simplex.c` and `primal_std.c` to distinguish tableau, revised, and dual paths. Derive reduced costs and the ratio test. Draw the feasible polygon and trace pivots, phase I, degeneracy, and the implemented anti-cycling behavior. | Completed — verified at `1f2ddee` |
| 2 | [`linear_algebra.md`](linear_algebra.md) | Follow `linalg.c` and its callers for dense and sparse LU, Cholesky, ordering, fill and LDLᵀ. Compare block elimination with an augmented solve and draw matrix sparsity before and after ordering. Label experimental backends at the call sites. | Completed — verified at `1f2ddee` |
| 2 | [`interior_point.md`](interior_point.md) | Derive the LP primal–dual Newton equations, affine predictor and corrector. In `ipm.c`, trace the dense augmented and sparse normal-equation paths, including private bound-row elimination and direction checks. Plot measured residuals and complementarity for a small reproducible problem. | Completed — verified at `1f2ddee` |
| 2 | [`quadratic_programming.md`](quadratic_programming.md) | Explain convexity of the Hessian and derive the QP Newton system, regularization and condensed private-slack system from `ipm.c` and `primal_std.c`. Trace small and large eligible QPs, dense fallback, and how `primal_quad.c` encodes convex QCQP. Plot objective contours and active constraints. | Completed — verified at `1f2ddee` |
| 3 | [`socp.md`](conic-socp.md) | Expand [the SOCP chapter](conic-socp.md) with a Lorentz cone section, a worked Newton step, and source-verified pseudocode. Cover Jordan operations, spectral NT scaling, dense LU, sparse NT/Cholesky, sparse augmented LU fallback, fixed-coordinate lifting and predictor–corrector differences. | Completed — verified at `1f2ddee` |
| 3 | [`semidefinite_programming.md`](semidefinite_programming.md) | Derive trace inner products, PSD complementarity, NT scaling and Schur directions. Trace dispatch and recovery in `primal_sdptask.c`, `sdp.c` and `sdp_corrected.c`. Show a 2×2 PSD section and an example with primal and dual matrices. Audit spectral shortcuts and mixed-cone routing. | Completed — verified at `1f2ddee` |
| 3 | [`exponential_cone.md`](exponential_cone.md) | Split the EXP material from [the shared note](exp-power-cones.md). Derive the epigraph perspective, closure, barrier, gradient and Hessian. Match PRIMAL's coordinate order and dual conventions. Explain the native scaling, PEXP corrector, routing and tangent-cut fallback from `expcone.c`, `expcone_corrector.c`, `sdp.c` and `primal_conicopt.c`. | Completed — verified at `1f2ddee` |
| 3 | [`power_cone.md`](power_cone.md) | Derive weighted geometric-mean geometry and the barrier in the code's coordinate order. Distinguish primal and dual parameterizations. Work a fractional-power epigraph and plot slices for several exponents. Audit `expcone.c` and the shared native/fallback routes. | Completed — verified at `1f2ddee` |
| 4 | [`mixed_integer_optimization.md`](mixed_integer_optimization.md) | Trace node relaxations, branching, bounds, pruning and incumbent acceptance in `primal_mip.c` and `primal_mip_opt.c`. Draw a branch-and-bound tree. Explain probing, cuts and heuristics with their implemented scope. Link the [ACC/DJC case study](disjunctive-constraints.md) and distinguish integrality from feasibility tolerances. | Completed — verified at `1f2ddee` |
| 4 | [`certificates.md`](certificates.md) | Derive Farkas alternatives and recession conditions. Verify signs and normalization from `simplex.c`, `primal_std.c` and `primal_verdict.c`. Work infeasible and unbounded examples by substitution, then trace their mapped rays through the public getters. | Completed — verified at `1f2ddee` |
| 4 | [`numerical_stability.md`](numerical_stability.md) | Derive scaling and residual normalization from the relevant routines in `scaling.c`, `ipm.c`, `socp.c` and `sdp.c`. Compare residuals in scaled and original units. Explain pivot tests, refinement, stopping criteria and near-optimal acceptance with route-specific conditions. | Completed — verified at `1f2ddee` |
| 4 | [`architecture.md`](architecture.md) | Follow `PRIMAL_optimize` from model dispatch to postsolve, certificate validation and public getters. Diagram routing and fallbacks. Trace one LP, one QP and one mixed-cone model through actual functions. | Completed — verified at `1f2ddee` |

Technical notes on [tangent cuts](tangent-cuts.md), [QCQP encoding](qcqp-encoding.md)
and [disjunctive constraints](disjunctive-constraints.md) can remain companion
readings. Keep [HSD](hsd.md) and [symmetric SOCP KKT](socp-symmetric-kkt.md)
explicitly marked as experimental.

## Solver changes the chapters must cover

These commits changed the solver after the drafts' source material was
written. Each chapter listed must describe the new route, its eligibility
condition and its fallback, and cite the test that checks it.

| Commit | Change | Code | Test | Chapter |
|---|---|---|---|---|
| `88f3597` | PEXP higher-order corrector; cached exp/power scaling; corrected conic entry point | `expcone_corrector.c` (`expcone_pexp_correction`, `expcone_pexp_corrections`), `expcone_scale_cached`, `sdp_corrected.c` (`sdp_ipm_csr_corrected`) | `T299` | `exponential_cone.md`, `semidefinite_programming.md` |
| `adb7f74` | Recovery of accurate steps near singular PSD faces; progress-stall test | `sdp.c` (`sdp_primal_step_residual`), `sdp_progress.h` (`sdp_progress_stalled`) | `T301`, `T302` | `semidefinite_programming.md`, `numerical_stability.md` |
| `314d4f0` | After a repaired PSD step, a pure-SDP solve stops once the original equations, both cones and half the gap tolerance pass | `sdp.c` | — | `semidefinite_programming.md` |
| `1bf265c` | Faster PSD step-length checks; augmented solve with singleton elimination in recovery | `sdp.c` (`psd_step_limit`, `sdp_aug_solve`, `sdp_aug_singletons`), `linalg.c` | — | `semidefinite_programming.md`, `linear_algebra.md` |
| `bd53531` | Fixed-diagonal SDPs certified directly | `sdp.c` (`sdp_fixed_diagonal`) | — | `semidefinite_programming.md`, `certificates.md` |
| `56ea21f` | Sparse KKT selected for mixed exponential models within a Schur budget | `sdp.c` (`sdp_schur_within_budget`) | — | `exponential_cone.md`, `architecture.md` |
| `b452e46` | Eligible MIP LP relaxations start from the slack basis | `primal_std.c` (`mip_slack_solution`), `simplex.c` (`simplex_solve_std_slack`, `simplex_solve_std_tab`) | — | `mixed_integer_optimization.md`, `simplex.md` |
| `a09509a` | Sparse interior point for LPs with `m*n > 2000` and `m <= 2000` (was `m*n > 1e4`) | `primal_std.c` (`use_sparse`) | — | `interior_point.md`, `architecture.md` |
| `9906d96` | Variable bound rows eliminated from the LP normal equations | `ipm.c` (`LpNormal`, `lp_normal_solve`, `lp_normal_full_solve`) | `T303` | `interior_point.md` |
| `31f8431` | QP slack systems reduced; dense Hessian blocks factored | `ipm.c` (`QpKkt`, `qp_kkt_reduce`, `qp_kkt_solve`, `qp_kkt_residual`) | `T304` | `quadratic_programming.md` |
| `da2e477`, `6aecdf2` | QP Newton systems condensed with private slacks, small systems included | `ipm.c` (`qp_solve_condensed`), `primal_std.c` (`small_qp_condensable`) | `T304` | `quadratic_programming.md` |
| `01b4673` | Models with at least 16 three-dimensional SOC blocks, at most 4 equalities and coordinate-slack rows go to the sparse SOCP solver first, without the dense retry | `primal_conicopt.c` (`prefer_sparse`) | — | `socp.md`, `architecture.md` |

A dash in the test column means no dedicated `T<n>` case was found for the
change; the chapter must then give a reproducible example instead.

## Audit of the drafts

**Status (2026-11-11, later the same day): closed.** Every gap below was
addressed in the rewrite of the fifteen chapters, each of which now carries
learning objectives, a derivation, pseudocode, source links, a verified worked
example, a figure or measured trace, costs and failure modes, an exercise with
its answer and checked references. The text of the original audit follows as a
record of what was found.

Audited on 2026-11-11 against `1f2ddee` (structure, links, coverage of the
changes above, and spot checks of implementation claims). Every chapter has a
draft and a reference list, but none yet meets the chapter structure and the
completion criteria below, so all are marked **Draft**.

Gaps common to every chapter:

- No learning objectives or prerequisites (structure item 1).
- No exercise (item 9).
- No verified revision recorded for the implementation claims, and few `T<n>`
  cases cited (at most one per chapter).
- Pseudocode for a complete iteration is missing everywhere except
  `simplex.md` and `exponential_cone.md` (item 4).

Gaps per chapter:

| Chapter | Missing or wrong |
|---|---|
| `mathematical_foundations.md` | No figure (the supporting hyperplane in the deliverable). |
| `problem_formulations.md` | No round trip of the modeling example back from canonical form. |
| `presolve.md` | No reverse postsolve or ray-lifting example. |
| `simplex.md` | Figure `figures/simplex-pivots.svg` does not exist. Slack-basis start for MIP relaxations (`b452e46`) not covered. |
| `linear_algebra.md` | No worked example; no sparsity figure before and after ordering. `sdp_aug_solve` / `sdp_aug_singletons` (`1bf265c`) not covered. |
| `interior_point.md` | Figure `figures/lp-central-path.svg` does not exist and the plan asks for a measured trace, not an illustration. Sparse-route threshold (`a09509a`) not covered. |
| `quadratic_programming.md` | Figure `figures/qp-contours.svg` does not exist. `small_qp_condensable` and `qp_kkt_reduce` (`6aecdf2`, `31f8431`) not covered. |
| `socp.md` | No pseudocode; no `T<n>` cases. Small sparse routing (`01b4673`) not covered. |
| `semidefinite_programming.md` | No source links. None of `adb7f74`, `314d4f0`, `1bf265c`, `bd53531` covered; `sdp_corrected.c` (named in the deliverable) not mentioned. |
| `exponential_cone.md` | No source links. `expcone_pexp_correction` and the sparse-KKT budget (`56ea21f`) not covered. |
| `power_cone.md` | No source links; `ppow_core` and the power-cone scaling point (`expcone_dual_point`) not cited. No slice plot for several exponents. |
| `mixed_integer_optimization.md` | No source links. **Wrong claim:** the flowchart and the cut list present conflict cuts as part of the default search, but they are off unless `GMB_MIP_CONFLICT` is set (`primal_mip.c`, `mip_conflict_cuts`). Slack-basis start (`b452e46`) not covered. |
| `certificates.md` | No source links; the direct certificate of fixed-diagonal SDPs (`bd53531`) not covered. |
| `numerical_stability.md` | No source links; no worked example comparing scaled and original residuals. |
| `architecture.md` | No source links; no worked trace of one LP, one QP and one mixed-cone model; the new routing decisions (`a09509a`, `01b4673`, `56ea21f`) not shown. |

## Order of work

1. Fix what is wrong before adding material: the MIP conflict-cut claim and
   the three missing figures (draw them, or remove the references).
2. Cover the solver changes above, chapter by chapter, with the test or a
   reproducible example for each.
3. Bring each chapter to the chapter structure: objectives, pseudocode,
   source links with the verified revision, exercise.
4. Run the completion criteria, then mark the chapter **Completed** in the
   inventory with the revision it was verified against.

## Source audit before writing

The initial SOCP and DJC revisions were checked against revision `6aecdf2`.

For each chapter, record the verified revision, file and function for every
implementation claim. Follow the normal public entry point, then check
alternative routes, eligibility conditions, fallback order, tolerance units
and status publication. Record the relevant `T<n>` case or a reproducible
example and the quantities it asserts. A historical benchmark is evidence
for its measured configuration. Link current performance claims to dated
measurements.

Start the audit with these concrete checks:

- **SOCP.** Read `socp_solve`, `nt_setup` and `socp_solve_sparse_impl`
  separately. Compare the dense centered RHS with the production sparse
  second-order corrector. Trace the dispatcher separately from per-iteration
  NT rejection and sparse LU.
- **DJC.** Derive the finite box maximum for each oriented inequality.
  Check `djc_encode` and `djc_rewrite_row`, including their `1e6` fallback.
  Compare the ACC registry with the four accepted DJC domains in
  `djc_validate`. Verify an AFE refresh and a rejected nonlinear domain.
- **QP.** Follow `small_qp_condensable`, `qp_factor_condensed` and
  `qp_solve_condensed`. Derive the elimination from the original regularized
  equations and explain how the reconstructed direction is checked. Verify
  small, large and ineligible models.
- **Presolve, SDP and MIP.** Audit claims about when a rule is applied, which
  residuals are measured, and what is published. Old experiments and labels
  in comments are leads to verify against executable branches.

## Chapter structure

1. State the learning objectives and prerequisites.
2. Motivate the problem with an example and introduce notation and assumptions.
3. Derive the algorithm from the problem equations. Keep signs, dimensions,
   coordinate order and regularization explicit.
4. Give pseudocode for a complete iteration, including stopping and recovery.
5. Explain the PRIMAL implementation with repository-relative source links,
   named functions and tests. Mark general theory, production routes,
   optional features and experimental routes with clear headings.
6. Work a numerical example with intermediate values and final residuals,
   objective values, or a checked certificate.
7. Include a figure and a caption that explains what to look for.
8. Explain costs, fill, conditioning, degeneracy and failure modes.
9. End with one short exercise and a verified bibliography.

## Figures and worked examples

Place figure outputs and their generators under `algorithms/figures/` as
chapters are completed. Use Mermaid for flowcharts and standard plotting
tools for mathematical curves and measured traces. Keep a fixed seed where
applicable. Document the generation command, inputs, axes, units, and whether
a plot is analytic or measured. Save trace data when a convergence figure
depends on a particular build. Use captions and alt text that are readable
without color.

Check analytic examples by substitution. For Newton derivations, compare a
small step with a direct solve of the original system and report its residual.
For cone figures, check boundary points against the stated defining inequality.
For measured solver examples, record the status, public getter values,
tolerances and code revision. Review figures at the size they render in GitHub.

## Completion criteria

- All chapters link from the index in prerequisite order.
- Every formula used to describe PRIMAL agrees with the current code or names
  the precise route and conditions under which it applies.
- Worked examples and figures are reproducible and their claims are checked.
- Source links, Markdown links, anchors and GitHub math rendering are reviewed.
- A student can follow the derivation with the stated prerequisites. A
  developer can locate the implementation, its tests and its failure paths.
- Primary references are verified when each chapter is written. Start with
  [Boyd and Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/) for convexity
  and duality, [Mehrotra](https://doi.org/10.1137/0802028) for the
  predictor–corrector, [Nesterov and Todd](https://doi.org/10.1287/moor.22.1.1)
  for symmetric cones, and [Vielma](https://doi.org/10.1137/130915303) for
  disjunctive formulations.
