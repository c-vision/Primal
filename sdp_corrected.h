/*
 * PrimalSolver - a convex optimization solver in C99 (LP/QP/SOCP/SDP/exp-power/MIP).
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License.  A copy of the License
 * is in the repository root (LICENSE) and at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

/* Private CSR entry point for the higher-order PEXP engine. */
#ifndef SDP_CORRECTED_H
#define SDP_CORRECTED_H
#include "sdp.h"

/* Internal separately compiled higher-order PEXP engine. */
int sdp_ipm_csr_corrected(int m, int n, const SdpCsr *A, const double *b, const double *c,
            int nb, const int *dims,
            const double *const *Cbar, const SdpBlk *Abar,
            int nsoc, const int *socdims,
            const double *const *Csoc, const SdpBlk *Asoc,
            int nep, const int *ekind, const double *ealpha,
            const double *const *Cexp, const SdpBlk *Aexp,
            int max_iter, double tol,
            double rtol_pri, double rtol_dual, double rtol_gap,
            double near_rel,
            double *x, double *const *Xbar, double *y, double *const *Sbar,
            double *const *Zsoc, double *const *Ssoc,
            double *const *Zexp, double *const *Sexp, const double *xwarm,
            const double *ywarm,
            int *fb_ok);

#endif
