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

/* Private higher-order PEXP corrector and per-iterate scaling workspace. */
#ifndef EXPCONE_CORRECTOR_H
#define EXPCONE_CORRECTOR_H
#include "expcone.h"

/* One iterate, two Mehrotra right sides; reset valid before advancing x/s. */
typedef struct {
    double theta[9], thin[9], grad[3], scale;
    int valid, branch;
} ExpConeScaleCache;
int expcone_scale_cached(ExpConeScaleCache *cache, int secant, int kind,
    double alpha, const double *z, const double *s, double smu,
    double *theta, double *thin, double *rt, double *scale);

/* PEXP corrector: -0.5 D^3 f(x)[H(x)^-1 ds, dz].
 * Returns -1 with eta=0 if the Hessian solve or contraction is unusable. */
int expcone_pexp_correction(const double *x, const double *ds,
                           const double *dz, double *eta);
void expcone_pexp_corrections(int nep, const int *kind, const double *z,
                             const double *ds, const double *dz, double *eta);

#endif
