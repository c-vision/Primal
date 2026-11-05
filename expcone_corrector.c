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

/* Higher-order PEXP corrector and scaling cache, see expcone_corrector.h. */
#include <math.h>
#include <string.h>
#include "linalg.h"
#include "expcone_corrector.h"

/* PEXP third-order correction; isolated to preserve barrier compilation. */
int expcone_scale_cached(ExpConeScaleCache *cache, int secant, int kind,
    double alpha, const double *z, const double *s, double smu,
    double *theta, double *thin, double *rt, double *scale) {
    if (cache->valid) {
        memcpy(theta, cache->theta, sizeof cache->theta);
        memcpy(thin, cache->thin, sizeof cache->thin);
        *scale = cache->scale;
        for (int a = 0; a < 3; a++) rt[a] = (-s[a] - smu*cache->grad[a]) / *scale;
        return cache->branch;
    }
    int branch = secant ? expcone_scaling_da(kind, alpha, z, s, smu, theta, thin, rt, scale)
                       : expcone_scaling(kind, alpha, z, s, smu, theta, thin, rt, scale);
    if (branch >= 0 && !expcone_grad(kind, alpha, z, cache->grad)) {
        memcpy(cache->theta, theta, sizeof cache->theta);
        memcpy(cache->thin, thin, sizeof cache->thin);
        cache->scale = *scale; cache->branch = branch; cache->valid = 1;
    }
    return branch;
}

void expcone_pexp_corrections(int nep, const int *kind, const double *z,
                             const double *ds, const double *dz, double *eta) {
    for (int i = 0; i < nep; i++) {
        for (int a = 0; a < 3; a++) eta[3*i+a] = 0;
        if (kind[i] == EXPCONE_PEXP)
            expcone_pexp_correction(z + 3*i, ds + 3*i, dz + 3*i, eta + 3*i);
    }
}

int expcone_pexp_correction(const double *x, const double *ds,
                           const double *dz, double *eta) {
    double h[9], u[3] = {ds[0], ds[1], ds[2]};
    double *rows[3] = {h, h + 3, h + 6};
    DMat H = {3, 3, rows};
    for (int i = 0; i < 3; i++) eta[i] = 0;
    if (expcone_hess(EXPCONE_PEXP, 0, x, h) ||
        dmat_cholesky(&H, 3) || dmat_chol_solve(&H, u, 3)) return -1;
    /* For f=-log(psi)-log(t)-log(v), psi=v*log(t/v)-x2,
     * contract D^3 f analytically after solving H*u=ds.  In the orthant
     * analogue eta=ds*dz/x, giving the usual Mehrotra cross product.
     * This uses the full affine directions, before fraction-to-boundary. */
    double t = x[0], v = x[1], r = log(t / v), psi = v * r - x[2];
    double p[3] = {v / t, r - 1, -1};
    double B[9] = {-v/(t*t), 1/t, 0, 1/t, -1/v, 0, 0, 0, 0};
    double bu[3] = {0}, bv[3] = {0}, pu = 0, pv = 0, uv = 0;
    for (int i = 0; i < 3; i++) {
        pu += p[i] * u[i]; pv += p[i] * dz[i];
        for (int j = 0; j < 3; j++) { bu[i] += B[3*i+j]*u[j]; bv[i] += B[3*i+j]*dz[j]; }
        uv += u[i] * bv[i];
    }
    double third[3] = {
        2*v/(t*t*t)*u[0]*dz[0] - (u[0]*dz[1]+u[1]*dz[0])/(t*t),
        -u[0]*dz[0]/(t*t) + u[1]*dz[1]/(v*v), 0};
    double e[3];
    for (int i = 0; i < 3; i++) {
        double z = -2*p[i]*pu*pv/(psi*psi*psi)
            + (bu[i]*pv+bv[i]*pu+p[i]*uv)/(psi*psi) - third[i]/psi;
        if (i < 2) z -= 2*u[i]*dz[i]/(x[i]*x[i]*x[i]);
        e[i] = -0.5*z;
        if (!isfinite(e[i])) return -1;
    }
    for (int i = 0; i < 3; i++) eta[i] = e[i];
    return 0;
}
