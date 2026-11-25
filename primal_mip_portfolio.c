/*
 * PrimalSolver - a convex optimization solver in C99.
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 */

/* Cardinality-constrained allocations with disjoint SOC, paired EXP and
 * arrow PSD risk blocks. The mass row couples the blocks. Dropping it gives
 * a bound, which is used only when an original feasible point closes the gap. */
#include "primal_priv.h"
#include <stdint.h>

#define PORT_BLOCK 16
#define PORT_ARROW 8
#define PORT_WEIGHTS (2*PORT_BLOCK + PORT_ARROW/2)
#define PORT_VARS (3*PORT_WEIGHTS + 2 + 9*PORT_BLOCK)
#define PORT_ROWS (3*PORT_WEIGHTS + 7*PORT_BLOCK + 4 + (PORT_ARROW+1)*(PORT_ARROW+2)/2)

typedef struct {
    int size[3], start[3], weights, cap, scale, qrow, base, dim;
    int gate[PORT_WEIGHTS], shared;
    double mass, soc_radius, exp_budget, psd_head, scale_max;
    double map[3][PORT_BLOCK*PORT_BLOCK];
    double diag[PORT_ARROW*PORT_ARROW], slope[PORT_ARROW*PORT_ARROW];
} Portfolio;

typedef struct {
    double lower, upper, scale, x[PORT_BLOCK];
} PortfolioValue;

/* Read the scalar and matrix links in the form emitted by the CBF reader.
 * Consume every row and check all coefficients outside each expected block. */
static int portfolio_row(const double *row, int cols, int first, int count) {
    for (int j = 0; j < cols; j++)
        if ((j < first || j >= first + count) && row[j] != 0.0) return 0;
    return 1;
}

static int portfolio_read(PRIMALtask_t t, Portfolio *p, double *a) {
    int n = t->numvar, m = t->numcon, d = t->barDim[0], k = d-1;
    int q = t->cone_nmem[0]-1, e = (t->numcones-1)/2;
    int b = d/2, w = q+e+b;
    if (q < 1 || q > PORT_BLOCK || e < 1 || e > PORT_BLOCK ||
        t->numcones != 2*e+1 || k != 2*b || b < 1 || k > PORT_ARROW ||
        t->cone_type[0] != PRIMAL_CT_QUAD) return 0;
    int base = n - (q+1+6*e), scale = base-w-2*e-1, g = scale-w;
    int qrow = 3+g+2*w, erow = qrow+q+1, barrow = erow+6*e;
    int triangle = d*(d+1)/2;
    if (g < 1 || g > w || m != barrow+triangle || t->nbarA != triangle) return 0;

    for (int j = 0; j < n; j++) {
        int gated = j >= w && j < scale;
        int type = t->vartype[j];
        if (gated ? type != PRIMAL_VAR_TYPE_INT : type != PRIMAL_VAR_TYPE_CONT) return 0;
        if (j >= w && t->c[j] != 0.0) return 0;
        if (!isfinite(t->c[j])) return 0;
        if (j <= scale) {
            if (t->bkx[j] != PRIMAL_BK_LO || t->blx[j] != 0.0) return 0;
        } else if (t->bkx[j] != PRIMAL_BK_FR) return 0;
        for (int z = 0; z < t->cols[j].nz; z++) {
            double v = t->cols[j].val[z];
            if (!isfinite(v)) return 0;
            a[(size_t)t->cols[j].sub[z]*n+j] += v;
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            if (!isfinite(a[(size_t)i*n+j])) return 0;
        PRIMALboundkeye kind = i < 3+w+g ? PRIMAL_BK_LO : PRIMAL_BK_FX;
        if (t->bkc[i] != kind || !isfinite(t->blc[i])) return 0;
    }
    double mass = -t->blc[0], capacity = -t->blc[1], risk = -t->blc[2+w];
    if (!(mass > 0.0 && capacity >= 0.0 && capacity <= g && capacity == floor(capacity)
            && risk > 0.0 && risk < 0.25)) return 0;
    if (!portfolio_row(a,n,0,w) || !portfolio_row(a+n,n,w,g) ||
        !portfolio_row(a+(size_t)(2+w)*n,n,base-2*e,2*e)) return 0;
    for (int j = 0; j < w; j++) if (a[j] != -1.0) return 0;
    for (int j = w; j < scale; j++) if (a[n+j] != -1.0) return 0;
    for (int j = base-2*e; j < base; j++) if (a[(size_t)(2+w)*n+j] != -1.0) return 0;

    int uses[PORT_WEIGHTS] = {0};
    p->shared = -1;
    for (int j = 0; j < w; j++) {
        int r = 2+j, gate = -1;
        const double *row = a+(size_t)r*n;
        if (t->blc[r] != 0.0 || row[j] != -1.0) return 0;
        for (int col = 0; col < n; col++) {
            if (col == j || row[col] == 0.0) continue;
            if (gate != -1 || col < w || col >= scale || row[col] != 1.0) return 0;
            gate = col-w;
        }
        if (gate < 0) return 0;
        if (++uses[gate] > 1) {
            if (p->shared >= 0 && p->shared != gate) return 0;
            p->shared = gate;
        }
        p->gate[j] = gate;
    }
    for (int j = 0; j < g; j++) {
        int r = 3+w+j;
        const double *row = a+(size_t)r*n;
        if (!uses[j] || t->blc[r] != -1.0 ||
            row[w+j] != -1.0 || !portfolio_row(row,n,w+j,1)) return 0;
    }
    int at = 3+w+g;
    for (int j = 0; j < w; j++) {
        int r = at+j, col = scale+1+j;
        double *row = a+(size_t)r*n;
        if (t->blc[r] != 0.0 || row[col] != -1.0) return 0;
        row[col] = 0.0;
        int block = j < q ? 0 : j < q+e ? 1 : 2;
        int start = block == 0 ? 0 : block == 1 ? q : q+e;
        int count = block == 0 ? q : block == 1 ? e : b;
        if (!portfolio_row(row,n,start,count)) return 0;
        if (block < 2) memcpy(p->map[block]+(j-start)*count,row+start,(size_t)count*sizeof(double));
        row[col] = -1.0;
    }
    for (int j = 0; j <= q; j++) {
        int col = base+j, r = qrow+j;
        double *row = a+(size_t)r*n;
        if (t->cone_mem[0][j] != col || row[col] != -1.0) return 0;
        row[col] = 0.0;
        if (j == 0) {
            p->soc_radius = -t->blc[r];
            if (!(p->soc_radius > 0.0) || !portfolio_row(row,n,0,0)) return 0;
        } else if (t->blc[r] != 0.0 || row[scale+j] != 1.0 ||
                   !portfolio_row(row,n,scale+j,1)) return 0;
        row[col] = -1.0;
    }
    /* Match the two signs of each affine exponential argument. The budget
     * sums their epigraphs u log(u) with unit coefficients. */
    uint64_t used = 0;
    for (int j = 0; j < 2*e; j++) {
        int cone = 1+j, col = base+q+1+3*j, r = erow+3*j;
        if (t->cone_type[cone] != PRIMAL_CT_PEXP || t->cone_nmem[cone] != 3) return 0;
        for (int h = 0; h < 3; h++)
            if (t->cone_mem[cone][h] != col+h || a[(size_t)(r+h)*n+col+h] != -1.0) return 0;
        if (t->blc[r] != -1.0 || !portfolio_row(a+(size_t)r*n,n,col,1)) return 0;
        int zcol = scale+1+q+j/2;
        double *arg = a+(size_t)(r+1)*n, *epi = a+(size_t)(r+2)*n;
        arg[col+1] = epi[col+2] = 0.0;
        if (t->blc[r+1] != -1.0 || arg[zcol] != (j%2 == 0 ? 1.0 : -1.0) ||
            !portfolio_row(arg,n,zcol,1) || t->blc[r+2] != 0.0 ||
            !portfolio_row(epi,n,base-2*e,2*e)) return 0;
        int found = -1;
        for (int h = 0; h < 2*e; h++) {
            double val = epi[base-2*e+h];
            if (val == 0.0) continue;
            if (found >= 0 || val != -1.0 || (used & (UINT64_C(1)<<h))) return 0;
            found = h;
        }
        if (found < 0) return 0;
        used |= UINT64_C(1)<<found;
        arg[col+1] = epi[col+2] = -1.0;
    }
    int seen[(PORT_ARROW+1)*(PORT_ARROW+2)/2] = {0};
    for (int z = 0; z < triangle; z++) {
        int r = t->barA_con[z] - barrow, sm = t->barA_sym[z];
        if (r < 0 || r >= triangle || seen[r]++ || t->barA_bar[z] != 0 ||
            t->sym_dim[sm] != d || t->sym_nnz[sm] != 1 || t->sym_val[sm][0] != 1.0) return 0;
        int i = t->sym_subi[sm][0], j = t->sym_subj[sm][0];
        if (i < j) { int swap = i; i = j; j = swap; }
        if (r != i*(i+1)/2+j || t->barA_coef[z] != (i == j ? -1.0 : -0.5)) return 0;
    }
    p->psd_head = -t->blc[barrow];
    if (!(p->psd_head > 0.0) || !portfolio_row(a+(size_t)barrow*n,n,0,0)) return 0;
    p->scale_max = INFINITY;
    for (int i = 1; i < d; i++) for (int j = 0; j <= i; j++) {
        int r = barrow+i*(i+1)/2+j;
        const double *row = a+(size_t)r*n;
        double constant = -t->blc[r];
        if (j == 0) {
            /* Expand affine definitions for the off-diagonal arrow entries. */
            if (constant != 0.0) return 0;
            for (int col = 0; col < n; col++) {
                if (row[col] == 0.0) continue;
                if (col >= q+e && col < w) {
                    p->map[2][(i-1)*b+col-q-e] += row[col];
                } else if (col >= scale+1+q+e && col < scale+1+w) {
                    const double *def = a+(size_t)(at+col-scale-1)*n;
                    for (int h = 0; h < b; h++)
                        p->map[2][(i-1)*b+h] += row[col]*def[q+e+h];
                } else return 0;
            }
        } else {
            if (!portfolio_row(row,n,scale,1)) return 0;
            if (i <= b) {
                if (i == j ? constant <= 0.0 : constant != 0.0) return 0;
            } else if (constant != 0.0 ||
                       (i == j ? row[scale] <= 0.0 : row[scale] != 0.0)) return 0;
            p->diag[(i-1)*k+j-1] = p->diag[(j-1)*k+i-1] = constant;
            p->slope[(i-1)*k+j-1] = p->slope[(j-1)*k+i-1] = row[scale];
            if (i == j && row[scale] < 0.0)
                p->scale_max = fmin(p->scale_max,nextafter(constant/-row[scale],INFINITY));
        }
    }
    if (!(p->scale_max > 0.0) || !isfinite(p->scale_max)) return 0;
    p->size[0]=q; p->size[1]=e; p->size[2]=b;
    p->start[1]=q; p->start[2]=q+e;
    p->weights=w; p->cap=(int)capacity; p->scale=scale;
    p->qrow=qrow; p->base=base; p->dim=d; p->mass=mass; p->exp_budget=risk;
    return 1;
}

/* NNLS on the Gram matrix. The active set is only used to find a good
 * direction. The positive stationarity residual is included in every bound. */
static int portfolio_direction(int rows, int n, const double *a, const double *c,
                               unsigned mask, double *x, double *dual, double *residual) {
    double gram[PORT_BLOCK*PORT_BLOCK], factor[PORT_BLOCK*PORT_BLOCK];
    double *row[PORT_BLOCK], rhs[PORT_BLOCK];
    DMat mat = {n,n,row};
    for (int i = 0; i < n; i++) {
        row[i] = factor+i*n;
        for (int j = 0; j <= i; j++) {
            double q = 0.0;
            for (int r = 0; r < rows; r++) q += a[r*n+i]*a[r*n+j];
            gram[i*n+j] = gram[j*n+i] = q;
        }
    }
    unsigned active = mask;
    for (int step = 0; step <= 3*n; step++) {
        int idx[PORT_BLOCK], count = 0;
        for (int i = 0; i < n; i++) {
            x[i] = 0.0;
            if (active & (1u<<i)) idx[count++] = i;
        }
        for (int i = 0; i < count; i++) {
            rhs[i] = c[idx[i]];
            for (int j = 0; j <= i; j++) row[i][j] = gram[idx[i]*n+idx[j]];
        }
        if (count && (dmat_cholesky(&mat,count) || dmat_chol_solve(&mat,rhs,count))) return 0;
        int remove = -1;
        for (int i = 0; i < count; i++) {
            if (!isfinite(rhs[i])) return 0;
            x[idx[i]] = fmax(0.0,rhs[i]);
            if (rhs[i] < 0.0 && (remove < 0 || rhs[i] < rhs[remove])) remove = i;
        }
        if (remove >= 0 && step < 3*n) {
            active &= ~(1u<<idx[remove]);
            continue;
        }
        double error = 1e-13;
        int add = -1;
        for (int i = 0; i < n; i++) if ((mask & ~active) & (1u<<i)) {
            double r = c[i]-dvec_dot(n,gram+i*n,x);
            if (r > error) { error = r; add = i; }
        }
        if (add < 0 || step == 3*n) break;
        active |= 1u<<add;
    }
    for (int r = 0; r < rows; r++) {
        dual[r] = dvec_dot(n,a+r*n,x);
        if (!isfinite(dual[r])) return 0;
    }
    *residual = 0.0;
    for (int j = 0; j < n; j++) if (mask & (1u<<j)) {
        double v = c[j], sum = fabs(v);
        for (int r = 0; r < rows; r++) {
            double term = a[r*n+j]*dual[r];
            v -= term;
            sum += fabs(term);
        }
        if (!isfinite(sum)) return 0;
        *residual = fmax(*residual,v+64.0*DBL_EPSILON*sum);
    }
    return 1;
}

/* phi(z) = (1+z)log(1+z)+(1-z)log(1-z). As a function of z^2 it
 * is convex, so each tangent gives an ellipsoidal outer bound. */
static double portfolio_entropy(double z) {
    if (fabs(z) >= 1.0) return INFINITY;
    return log1p(-z*z)+z*(log1p(z)-log1p(-z));
}

static int portfolio_norm(const Portfolio *p, int block, const double *c,
                          unsigned mask, double incumbent, PortfolioValue *v) {
    int n = p->size[block];
    const double *a = p->map[block];
    double theta[PORT_BLOCK], z[PORT_BLOCK] = {0};
    for (int i = 0; i < n; i++) theta[i] = 1.0;
    double radius2 = block == 0 ? p->soc_radius*p->soc_radius : p->exp_budget;
    for (int step = 0; step < (block == 0 ? 1 : 8); step++) {
        double scaled[PORT_BLOCK*PORT_BLOCK], x[PORT_BLOCK], dual[PORT_BLOCK], residual;
        double budget = radius2;
        for (int i = 0; i < n; i++) {
            double quadratic = theta[i]*z[i]*z[i], entropy = portfolio_entropy(z[i]);
            budget += quadratic-entropy + 64.0*DBL_EPSILON*(quadratic+fabs(entropy));
            double root = sqrt(theta[i]);
            for (int j = 0; j < n; j++) scaled[i*n+j] = root*a[i*n+j];
        }
        if (!(budget > 0.0) ||
            !portfolio_direction(n,n,scaled,c,mask,x,dual,&residual)) return 0;
        double dual_norm = dvec_dot(n,dual,dual);
        double upper = sqrt(budget*dual_norm)*(1.0+64.0*DBL_EPSILON) + p->mass*residual;
        v->upper = step == 0 ? upper : fmin(v->upper,upper);
        if (incumbent >= v->upper || dual_norm == 0.0) break;
        double multiplier = sqrt(budget/dual_norm);
        for (int i = 0; i < n; i++) z[i] = dvec_dot(n,a+i*n,x)*multiplier;
        if (block == 1) {
            double lo = 0.0, hi = 1.0;
            for (int round = 0; round < 40; round++) {
                double mid = 0.5*(lo+hi), used = 0.0;
                for (int i = 0; i < n; i++) used += portfolio_entropy(mid*z[i]);
                if (used <= radius2) lo = mid;
                else hi = mid;
            }
            multiplier *= lo;
            for (int i = 0; i < n; i++) {
                z[i] *= lo;
                theta[i] = z[i] == 0.0 ? 1.0 : atanh(z[i])/z[i];
            }
        }
        multiplier *= 1.0-32.0*DBL_EPSILON;
        for (int i = 0; i < n; i++) x[i] *= multiplier;
        double lower = dvec_dot(n,c,x);
        if (lower > v->lower) {
            v->lower = lower;
            memcpy(v->x,x,(size_t)n*sizeof(double));
        }
        if (v->upper-v->lower <= 5e-11) break;
    }
    return isfinite(v->upper);
}

/* At s, D(s)=L L^T reduces the Schur complement to ||L^-1 Bx||^2<=h.
 * The lift y=L^-T dual gives a global bound through
 * (y^T Bx)^2 <= h y^T D(s') y for every feasible s'. */
static int portfolio_psd_point(const Portfolio *p, const double *c, unsigned mask,
                               double s, PortfolioValue *v, double *derivative) {
    int n = p->size[2], k = p->dim-1;
    double buf[PORT_ARROW*PORT_ARROW], *row[PORT_ARROW], a[PORT_ARROW*PORT_BLOCK];
    DMat mat = {k,k,row};
    for (int i = 0; i < k; i++) {
        row[i] = buf+i*k;
        for (int j = 0; j <= i; j++) row[i][j] = p->diag[i*k+j]+s*p->slope[i*k+j];
    }
    if (dmat_cholesky(&mat,k)) return 0;
    const double *b = p->map[2];
    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++) {
            double val = b[i*n+j];
            for (int h = 0; h < i; h++) val -= row[i][h]*a[h*n+j];
            a[i*n+j] = val/row[i][i];
        }
    double x[PORT_BLOCK], dual[PORT_ARROW], residual;
    if (!portfolio_direction(k,n,a,c,mask,x,dual,&residual)) return 0;
    double norm2 = dvec_dot(k,dual,dual);
    double multiplier = norm2 > 0.0 ? sqrt(p->psd_head/norm2)*(1.0-32.0*DBL_EPSILON) : 0.0;
    for (int i = 0; i < n; i++) x[i] *= multiplier;
    for (int i = k-1; i >= 0; i--) {
        for (int j = i+1; j < k; j++) dual[i] -= row[j][i]*dual[j];
        dual[i] /= row[i][i];
    }
    residual = 0.0;
    for (int j = 0; j < n; j++) if (mask & (1u<<j)) {
        double r = c[j], sum = fabs(r);
        for (int i = 0; i < k; i++) {
            double term = b[i*n+j]*dual[i];
            r -= term;
            sum += fabs(term);
        }
        if (!isfinite(sum)) return 0;
        residual = fmax(residual,r+64.0*DBL_EPSILON*sum);
    }
    double intercept = 0.0, slope = 0.0, magnitude = 0.0;
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++) {
            double pair = dual[i]*dual[j];
            double constant = p->diag[i*k+j]*pair, gradient = p->slope[i*k+j]*pair;
            intercept += constant;
            slope += gradient;
            magnitude += fabs(constant)+p->scale_max*fabs(gradient);
        }
    double max_risk = intercept + fmax(0.0,slope*p->scale_max) + 64.0*DBL_EPSILON*magnitude;
    double upper = sqrt(p->psd_head*fmax(0.0,max_risk))+p->mass*residual;
    if (!isfinite(upper)) return 0;
    v->upper = fmin(v->upper,upper);
    double lower = dvec_dot(n,c,x);
    if (lower >= v->lower) {
        v->lower = lower;
        v->scale = s;
        memcpy(v->x,x,(size_t)n*sizeof(double));
    }
    *derivative = slope;
    return 1;
}

static int portfolio_psd(const Portfolio *p, const double *c, unsigned mask,
                         double incumbent, PortfolioValue *v) {
    double lo = 0.0, hi = p->scale_max;
    v->upper = INFINITY;
    /* In the admitted pencils, the upper block is positive definite at
     * zero and the lower block is s times a positive diagonal. */
    for (int i = 0; i < 50; i++) {
        double s = 0.5*(lo+hi), derivative;
        if (!portfolio_psd_point(p,c,mask,s,v,&derivative)) { hi = s; continue; }
        if (derivative > 0.0) lo = s;
        else hi = s;
        if (v->upper <= incumbent || v->upper-v->lower <= 5e-11) break;
    }
    return isfinite(v->upper);
}

static int portfolio_live(PRIMALtask_t t) {
    return t->mip_deadline < 0.0 || (double)clock() < t->mip_deadline;
}

static int portfolio_group(PRIMALtask_t t, const Portfolio *p, int block, int shared,
                           int cap, int *visits, PortfolioValue *value) {
    int n = p->size[block], offset = p->start[block];
    double c[PORT_BLOCK];
    unsigned forbidden = 0, pay = 0;
    for (int j = 0; j < n; j++) {
        c[j] = -t->c[offset+j];
        if (p->gate[offset+j] != p->shared) pay |= 1u<<j;
        else if (!shared) forbidden |= 1u<<j;
    }
    for (unsigned mask = 0; mask < (1u<<n); mask++) {
        if (mask & forbidden) continue;
        int count = 0;
        for (unsigned bits = mask & pay; bits; bits &= bits-1) count++;
        if (count > cap) continue;
        if (++*visits > t->mip_max_nodes || !portfolio_live(t)) return 0;
        double incumbent = value[count].lower;
        PortfolioValue v = {0};
        int ok = block == 2 ? portfolio_psd(p,c,mask,incumbent,&v)
                            : portfolio_norm(p,block,c,mask,incumbent,&v);
        if (!ok) return 0;
        double upper = fmax(v.upper,value[count].upper);
        if (v.lower > incumbent || (block == 2 && mask == 0)) value[count] = v;
        value[count].upper = upper;
    }
    for (int k = 1; k <= cap; k++) {
        double upper = fmax(value[k].upper,value[k-1].upper);
        if (value[k-1].lower >= value[k].lower) value[k] = value[k-1];
        value[k].upper = upper;
    }
    return 1;
}

static int portfolio_publish(PRIMALtask_t t, const Portfolio *p, const double *a,
                             const double *weight, double scale, double bound) {
    int n = t->numvar, w = p->weights, d = p->dim, base = p->base;
    int q = p->size[0], e = p->size[1], erow = p->qrow+q+1, barrow = erow+6*e;
    double x[PORT_VARS] = {0}, bar[(PORT_ARROW+1)*(PORT_ARROW+1)];
    double tol = fmin(1e-8,fmin(t->mip_tol_feas,t->tol_co_pfeas));
    for (int j = 0; j < w; j++) {
        x[j] = weight[j];
        if (x[j] > 0.0) x[w+p->gate[j]] = 1.0;
    }
    x[p->scale] = scale;
    for (int j = 0; j < w; j++)
        x[p->scale+1+j] = dvec_dot(w,a+(size_t)(p->qrow-w+j)*n,x);
    for (int j = 0; j < 2*e; j++) {
        int r = erow+3*j;
        double u = 1.0+dvec_dot(base,a+(size_t)(r+1)*n,x);
        if (!(u > 0.0)) return 0;
        for (int col = base-2*e; col < base; col++)
            if (a[(size_t)(r+2)*n+col] != 0.0) x[col] = u*log1p(u-1.0);
    }
    for (int j = base; j < n; j++)
        x[j] = dvec_dot(base,a+(size_t)(p->qrow+j-base)*n,x)-t->blc[p->qrow+j-base];
    for (int i = 0; i < d; i++) for (int j = 0; j <= i; j++) {
        int r = barrow+i*(i+1)/2+j;
        bar[i*d+j] = bar[j*d+i] = dvec_dot(base,a+(size_t)r*n,x)-t->blc[r];
    }
    if (!(bar_min_eig(d,bar,NULL) >= -tol)) return 0;
    double lx[PORT_VARS], ux[PORT_VARS], lc[PORT_ROWS], uc[PORT_ROWS];
    for (int j = 0; j < n; j++) {
        bound_range(t->bkx[j],t->blx[j],t->bux[j],&lx[j],&ux[j]);
        if (!isfinite(x[j])) return 0;
    }
    for (int r = 0; r < t->numcon; r++)
        bound_range(t->bkc[r],t->blc[r],t->buc[r],&lc[r],&uc[r]);
    /* mip_point_measures considers the scalar part of a row. Move the
     * reconstructed bar entries to the right-hand side before that check. */
    for (int i = 0; i < d; i++) for (int j = 0; j <= i; j++) {
        int r = barrow+i*(i+1)/2+j;
        lc[r] += bar[i*d+j];
        uc[r] += bar[i*d+j];
    }
    if (!mip_point_measures(t,lx,ux,lc,uc,x,tol,t->mip_tol_inther)) return 0;
    double linear = dvec_dot(n,t->c,x), gap = linear+bound;
    double obj = t->cfix+linear;
    bound = nextafter(t->cfix-bound,-INFINITY);
    double limit = fmin(1e-8,t->tol_co_gap)*(1.0+fabs(obj)+fabs(bound));
    double mip_limit = fmax(t->mip_tol_abs_gap,t->mip_tol_rel_gap*(1.0+fabs(obj)));
    if (getenv("GMB_DBG"))
        fprintf(stderr,"  [portfolio] objective %.17g bound %.17g\n",obj,bound);
    if (!isfinite(obj) || !isfinite(bound) || !(gap >= 0.0 && gap <= fmin(limit,mip_limit)) ||
        !portfolio_live(t)) return 0;
    memcpy(t->x,x,(size_t)n*sizeof(double));
    memcpy(t->barx[0],bar,(size_t)d*d*sizeof(double));
    t->pobj=obj;
    t->dobj=bound;
    t->has_sol=1;
    t->solsta=PRIMAL_SOL_STA_INTEGER_OPTIMAL;
    tlog(t,"MIP: portfolio bound certified by separate conic risk blocks\n");
    return 1;
}

int mip_portfolio_solve(PRIMALtask_t t) {
    if (t->numbarvar != 1 || t->nbarC || t->numcones < 3 || t->has_qobj || t->has_qcon ||
        t->numsos || t->has_warm || t->cbfn || t->progcb || t->numvar > PORT_VARS ||
        t->numcon > PORT_ROWS || !portfolio_live(t)) return 0;
    Portfolio p = {0};
    double *a = calloc((size_t)t->numvar*t->numcon,sizeof(double));
    if (!a) return 0;
    int solved = 0;
    if (portfolio_read(t,&p,a)) {
        double weight[PORT_WEIGHTS] = {0}, scale = 0.0, best = -INFINITY, bound = 0.0;
        int visits = 0, ok = 1;
        for (int shared = 0; shared <= (p.shared >= 0) && shared <= p.cap; shared++) {
            PortfolioValue value[3][PORT_WEIGHTS+1] = {0};
            int cap = p.cap-shared;
            for (int b = 0; b < 3; b++)
                if (!portfolio_group(t,&p,b,shared,cap,&visits,value[b])) { ok = 0; break; }
            if (!ok) break;
            for (int i = 0; i <= cap; i++) for (int j = 0; j <= cap-i; j++) {
                int k = cap-i-j;
                bound = fmax(bound,value[0][i].upper+value[1][j].upper+value[2][k].upper);
                double total = value[0][i].lower+value[1][j].lower+value[2][k].lower;
                if (total <= best) continue;
                best = total;
                scale = value[2][k].scale;
                memcpy(weight,value[0][i].x,(size_t)p.size[0]*sizeof(double));
                memcpy(weight+p.start[1],value[1][j].x,(size_t)p.size[1]*sizeof(double));
                memcpy(weight+p.start[2],value[2][k].x,(size_t)p.size[2]*sizeof(double));
            }
        }
        /* Leave a rounding margin in the accumulated upper bound. */
        if (ok) solved = portfolio_publish(t,&p,a,weight,scale,bound+5e-13*(1.0+bound));
    }
    free(a);
    return solved;
}
