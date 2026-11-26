/*
 * PrimalSolver - a convex optimization solver in C99.
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 */

/* Binary selection in an affine PSD matrix
 *          [ I + sum_j x_j H_j   b ]
 *          [         b^T        t ] >= 0,
 * with objective min t and a cardinality constraint. A rank-one majorant
 * bounds subtrees. Full matrix arithmetic certifies the inspected supports. */
#include "primal_priv.h"
#include <stdint.h>

enum { SELECT_N = 32, SELECT_D = 48, SELECT_K = 8,
       GROUP_N = 64, GROUP_D = 16,
       SELECT_ROWS = (SELECT_D+1)*(SELECT_D+2)/2+2*GROUP_N+1 };

typedef struct {
    int n, d, cap, head;
    double *matrix;
    double b[SELECT_D], u[SELECT_N][SELECT_D], g[SELECT_N][SELECT_N], h[SELECT_N];
    double shift, upper, lower;
    uint32_t best;
    int visits, inspected;
    PRIMALtask_t task;
} PsdSelect;

static int select_live(const PsdSelect *p) {
    return p->task->mip_deadline < 0.0 || (double)clock() < p->task->mip_deadline;
}

/* Recover one affine PSD block from entrywise equalities, with exactly
 * one link per triangular entry. Negative row entries mark scalar rows. */
static int selection_links(PRIMALtask_t t, int dim, int *entry, double *constant) {
    int tri = dim*(dim+1)/2;
    if (t->nbarA != tri || t->numcon > SELECT_ROWS) return 0;
    int seen[(SELECT_D+1)*(SELECT_D+2)/2] = {0};
    for (int r = 0; r < t->numcon; r++) entry[r] = -1;
    for (int k = 0; k < tri; k++) {
        int sm = t->barA_sym[k], r = t->barA_con[k];
        if (t->barA_bar[k] != 0 || t->sym_dim[sm] != dim || t->sym_nnz[sm] != 1 ||
            t->sym_val[sm][0] != 1.0 || entry[r] != -1 || t->bkc[r] != PRIMAL_BK_FX)
            return 0;
        int i = t->sym_subi[sm][0], j = t->sym_subj[sm][0];
        if (i < j) { int tmp = i; i = j; j = tmp; }
        int index = i*(i+1)/2+j;
        if (seen[index]++ || t->barA_coef[k] != (i == j ? -1.0 : -0.5)) return 0;
        double v = -t->blc[r];
        if (!isfinite(v)) return 0;
        constant[i*dim+j] = constant[j*dim+i] = v;
        entry[r] = i*dim+j;
    }
    return 1;
}

static int selection_feasible(PRIMALtask_t t, const int *entry,
                              const double *x, const double *bar, double tol) {
    double lx[GROUP_N+1], ux[GROUP_N+1], lc[SELECT_ROWS], uc[SELECT_ROWS];
    for (int j = 0; j < t->numvar; j++) {
        if (!isfinite(x[j])) return 0;
        bound_range(t->bkx[j],t->blx[j],t->bux[j],&lx[j],&ux[j]);
    }
    for (int r = 0; r < t->numcon; r++) {
        bound_range(t->bkc[r],t->blc[r],t->buc[r],&lc[r],&uc[r]);
        if (entry[r] >= 0) {
            lc[r] += bar[entry[r]];
            uc[r] += bar[entry[r]];
        }
    }
    return mip_point_measures(t,lx,ux,lc,uc,x,tol,t->mip_tol_inther);
}

/* Read the entrywise equality representation emitted for an affine CBF
 * PSD constraint. The scalar rows must give all the binary upper bounds
 * and a single cardinality cap. */
static int select_read(PsdSelect *p) {
    PRIMALtask_t t = p->task;
    int n = p->n, d = p->d, dim = d+1, cols = n+1;
    int tri = dim*(dim+1)/2;
    if (t->numcon != n+1+tri || t->nbarA != tri) return 0;
    int col_to_item[SELECT_N+1], head = -1, count = 0;
    for (int j = 0; j < cols; j++) {
        col_to_item[j] = -1;
        if (t->c[j] == 1.0 && t->bkx[j] == PRIMAL_BK_FR &&
            t->vartype[j] == PRIMAL_VAR_TYPE_CONT && head < 0) {
            head = j;
        } else if (t->c[j] == 0.0 && t->bkx[j] == PRIMAL_BK_LO && t->blx[j] == 0.0 &&
                   (t->vartype[j] == PRIMAL_VAR_TYPE_INT || t->vartype[j] == PRIMAL_VAR_TYPE_INT_BIN)
                   && count < n) {
            col_to_item[j] = count++;
        } else return 0;
    }
    if (head < 0 || count != n) return 0;
    p->head = head;

    int row_entry[SELECT_ROWS];
    double constant[(SELECT_D+1)*(SELECT_D+1)];
    if (!selection_links(t,dim,row_entry,constant)) return 0;
    for (int i = 0; i < dim; i++) for (int j = 0; j <= i; j++) {
        double v = constant[i*dim+j];
        if (i == d && j < d) p->b[j] = v;
        else if (v != (i == j && i < d ? 1.0 : 0.0)) return 0;
    }
    double scalar[SELECT_N+1][SELECT_N+1] = {{0}};
    int scalar_count = 0, scalar_row[SELECT_N+1];
    for (int r = 0; r < t->numcon; r++) if (row_entry[r] == -1) {
        if (scalar_count > n || t->bkc[r] != PRIMAL_BK_LO) return 0;
        scalar_row[scalar_count] = r;
        row_entry[r] = -2-scalar_count++;
    }
    if (scalar_count != n+1) return 0;
    double head_coef = 0.0;
    for (int j = 0; j < cols; j++) {
        int item = col_to_item[j];
        const Col *c = &t->cols[j];
        for (int k = 0; k < c->nz; k++) {
            int entry = row_entry[c->sub[k]];
            double v = c->val[k];
            if (!isfinite(v)) return 0;
            if (v == 0.0) continue;
            if (entry < 0) {
                scalar[-entry-2][j] += v;
                continue;
            }
            int a = entry/dim, b = entry%dim;
            if (j == head) {
                if (a != d || b != d) return 0;
                head_coef += v;
            } else {
                if (a >= d) return 0;
                double *mat = p->matrix+(size_t)item*d*d;
                mat[a*d+b] += v;
                if (a != b) mat[b*d+a] += v;
            }
        }
    }
    if (head_coef != 1.0) return 0;
    int cap = -1;
    uint32_t uppers = 0;
    for (int i = 0; i < scalar_count; i++) {
        if (scalar[i][head] != 0.0) return 0;
        int used = 0, last = -1;
        for (int j = 0; j < cols; j++) {
            if (scalar[i][j] == 0.0) continue;
            if (scalar[i][j] != -1.0) return 0;
            used++;
            last = col_to_item[j];
        }
        double rhs = -t->blc[scalar_row[i]];
        if (used == 1 && last >= 0 && rhs == 1.0 &&
            !(uppers & (UINT32_C(1)<<last))) uppers |= UINT32_C(1)<<last;
        else if (used == n && cap < 0 && rhs >= 0.0 && rhs <= SELECT_K && rhs == floor(rhs))
            cap = (int)rhs;
        else return 0;
    }
    if (cap < 0) return 0;
    p->cap = cap < n ? cap : n;
    return 1;
}

/* Bound each rounded contribution by u_j u_j^T +/- e_j I via the
 * maximum absolute row sum of the symmetric difference. */
static int select_majorant(PsdSelect *p) {
    int n = p->n, d = p->d;
    double errors[SELECT_N];
    for (int k = 0; k < n; k++) {
        const double *mat = p->matrix+(size_t)k*d*d;
        int pivot = 0;
        for (int i = 1; i < d; i++) if (mat[i*d+i] > mat[pivot*d+pivot]) pivot = i;
        double pivot_value = mat[pivot*d+pivot];
        if (!(pivot_value > 0.0) || !isfinite(pivot_value)) return 0;
        for (int i = 0; i < d; i++) p->u[k][i] = mat[pivot*d+i]/sqrt(pivot_value);
        double error = 0.0;
        for (int i = 0; i < d; i++) {
            double row = 0.0, mag = 0.0;
            for (int j = 0; j < d; j++) {
                double product = p->u[k][i]*p->u[k][j];
                row += fabs(mat[i*d+j]-product);
                mag += fabs(mat[i*d+j])+fabs(product);
            }
            if (!isfinite(mag)) return 0;
            error = fmax(error,row+128.0*DBL_EPSILON*(1.0+mag));
        }
        int i = k;
        for (; i > 0 && error > errors[i-1]; i--) errors[i] = errors[i-1];
        errors[i] = error;
    }
    for (int i = 0; i < p->cap; i++) p->shift += errors[i];
    /* Reject matrices for which this bound is too weak to verify SPD. */
    p->shift += 128.0*DBL_EPSILON;
    if (!(p->shift < 0.1)) return 0;
    double scale = 1.0+p->shift;
    for (int i = 0; i < n; i++) {
        p->h[i] = dvec_dot(d,p->u[i],p->b)/scale;
        for (int j = 0; j <= i; j++)
            p->g[i][j] = p->g[j][i] = dvec_dot(d,p->u[i],p->u[j])/scale;
    }
    return 1;
}

static void select_matrix(const PsdSelect *p, int depth, const int *items, double *mat) {
    int d = p->d;
    memset(mat,0,(size_t)d*d*sizeof(double));
    for (int i = 0; i < d; i++) mat[i*d+i] = 1.0;
    for (int k = 0; k < depth; k++) {
        const double *value = p->matrix+(size_t)items[k]*d*d;
        for (int ij = 0; ij < d*d; ij++) mat[ij] += value[ij];
    }
}

/* For any y, q(y)=2 b^T y-y^T A y is a lower bound on b^T A^-1 b.
 * Its error is r^T A^-1 r, bounded with A >= (1-shift) I. */
static int select_exact(PsdSelect *p, int depth, const int *items, uint32_t mask) {
    int d = p->d;
    double mat[SELECT_D*SELECT_D], factor[SELECT_D*SELECT_D], *row[SELECT_D], y[SELECT_D];
    select_matrix(p,depth,items,mat);
    memcpy(factor,mat,(size_t)d*d*sizeof(double));
    memcpy(y,p->b,(size_t)d*sizeof(double));
    for (int i = 0; i < d; i++) row[i] = factor+i*d;
    DMat chol = {d,d,row};
    if (dmat_cholesky(&chol,d) || dmat_chol_solve(&chol,y,d)) return 0;
    double val = 0.0, error2 = 0.0, margin = 0.0;
    for (int i = 0; i < d; i++) {
        double r = p->b[i], mag = fabs(r), additions = 0.0;
        for (int j = 0; j < d; j++) {
            double term = mat[i*d+j]*y[j];
            r -= term;
            mag += fabs(term);
        }
        /* Charge rounding in the accumulation of the original matrix
         * using the magnitudes of its individual contributions. */
        for (int k = 0; k < depth; k++) {
            const double *a = p->matrix+((size_t)items[k]*d+i)*d;
            for (int j = 0; j < d; j++) additions += fabs(a[j]*y[j]);
        }
        double err = 256.0*DBL_EPSILON*(1.0+mag+additions);
        double term = y[i]*(p->b[i]+r);
        val += term;
        error2 += (fabs(r)+err)*(fabs(r)+err);
        margin += fabs(y[i])*err+128.0*DBL_EPSILON*fabs(term);
    }
    margin += 1e-13*(1.0+fabs(val));
    double lower = val-margin, upper = val+margin+error2/(1.0-p->shift);
    if (!isfinite(lower) || !isfinite(upper)) return 0;
    p->lower = fmin(p->lower,lower);
    if (upper < p->upper) { p->upper = upper; p->best = mask; }
    p->inspected++;
    return 1;
}

static double select_score(const PsdSelect *p, int j, const double *y) {
    double v = 0.0, mag = 0.0;
    for (int i = 0; i < p->d; i++) {
        double term = p->u[j][i]*y[i];
        v += term;
        mag += fabs(term);
    }
    v = fabs(v)+128.0*DBL_EPSILON*mag;
    return v*v*(1.0+16.0*DBL_EPSILON);
}

static int select_search(PsdSelect *p, int depth, int *items, uint32_t chosen,
                         int remaining, const int *list) {
    if (p->visits >= p->task->mip_max_nodes || (++p->visits%64 == 0 && !select_live(p)))
        return 0;
    double z[SELECT_K], *row[SELECT_K], small[SELECT_K][SELECT_K];
    for (int i = 0; i < depth; i++) {
        row[i] = small[i];
        z[i] = p->h[items[i]];
        for (int j = 0; j <= i; j++) small[i][j] = p->g[items[i]][items[j]]+(i == j);
    }
    DMat chol = {depth,depth,row};
    if (depth && (dmat_cholesky(&chol,depth) || dmat_chol_solve(&chol,z,depth))) return 0;
    double y[SELECT_D], lower = 0.0, mag = 0.0;
    for (int i = 0; i < p->d; i++) {
        double v = p->b[i];
        for (int j = 0; j < depth; j++) v -= p->u[items[j]][i]*z[j];
        y[i] = v/(1.0+p->shift);
        double xy = 2.0*p->b[i]*y[i], yy = (1.0+p->shift)*y[i]*y[i];
        lower += xy-yy;
        mag += fabs(xy)+yy;
    }
    for (int i = 0; i < depth; i++) {
        double score = select_score(p,items[i],y);
        lower -= score;
        mag += score;
    }
    if (!isfinite(lower) || !isfinite(mag)) return 0;
    double here = lower-256.0*DBL_EPSILON*(1.0+mag);
    int capacity = p->cap-depth;
    if (!capacity || !remaining) {
        if (here >= p->upper) { p->lower = fmin(p->lower,here); return 1; }
        return select_exact(p,depth,items,chosen);
    }
    /* Any completion can add at most the largest remaining scores.
     * Explore those choices first to improve the upper bound early. */
    double score[SELECT_N];
    int order[SELECT_N];
    for (int k = 0; k < remaining; k++) {
        double gain = select_score(p,list[k],y);
        if (!isfinite(gain)) return 0;
        int i = k;
        for (; i > 0 && gain > score[i-1]; i--) {
            score[i] = score[i-1]; order[i] = order[i-1];
        }
        score[i] = gain; order[i] = list[k];
    }
    double gain = 0.0;
    for (int k = 0; k < remaining && k < capacity; k++) gain += score[k];
    double bound = here-gain*(1.0+32.0*DBL_EPSILON);
    if (bound >= p->upper) { p->lower = fmin(p->lower,bound); return 1; }
    /* Cover the current support, then partition the nonempty extensions
     * by their first item in this node's order. */
    if (here < p->upper) {
        if (!select_exact(p,depth,items,chosen)) return 0;
    } else p->lower = fmin(p->lower,here);
    for (int i = 0; i < remaining; i++) {
        items[depth] = order[i];
        if (!select_search(p,depth+1,items,chosen | (UINT32_C(1)<<order[i]),
                           remaining-i-1,order+i+1)) return 0;
    }
    return 1;
}

static int select_publish(PsdSelect *p) {
    PRIMALtask_t t = p->task;
    int d = p->d, dim = d+1, n = p->n;
    double pobj = t->cfix+p->upper, dobj = nextafter(t->cfix+p->lower,-INFINITY);
    double gap = p->upper-p->lower;
    double tolerance = fmin(1e-8,t->tol_co_gap)*(1.0+fabs(pobj)+fabs(dobj));
    double mip_gap = fmax(t->mip_tol_abs_gap,t->mip_tol_rel_gap*(1.0+fabs(pobj)));
    if (!isfinite(pobj) || !isfinite(dobj) || !(gap >= 0.0 && gap <= fmin(tolerance,mip_gap)))
        return 0;
    double x[SELECT_N+1] = {0}, bar[(SELECT_D+1)*(SELECT_D+1)], block[SELECT_D*SELECT_D];
    int items[SELECT_K], count = 0;
    for (int j = 0, k = 0; j <= n; j++) {
        if (j == p->head) x[j] = p->upper;
        else {
            if (p->best & (UINT32_C(1)<<k)) { x[j] = 1.0; items[count++] = k; }
            k++;
        }
    }
    if (count > p->cap) return 0;
    select_matrix(p,count,items,block);
    for (int i = 0; i < d; i++) {
        memcpy(bar+i*dim,block+i*d,(size_t)d*sizeof(double));
        bar[i*dim+d] = bar[d*dim+i] = p->b[i];
    }
    bar[d*dim+d] = p->upper;
    double ftol = fmin(1e-8,fmin(t->mip_tol_feas,t->tol_co_pfeas));
    int entry[SELECT_ROWS];
    double constant[(SELECT_D+1)*(SELECT_D+1)];
    double eig = bar_min_eig(dim,bar,NULL);
    if (!isfinite(eig) || !(eig >= -ftol) ||
        !selection_links(t,dim,entry,constant) ||
        !selection_feasible(t,entry,x,bar,ftol) || !select_live(p)) return 0;
    memcpy(t->x,x,(size_t)(n+1)*sizeof(double));
    memcpy(t->barx[0],bar,(size_t)dim*dim*sizeof(double));
    t->pobj = pobj;
    t->dobj = dobj;
    t->has_sol = 1;
    t->solsta = PRIMAL_SOL_STA_INTEGER_OPTIMAL;
    tlog(t,"MIP: PSD selection bound certified by rank-one majorants\n");
    return 1;
}

static int select_lowrank(PRIMALtask_t t, int n, int d) {
    if (n > SELECT_N || d < 1 || d > SELECT_D) return 0;
    PsdSelect p = {0};
    p.n = n; p.d = d; p.task = t;
    p.upper = p.lower = INFINITY;
    p.matrix = calloc((size_t)n*d*d,sizeof(double));
    if (!p.matrix) return 0;
    int solved = 0;
    if (select_live(&p) && select_read(&p) && select_majorant(&p)) {
        int items[SELECT_K], list[SELECT_N];
        for (int i = 0; i < n; i++) list[i] = i;
        solved = select_search(&p,0,items,0,n,list) && select_publish(&p);
        if (getenv("GMB_DBG"))
            fprintf(stderr,"  [psd-selection] nodes=%d exact=%d lower=%.17g upper=%.17g solved=%d\n",
                    p.visits,p.inspected,p.lower,p.upper,solved);
    }
    free(p.matrix);
    return solved;
}

/* In the second structure, each binary belongs to a disjoint at-most-one
 * group. The only continuous variable has an upper bound and a PSD
 * coefficient. Fixing it at its upper bound preserves feasibility. */
typedef struct {
    PRIMALtask_t t;
    int n, d, head, entry[SELECT_ROWS];
    double cap, base[GROUP_D*GROUP_D], *mat, magnitude[GROUP_N+1], tol;
    int order[GROUP_N], group[GROUP_N+1];
} PsdGroups;

typedef struct {
    uint64_t mask;
    double cost;
    int last;
} Selection;

static int groups_read(PsdGroups *p) {
    PRIMALtask_t t = p->t;
    int n = p->n, d = p->d, dim2 = d*d, cols = n+1;
    int scalar_count = t->numcon-d*(d+1)/2;
    if (scalar_count < n+2 || scalar_count > 2*n+1 ||
        !selection_links(t,d,p->entry,p->base)) return 0;
    p->head = -1;
    for (int j = 0; j < cols; j++) {
        if (!isfinite(t->c[j])) return 0;
        if (t->c[j] == 0.0 && t->bkx[j] == PRIMAL_BK_FR &&
            t->vartype[j] == PRIMAL_VAR_TYPE_CONT && p->head < 0) p->head = j;
        else if (!(t->c[j] > 0.0 && t->bkx[j] == PRIMAL_BK_LO && t->blx[j] == 0.0 &&
                   (t->vartype[j] == PRIMAL_VAR_TYPE_INT || t->vartype[j] == PRIMAL_VAR_TYPE_INT_BIN)))
            return 0;
        p->group[j] = -1;
    }
    if (p->head < 0) return 0;
    double scalar[2*GROUP_N+1][GROUP_N+1] = {{0}};
    int scalar_row[2*GROUP_N+1], rs = 0;
    for (int r = 0; r < t->numcon; r++) if (p->entry[r] < 0) {
        if (t->bkc[r] != PRIMAL_BK_LO || !isfinite(t->blc[r])) return 0;
        scalar_row[rs] = r;
        p->entry[r] = -2-rs++;
    }
    for (int j = 0; j < cols; j++) for (int k = 0; k < t->cols[j].nz; k++) {
        int v = p->entry[t->cols[j].sub[k]];
        double a = t->cols[j].val[k];
        if (!isfinite(a)) return 0;
        if (v < 0) scalar[-v-2][j] += a;
        else {
            double *mat = p->mat+(size_t)j*dim2;
            mat[v] += a;
            if (v/d != v%d) mat[(v%d)*d+v/d] += a;
        }
    }
    uint64_t upper = 0;
    int cap_seen = 0, groups = 0;
    for (int r = 0; r < scalar_count; r++) {
        int count = 0, last = -1;
        for (int j = 0; j < cols; j++) {
            if (scalar[r][j] == 0.0) continue;
            if (scalar[r][j] != -1.0) return 0;
            count++; last = j;
        }
        double rhs = -t->blc[scalar_row[r]];
        if (count == 1 && last == p->head && !cap_seen) {
            p->cap = rhs; cap_seen = 1;
        } else if (count == 1 && last != p->head && rhs == 1.0) {
            int k = last-(last > p->head);
            if (upper & (UINT64_C(1)<<k)) return 0;
            upper |= UINT64_C(1)<<k;
        } else if (count >= 2 && scalar[r][p->head] == 0.0 && rhs == 1.0) {
            for (int j = 0; j < cols; j++) if (scalar[r][j] == -1.0) {
                if (p->group[j] != -1) return 0;
                p->group[j] = groups;
            }
            groups++;
        } else return 0;
    }
    if (!cap_seen || !groups) return 0;
    for (int j = 0, k = 0; j < cols; j++) {
        if (j == p->head) continue;
        if (p->group[j] < 0 || !(upper & (UINT64_C(1)<<k))) return 0;
        int i = k++;
        for (; i > 0 && t->c[j] < t->c[p->order[i-1]]; i--) p->order[i] = p->order[i-1];
        p->order[i] = j;
    }
    const double *slope = p->mat+(size_t)p->head*dim2;
    for (int k = 0; k < dim2; k++) {
        if (!(slope[k] >= 0.0) || (k/d != k%d && slope[k] != 0.0)) return 0;
        p->base[k] += p->cap*slope[k];
        p->magnitude[p->head] = fmax(p->magnitude[p->head],fabs(p->base[k])+fabs(p->cap*slope[k]));
    }
    for (int j = 0; j < cols; j++) if (j != p->head) {
        const double *mat = p->mat+(size_t)j*dim2;
        for (int k = 0; k < dim2; k++) {
            if (!isfinite(mat[k])) return 0;
            p->magnitude[j] = fmax(p->magnitude[j],fabs(mat[k]));
        }
    }
    return isfinite(p->magnitude[p->head]);
}

static int group_live(const PsdGroups *p) {
    return p->t->mip_deadline < 0.0 || (double)clock() < p->t->mip_deadline;
}

static int group_next(const PsdGroups *p, int start, uint64_t groups) {
    for (int j = start; j < p->n; j++)
        if (!(groups & (UINT64_C(1)<<p->group[p->order[j]]))) return j;
    return p->n;
}

static void group_push(Selection *heap, int *size, Selection item) {
    int i = (*size)++;
    for (; i && heap[(i-1)/2].cost > item.cost; i = (i-1)/2) heap[i] = heap[(i-1)/2];
    heap[i] = item;
}

static Selection group_pop(Selection *heap, int *size) {
    Selection first = heap[0], item = heap[--*size];
    int i = 0;
    for (int child = 1; child < *size; child = 2*i+1) {
        if (child+1 < *size && heap[child+1].cost < heap[child].cost) child++;
        if (heap[child].cost >= item.cost) break;
        heap[i] = heap[child]; i = child;
    }
    if (*size) heap[i] = item;
    return first;
}

/* A negative principal minor or a verified negative Rayleigh quotient
 * excludes the support. If the eigensolver leaves the claim ambiguous,
 * decline the specialized route. */
static int group_psd(const PsdGroups *p, const double *mat, double rounding) {
    int d = p->d;
    if (!isfinite(rounding)) return -1;
    for (int i = 0; i < d; i++) {
        double a = mat[i*d+i]+rounding;
        if (a < 0.0) return 0;
        for (int j = 0; j < i; j++) {
            double b = fmax(0.0,mat[j*d+j]+rounding);
            double off = fmax(0.0,fabs(mat[i*d+j])-rounding);
            double lhs = off*off, rhs = a*b;
            if (lhs-rhs > 16.0*DBL_EPSILON*(lhs+rhs)) return 0;
        }
    }
    double eig[GROUP_D], vectors[GROUP_D*GROUP_D], v[GROUP_D];
    for (int i = 0; i < d; i++) eig[i] = NAN;
    dmat_eig_jacobi(d,mat,eig,vectors);
    int min = 0;
    for (int i = 0; i < d; i++) {
        if (!isfinite(eig[i])) return -1;
        if (eig[i] < eig[min]) min = i;
    }
    if (rounding <= 0.1*p->tol && eig[min] >= -p->tol+rounding) return 1;
    for (int i = 0; i < d; i++) v[i] = vectors[i*d+min];
    double form = 0.0, mag = 0.0, norm1 = 0.0;
    for (int i = 0; i < d; i++) {
        norm1 += fabs(v[i]);
        for (int j = 0; j < d; j++) {
            double term = mat[i*d+j]*v[i]*v[j];
            form += term; mag += fabs(term);
        }
    }
    double margin = rounding*norm1*norm1 + 512.0*DBL_EPSILON*mag;
    return form < -margin ? 0 : -1;
}

static int groups_search(PsdGroups *p) {
    PRIMALtask_t t = p->t;
    int limit = t->mip_max_nodes < 100000 ? t->mip_max_nodes : 100000;
    if (limit <= 0) return 0;
    Selection *heap = malloc(((size_t)limit+1)*sizeof(*heap));
    if (!heap) return 0;
    int size = 0, solved = 0, dim2 = p->d*p->d;
    group_push(heap,&size,(Selection){0,0.0,-1});
    for (int visit = 0; visit < limit && size && group_live(p); visit++) {
        Selection s = group_pop(heap,&size);
        double x[GROUP_N+1] = {0}, bar[GROUP_D*GROUP_D];
        x[p->head] = p->cap;
        memcpy(bar,p->base,(size_t)dim2*sizeof(double));
        double magnitude = p->magnitude[p->head];
        double prefix_cost = 0.0, cost = 0.0;
        uint64_t prefix_groups = 0, all_groups = 0, prefix = s.mask;
        for (int j = 0; j <= s.last; j++) if (s.mask & (UINT64_C(1)<<j)) {
            int col = p->order[j];
            x[col] = 1.0;
            prefix_cost = cost; prefix_groups = all_groups;
            cost += t->c[col];
            all_groups |= UINT64_C(1)<<p->group[col];
            const double *a = p->mat+(size_t)col*dim2;
            for (int k = 0; k < dim2; k++) bar[k] += a[k];
            magnitude += p->magnitude[col];
        }
        int psd = group_psd(p,bar,256.0*DBL_EPSILON*(1.0+magnitude));
        if (psd < 0) break;
        if (psd) {
            /* The heap order covers every cheaper support. The positive
             * costs make both descendants cost-monotone. Round the
             * global bound outward before publishing it. */
            double pobj = t->cfix+cost, error = 512.0*DBL_EPSILON*(1.0+fabs(cost));
            double dobj = nextafter(t->cfix+(cost-error),-INFINITY);
            double gap = pobj-dobj;
            double limit_gap = fmin(fmin(1e-8,t->tol_co_gap)*(1.0+fabs(pobj)+fabs(dobj)),
                                    fmax(t->mip_tol_abs_gap,t->mip_tol_rel_gap*(1.0+fabs(pobj))));
            if (!(gap >= 0.0 && gap <= limit_gap) ||
                !selection_feasible(t,p->entry,x,bar,p->tol) || !group_live(p)) break;
            memcpy(t->x,x,(size_t)t->numvar*sizeof(double));
            memcpy(t->barx[0],bar,(size_t)dim2*sizeof(double));
            t->pobj = pobj; t->dobj = dobj; t->has_sol = 1;
            t->solsta = PRIMAL_SOL_STA_INTEGER_OPTIMAL;
            tlog(t,"MIP: PSD group selection optimality certified in cost order\n");
            if (getenv("GMB_DBG"))
                fprintf(stderr,"  [psd-groups] supports=%d bound=%.17g\n",visit+1,dobj);
            solved = 1;
            break;
        }
        int next = group_next(p,s.last+1,all_groups);
        if (next < p->n)
            group_push(heap,&size,(Selection){s.mask | (UINT64_C(1)<<next),cost+t->c[p->order[next]],next});
        if (s.last >= 0) {
            prefix &= ~(UINT64_C(1)<<s.last);
            next = group_next(p,s.last+1,prefix_groups);
            if (next < p->n)
                group_push(heap,&size,(Selection){prefix | (UINT64_C(1)<<next),
                                                 prefix_cost+t->c[p->order[next]],next});
        }
    }
    free(heap);
    return solved;
}

int mip_psd_select_solve(PRIMALtask_t t) {
    int n = t->numvar-1;
    if (n < 2 || n > GROUP_N || t->numbarvar != 1 || t->nbarC || t->numcones ||
        t->has_qobj || t->has_qcon || t->numsos || t->has_warm || t->cbfn || t->progcb)
        return 0;
    int d = t->barDim[0];
    if (d < 2 || d > SELECT_D+1) return 0;
    if (select_lowrank(t,n,d-1)) return 1;
    if (d > GROUP_D) return 0;
    PsdGroups p = {0};
    p.t = t; p.n = n; p.d = d;
    p.tol = fmin(1e-8,fmin(t->mip_tol_feas,t->tol_co_pfeas));
    p.mat = calloc((size_t)(n+1)*d*d,sizeof(double));
    if (!p.mat) return 0;
    int solved = group_live(&p) && groups_read(&p) && groups_search(&p);
    free(p.mat);
    return solved;
}
