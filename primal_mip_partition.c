/*
 * PrimalSolver - a convex optimization solver in C99.
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 */
#include "primal_priv.h"
#include <stdint.h>

enum { PART_VERTICES = 16, PART_EDGES = 120, PART_ROWS = 320, PART_GROUPS = 5 };

typedef struct {
    int n, edges, groups;
    int pair[PART_VERTICES*PART_VERTICES], first[PART_EDGES], second[PART_EDGES];
    int bar_entry[PART_ROWS], allowed[PART_EDGES];
    int64_t cost[PART_VERTICES*PART_VERTICES];
    double row_sign[PART_ROWS], lo[PART_ROWS], up[PART_ROWS];
    double *a, within, across;
} Partition;

static int partition_time_left(PRIMALtask_t t) {
    return t->mip_deadline < 0.0 || (double)clock() < t->mip_deadline;
}

static int partition_bit(unsigned mask) {
    int j = 0;
    while (!(mask & (1u << j))) j++;
    return j;
}

static int partition_read(PRIMALtask_t t, Partition *p) {
    int n = p->n, nv = p->edges;
    for (int j = 0; j < n*n; j++) p->pair[j] = -1;
    for (int j = 0; j < nv; j++) p->first[j] = -1;
    for (int r = 0; r < t->numcon; r++) {
        p->bar_entry[r] = -1;
        bound_range(t->bkc[r],t->blc[r],t->buc[r],&p->lo[r],&p->up[r]);
        if (!(p->lo[r] <= p->up[r])) return 0;
    }
    double lo[PART_EDGES], hi[PART_EDGES];
    for (int j = 0; j < nv; j++) {
        if (t->vartype[j] != PRIMAL_VAR_TYPE_INT && t->vartype[j] != PRIMAL_VAR_TYPE_INT_BIN)
            return 0;
        if (!(fabs(t->c[j]) <= 1e9 && floor(t->c[j]) == t->c[j])) return 0;
        bound_range(t->bkx[j],t->blx[j],t->bux[j],&lo[j],&hi[j]);
        if (!(lo[j] <= hi[j])) return 0;
        if (t->vartype[j] == PRIMAL_VAR_TYPE_INT_BIN) {
            lo[j] = fmax(0.0,lo[j]);
            hi[j] = fmin(1.0,hi[j]);
        }
        const Col *c = &t->cols[j];
        for (int k = 0; k < c->nz; k++) {
            double *v = p->a+(size_t)c->sub[k]*nv+j;
            if (*v != 0.0 || !isfinite(c->val[k])) return 0;
            *v = c->val[k];
        }
    }
    for (int k = 0; k < t->nbarA; k++) {
        int r = t->barA_con[k], sym = t->barA_sym[k];
        double weight = t->barA_coef[k];
        if (p->bar_entry[r] >= 0 || t->sym_nnz[sym] != 1 ||
            (fabs(weight) != 1.0 && fabs(weight) != 0.5)) return 0;
        int i = t->sym_subi[sym][0], j = t->sym_subj[sym][0];
        if (i < 0 || i >= n || j < 0 || j >= n) return 0;
        double scale = (i == j ? weight : 2.0*weight)*t->sym_val[sym][0];
        if (fabs(scale) != 1.0) return 0;
        p->bar_entry[r] = i*n+j;
        p->row_sign[r] = scale;
    }

    int links = 0;
    unsigned diagonal = 0;
    double slope = 0.0;
    for (int r = 0; r < t->numcon; r++) {
        const double *a = p->a+(size_t)r*nv;
        int var = -1, count = 0, entry = p->bar_entry[r];
        for (int j = 0; j < nv; j++) if (a[j] != 0.0) { var = j; count++; }
        if (entry < 0) {
            if (count == 1 && fabs(a[var]) == 1.0) {
                double sign = a[var];
                lo[var] = fmax(lo[var],sign > 0.0 ? p->lo[r] : -p->up[r]);
                hi[var] = fmin(hi[var],sign > 0.0 ? p->up[r] : -p->lo[r]);
            }
            continue;
        }
        if (p->lo[r] != p->up[r]) return 0;
        double rhs = p->lo[r]*p->row_sign[r];
        int i = entry/n, j = entry%n;
        if (i == j) {
            if (count || rhs != 1.0 || (diagonal & (1u << i))) return 0;
            diagonal |= 1u << i;
            continue;
        }
        if (count != 1 || p->first[var] >= 0 || p->pair[i*n+j] >= 0) return 0;
        double coefficient = -a[var]*p->row_sign[r];
        if (!links) { p->across = rhs; slope = coefficient; }
        if (rhs != p->across || coefficient != slope) return 0;
        p->pair[i*n+j] = p->pair[j*n+i] = var;
        p->first[var] = i;
        p->second[var] = j;
        p->cost[i*n+j] = p->cost[j*n+i] = (int64_t)t->c[var];
        links++;
    }
    if (diagonal != (1u << n)-1u || links != nv) return 0;
    for (int j = 0; j < nv; j++) {
        double low = ceil(lo[j]), high = floor(hi[j]);
        if (!(low >= 0.0 && high <= 1.0 && low <= high)) return 0;
        p->allowed[j] = (low == 0.0 ? 1 : 0) | (high == 1.0 ? 2 : 0);
    }

    p->within = p->across + slope;
    if (!(fabs(p->within-1.0) <= 1e-8 && p->across < 0.0 && p->across >= -1.0 &&
          1.0 + p->across - 2.0*p->within*p->within < -1e-8)) return 0;
    /* A nontransitive triple has a negative principal minor. For g+1
     * distinct groups, the all-ones direction on representatives
     * has a negative quadratic form when 1+g*across < 0. */
    for (int g = 1; g <= PART_GROUPS && g < n; g++)
        if (1.0+g*p->across < -1e-8) { p->groups = g; return 1; }
    return 0;
}

static int partition_feasible_blocks(PRIMALtask_t t, const Partition *p,
                                      unsigned char *valid, int64_t *sums) {
    unsigned size = 1u << p->n;
    memset(valid,1,size);
    for (int e = 0; e < p->edges; e++) if (p->allowed[e] != 3) {
        unsigned pair = (1u << p->first[e]) | (1u << p->second[e]);
        for (unsigned mask = 1; mask < size; mask++) {
            unsigned member = mask & pair;
            if ((p->allowed[e] == 1 && member == pair) ||
                (p->allowed[e] == 2 && member && member != pair)) valid[mask] = 0;
        }
    }
    for (int r = 0; r < t->numcon; r++) {
        if (p->bar_entry[r] >= 0) continue;
        const double *a = p->a+(size_t)r*p->edges;
        unsigned common = size-1;
        int64_t min = 0, max = 0;
        for (int e = 0; e < p->edges && common; e++) if (a[e] != 0.0) {
            if (!(fabs(a[e]) <= 1e9 && floor(a[e]) == a[e])) { common = 0; break; }
            common &= (1u << p->first[e]) | (1u << p->second[e]);
            if (a[e] < 0.0) min += (int64_t)a[e];
            else max += (int64_t)a[e];
        }
        if (!common || (min >= p->lo[r] && max <= p->up[r])) continue;
        if (!partition_time_left(t)) return 0;
        int anchor = partition_bit(common);
        sums[0] = 0;
        for (int j = 0; j < p->n; j++) {
            unsigned bit = 1u << j;
            int64_t term = j == anchor ? 0 : (int64_t)a[p->pair[anchor*p->n+j]];
            for (unsigned k = 0; k < bit; k++) sums[bit+k] = sums[k]+term;
        }
        for (unsigned mask = 1; mask < size; mask++)
            if ((mask & (1u << anchor)) && (sums[mask] < p->lo[r] || sums[mask] > p->up[r]))
                valid[mask] = 0;
    }
    return 1;
}

static int partition_search(PRIMALtask_t t, const Partition *p, int *labels, int64_t *minimum) {
    unsigned size = 1u << p->n;
    const int64_t inf = INT64_MAX/4;
    unsigned char *valid = malloc(size);
    int64_t *cost = malloc((size_t)3*size*sizeof(*cost));
    uint16_t *pick = calloc((size_t)(p->groups+1)*size,sizeof(*pick));
    int solved = 0;
    if (!valid || !cost || !pick || !partition_feasible_blocks(t,p,valid,cost)) goto done;
    int64_t *prev = cost+size, *next = prev+size;
    cost[0] = prev[0] = 0;
    for (unsigned mask = 1; mask < size; mask++) {
        int i = partition_bit(mask);
        unsigned rest = mask & (mask-1u);
        int64_t value = cost[rest];
        for (int j = i+1; j < p->n; j++) if (rest & (1u << j)) value += p->cost[i*p->n+j];
        cost[mask] = value;
        prev[mask] = valid[mask] ? value : inf;
        pick[size+mask] = (uint16_t)mask;
    }
    for (int g = 2; g <= p->groups; g++) {
        next[0] = 0;
        for (unsigned mask = 1; mask < size; mask++) {
            if ((mask & 127u) == 0 && !partition_time_left(t)) goto done;
            unsigned rest = mask & (mask-1u);
            int64_t best = inf;
            for (unsigned tail = rest;; tail = (tail-1u) & rest) {
                unsigned block = mask ^ tail;
                if (valid[block] && prev[tail] != inf && cost[block]+prev[tail] < best) {
                    best = cost[block]+prev[tail];
                    pick[(size_t)g*size+mask] = (uint16_t)block;
                }
                if (!tail) break;
            }
            next[mask] = best;
        }
        int64_t *swap = prev;
        prev = next;
        next = swap;
    }
    unsigned remaining = size-1;
    if (prev[remaining] == inf) goto done;
    *minimum = prev[remaining];
    for (int g = p->groups; remaining && g >= 1; g--) {
        unsigned block = pick[(size_t)g*size+remaining];
        if (!block) goto done;
        for (int j = 0; j < p->n; j++) if (block & (1u << j)) labels[j] = g;
        remaining ^= block;
    }
    solved = !remaining && partition_time_left(t);
done:
    free(valid);
    free(cost);
    free(pick);
    return solved;
}

static int partition_publish(PRIMALtask_t t, const Partition *p, const int *labels, int64_t minimum) {
    int n = p->n, nv = p->edges;
    double x[PART_EDGES], matrix[PART_VERTICES*PART_VERTICES];
    double tolerance = fmin(1e-8,fmin(t->mip_tol_feas,t->tol_co_pfeas));
    int64_t cost = 0;
    for (int j = 0; j < nv; j++) {
        x[j] = labels[p->first[j]] == labels[p->second[j]];
        double lo, up;
        bound_range(t->bkx[j],t->blx[j],t->bux[j],&lo,&up);
        if (!(x[j] >= lo-tolerance && x[j] <= up+tolerance)) return 0;
        cost += x[j] != 0.0 ? (int64_t)t->c[j] : 0;
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            matrix[i*n+j] = i == j ? 1.0 : labels[i] == labels[j] ? p->within : p->across;
    if (cost != minimum || !(bar_min_eig(n,matrix,NULL) >= -tolerance)) return 0;
    for (int r = 0; r < t->numcon; r++) {
        const double *a = p->a+(size_t)r*nv;
        int entry = p->bar_entry[r];
        double value = entry >= 0 ? p->row_sign[r]*matrix[entry] : 0.0;
        for (int j = 0; j < nv; j++) value += a[j]*x[j];
        if (!isfinite(value) || !(value >= p->lo[r]-tolerance && value <= p->up[r]+tolerance))
            return 0;
    }
    if (!partition_time_left(t) || !isfinite(t->cfix+(double)cost)) return 0;
    memcpy(t->x,x,(size_t)nv*sizeof(*x));
    memcpy(t->barx[0],matrix,(size_t)n*n*sizeof(*matrix));
    t->pobj = t->dobj = t->cfix+(double)cost;
    t->has_sol = 1;
    t->solsta = PRIMAL_SOL_STA_INTEGER_OPTIMAL;
    tlog(t,"MIP: partition optimality certified by subset dynamic programming\n");
    return 1;
}

int mip_partition_solve(PRIMALtask_t t) {
    if (t->numbarvar != 1 || t->nbarC || t->numcones || t->has_qobj || t->has_qcon ||
        t->numsos || t->has_warm || t->cbfn || t->progcb || t->numcon > PART_ROWS ||
        !partition_time_left(t)) return 0;
    int n = t->barDim[0], nv = t->numvar;
    if (n < 3 || n > PART_VERTICES || nv != n*(n-1)/2) return 0;
    Partition p = {0};
    p.n = n;
    p.edges = nv;
    p.a = calloc((size_t)t->numcon*nv,sizeof(*p.a));
    if (!p.a) return 0;
    int labels[PART_VERTICES];
    int64_t minimum;
    int solved = partition_read(t,&p) && partition_search(t,&p,labels,&minimum) &&
                 partition_publish(t,&p,labels,minimum);
    free(p.a);
    return solved;
}
