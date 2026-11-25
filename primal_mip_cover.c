/*
 * PrimalSolver - a convex optimization solver in C99.
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 */
#include "primal_priv.h"

enum {
    COVER_DEMANDS = 16, COVER_RESOURCES = 24, COVER_MAX_ROWS = 64,
    COVER_MAX_DEMAND = 8192, COVER_MAX_SUPPLY = 64, COVER_PRICE_ROUNDS = 28,
    COVER_DP_WORK_LIMIT = 2000000
};

typedef struct {
    int column, resource, units, upper;
    double cost;
} CoverItem;

typedef struct {
    int need, count;
    CoverItem item[COVER_RESOURCES];
} CoverDemand;

typedef struct {
    int demands, resources, max_need;
    int supply[COVER_RESOURCES];
    CoverDemand demand[COVER_DEMANDS];
    double price_cap;
} CoverModel;

typedef struct {
    double *value, *next;
    unsigned char *choice;
} CoverWorkspace;

static int cover_integer(double value, int max) {
    return value >= 0.0 && value <= max && floor(value) == value;
}

static int cover_gcd(int a, int b) {
    while (b) {
        int remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

/* A column covers one demand and consumes one unit of one resource.
 * Represent demand rows in half units, then compress to their exact lattice. */
static int cover_pattern(PRIMALtask_t t, const double *lx, const double *ux,
                          const double *lc, const double *uc, CoverModel *p) {
    int row_kind[COVER_MAX_ROWS], row_id[COVER_MAX_ROWS];
    for (int i = 0; i < t->numcon; i++) {
        row_kind[i] = 0;
        if (lc[i] == -INFINITY && uc[i] == INFINITY) continue;
        if (uc[i] == INFINITY && cover_integer(2.0 * lc[i], 2*COVER_MAX_DEMAND) &&
            p->demands < COVER_DEMANDS) {
            row_kind[i] = 1;
            int r = p->demands++;
            row_id[i] = r;
            p->demand[r].need = (int)(2.0 * lc[i]);
        } else if (lc[i] == -INFINITY && cover_integer(uc[i], COVER_MAX_SUPPLY) &&
                   p->resources < COVER_RESOURCES) {
            row_kind[i] = -1;
            row_id[i] = p->resources++;
            p->supply[row_id[i]] = (int)uc[i];
        } else return 0;
    }
    if (p->demands < 2 || p->resources < 2) return 0;
    for (int j = 0; j < t->numvar; j++) {
        const Col *c = &t->cols[j];
        int demand = -1, resource = -1, units = 0;
        if ((t->vartype[j] != PRIMAL_VAR_TYPE_INT && t->vartype[j] != PRIMAL_VAR_TYPE_INT_BIN) ||
            lx[j] != 0.0 || !(ux[j] >= 0.0) || !cover_integer(t->c[j], 1000000))
            return 0;
        for (int k = 0; k < c->nz; k++) {
            int r = c->sub[k];
            if (row_kind[r] == 1) {
                if (demand >= 0 || !cover_integer(2.0 * c->val[k], 2*COVER_MAX_DEMAND) ||
                    c->val[k] == 0.0) return 0;
                demand = row_id[r];
                units = (int)(2.0 * c->val[k]);
            } else if (row_kind[r] == -1) {
                if (resource >= 0 || c->val[k] != 1.0) return 0;
                resource = row_id[r];
            }
        }
        if (demand < 0 || resource < 0) return 0;
        CoverDemand *d = &p->demand[demand];
        for (int k = 0; k < d->count; k++)
            if (d->item[k].resource == resource) return 0;
        int upper = (int)fmin(floor(ux[j]), p->supply[resource]);
        if (t->vartype[j] == PRIMAL_VAR_TYPE_INT_BIN && upper > 1) upper = 1;
        d->item[d->count++] = (CoverItem){j, resource, units, upper, t->c[j]};
        p->price_cap = fmax(p->price_cap, t->c[j] + 1.0);
    }
    size_t work = 0;
    for (int r = 0; r < p->demands; r++) {
        CoverDemand *d = &p->demand[r];
        if (!d->need) continue;
        if (!d->count) return 0;
        int step = d->item[0].units;
        for (int k = 1; k < d->count; k++) step = cover_gcd(step, d->item[k].units);
        d->need = (d->need + step-1) / step;
        if (d->need > p->max_need) p->max_need = d->need;
        for (int k = 0; k < d->count; k++) {
            CoverItem *a = &d->item[k];
            a->units /= step;
            int choices = (d->need + a->units-1) / a->units;
            if (choices > a->upper) choices = a->upper;
            work += (size_t)(d->need+1) * (choices+1);
        }
    }
    if (!p->max_need || work > COVER_DP_WORK_LIMIT) return 0;
    return 1;
}

/* Suffix DP on the remaining demand. Amounts above the demand all
 * lead to the same state. The selected coordinates are exact integers. */
static double cover_plan(const CoverDemand *d, const int *available,
                         const double *price, CoverWorkspace *w, int *take) {
    int need = d->need, width = need + 1;
    if (!need) {
        memset(take, 0, (size_t)d->count * sizeof(*take));
        return 0.0;
    }
    double *next = w->next, *value = w->value;
    next[0] = 0.0;
    for (int i = 1; i <= need; i++) next[i] = INFINITY;
    for (int k = d->count-1; k >= 0; k--) {
        const CoverItem *a = &d->item[k];
        int supply = available[a->resource];
        if (supply > a->upper) supply = a->upper;
        double cost = a->cost + price[a->resource];
        unsigned char *choice = w->choice + (size_t)k * width;
        for (int i = 0; i <= need; i++) {
            double best = next[i];
            int units = 0, cap = (i + a->units-1) / a->units;
            if (cap > supply) cap = supply;
            for (int q = 1; q <= cap; q++) {
                int left = i - q*a->units;
                double v = q*cost + next[left > 0 ? left : 0];
                if (v < best) {
                    best = v;
                    units = q;
                }
            }
            value[i] = best;
            choice[i] = (unsigned char)units;
        }
        double *swap = next;
        next = value;
        value = swap;
    }
    double result = next[need];
    if (!isfinite(result)) return result;
    for (int k = 0; k < d->count; k++) {
        take[k] = w->choice[(size_t)k * width + need];
        need -= take[k] * d->item[k].units;
        if (need < 0) need = 0;
    }
    return result;
}

static int cover_time_left(PRIMALtask_t t) {
    return t->mip_deadline < 0.0 || (double)clock() < t->mip_deadline;
}

/* Cutting-plane master for nonnegative resource prices. Its output
 * ranks plans for the heuristic. It is never used as a node bound. */
static void cover_prices(PRIMALtask_t t, const CoverModel *p, CoverWorkspace *w,
                          double *best_price) {
    int m = p->demands, n = p->resources;
    PRIMALtask_t lp = NULL;
    if (PRIMAL_maketask(t->env, 0, 0, &lp) || PRIMAL_appendvars(lp, n+m)) goto done;
    lp->optimizer = PRIMAL_OPTIMIZER_PRIMAL_SIMPLEX;
    lp->log = 0;
    for (int j = 0; j < n; j++) {
        if (PRIMAL_putvarbound(lp, j, PRIMAL_BK_RA, 0.0, p->price_cap) ||
            PRIMAL_putcj(lp, j, p->supply[j])) goto done;
    }
    for (int r = 0; r < m; r++)
        if (PRIMAL_putvarbound(lp, n+r, PRIMAL_BK_FR, 0.0, 0.0) ||
            PRIMAL_putcj(lp, n+r, -1.0)) goto done;

    double price[COVER_RESOURCES] = {0}, theta[COVER_DEMANDS] = {0};
    double best_bound = -INFINITY;
    for (int round = 0; round < COVER_PRICE_ROUNDS && cover_time_left(t); round++) {
        double value = 0.0;
        int added = 0;
        for (int r = 0; r < m; r++) {
            if (!cover_time_left(t)) goto done;
            int take[COVER_RESOURCES];
            const CoverDemand *d = &p->demand[r];
            double v = cover_plan(d, p->supply, price, w, take);
            if (!isfinite(v)) goto done;
            value += v;
            if (round && theta[r] <= v + 1e-7) continue;
            int row = lp->numcon, nz = 0;
            int index[COVER_RESOURCES+1];
            double coef[COVER_RESOURCES+1], cost = 0.0;
            for (int k = 0; k < d->count; k++) {
                if (!take[k]) continue;
                index[nz] = d->item[k].resource;
                coef[nz++] = -take[k];
                cost += take[k] * d->item[k].cost;
            }
            index[nz] = n+r;
            coef[nz++] = 1.0;
            if (PRIMAL_appendcons(lp, 1) || PRIMAL_putconbound(lp, row, PRIMAL_BK_UP, 0.0, cost) ||
                PRIMAL_putarow(lp, row, nz, index, coef))
                goto done;
            added++;
        }
        for (int j = 0; j < n; j++) value -= price[j] * p->supply[j];
        if (value > best_bound) {
            best_bound = value;
            memcpy(best_price, price, (size_t)n * sizeof(*price));
        }
        if (!added || !cover_time_left(t)) break;
        /* Keep the outer solve's IPM deadline and objective cuts installed. */
        if (opt_routes(lp) != PRIMAL_RES_OK || !lp->has_sol ||
            lp->solsta != PRIMAL_SOL_STA_OPTIMAL) break;
        for (int j = 0; j < n; j++) price[j] = fmax(0.0, fmin(p->price_cap, lp->x[j]));
        memcpy(theta, lp->x + n, (size_t)m * sizeof(*theta));
    }
done:
    PRIMAL_deletetask(&lp);
}

void mip_cover_incumbent(PRIMALtask_t t, const double *lc, const double *uc,
                         double *lx, double *ux, double *candidate,
                         double *scratch, double *objective) {
    if (t->numcon > COVER_MAX_ROWS || t->numvar > COVER_DEMANDS * COVER_RESOURCES ||
        t->has_qobj || t->has_qcon || t->numcones || t->numbarvar || t->numsos ||
        t->cbfn || t->progcb || !cover_time_left(t)) return;
    for (int j = 0; j < t->numvar; j++)
        bound_range(t->bkx[j], t->blx[j], t->bux[j], &lx[j], &ux[j]);
    CoverModel p = {0};
    if (!cover_pattern(t, lx, ux, lc, uc, &p)) return;
    size_t width = (size_t)p.max_need + 1;
    double *dp = malloc(2 * width * sizeof(*dp));
    unsigned char *choice = malloc(COVER_RESOURCES * width * sizeof(*choice));
    if (!dp || !choice) { free(dp); free(choice); return; }
    CoverWorkspace w = {dp, dp + width, choice};
    double price[COVER_RESOURCES] = {0}, zero[COVER_RESOURCES] = {0};
    cover_prices(t, &p, &w, price);
    int order[COVER_DEMANDS];
    for (int r = 0; r < p.demands; r++) {
        int k = r;
        while (k > 0 && p.demand[order[k-1]].need < p.demand[r].need) {
            order[k] = order[k-1];
            k--;
        }
        order[k] = r;
    }
    double best = INFINITY;
    unsigned seed = 42;
    for (int trial = 0; trial < 14 && cover_time_left(t); trial++) {
        if (trial > 3)
            for (int r = p.demands-1; r > 0; r--) {
                seed = seed * 1664525u + 1013904223u;
                int k = (int)((seed >> 8) % (unsigned)(r+1));
                int saved = order[k];
                order[k] = order[r];
                order[r] = saved;
            }
        int remaining[COVER_RESOURCES];
        memcpy(remaining, p.supply, (size_t)p.resources * sizeof(*remaining));
        double cost = t->cfix;
        memset(scratch, 0, (size_t)t->numvar * sizeof(*scratch));
        int ok = 1;
        for (int i = 0; i < p.demands; i++) {
            if (!cover_time_left(t)) { ok = 0; break; }
            const CoverDemand *d = &p.demand[order[(trial % 2) ? p.demands-1-i : i]];
            int take[COVER_RESOURCES];
            if (!isfinite(cover_plan(d, remaining, trial == 2 || trial == 3 ? zero : price,
                                     &w, take))) { ok = 0; break; }
            for (int j = 0; j < d->count; j++) {
                const CoverItem *a = &d->item[j];
                remaining[a->resource] -= take[j];
                scratch[a->column] = take[j];
                cost += take[j] * a->cost;
            }
        }
        if (ok && cost < best &&
            mip_point_measures(t, lx, ux, lc, uc, scratch, t->mip_tol_feas, t->mip_tol_feas)) {
            best = cost;
            memcpy(candidate, scratch, (size_t)t->numvar * sizeof(*candidate));
        }
    }
    free(dp);
    free(choice);
    if (!isfinite(best)) return;
    *objective = best;
    tlog(t, "MIP: capacity-priced cover incumbent\n");
}
