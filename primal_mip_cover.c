/*
 * PrimalSolver - a convex optimization solver in C99.
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 */
#include "primal_priv.h"
#include <stdint.h>

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

/* Enumerate the paid choices separately from the free allocations. Rows
 * sharing a free-yield profile are pooled in the relaxation. For two
 * profiles, the fractional allocation is a one-constraint knapsack. */
typedef struct {
    CoverItem paid[COVER_RESOURCES];
    int count, demand, free_supply, kind, width;
    int64_t *cost;
} PaidDemand;

typedef struct {
    PRIMALtask_t task;
    PaidDemand row[COVER_DEMANDS];
    int n, visits, limit, free_order[COVER_RESOURCES], free_count;
    int yield[2][COVER_RESOURCES];
    int remaining[COVER_RESOURCES], deficit[2];
    int64_t suffix[COVER_DEMANDS+1], cutoff;
} CoverProof;

static const int64_t cover_no_plan = INT64_MAX/4;

static int cover_paid_dp(const CoverModel *p, PaidDemand *d) {
    int need = d->demand-d->free_supply;
    if (need < 0) need = 0;
    int width = need+1;
    d->width = width;
    d->cost = malloc((size_t)(d->count+1)*width*sizeof(*d->cost));
    if (!d->cost) return 0;
    int64_t *last = d->cost+(size_t)d->count*width;
    last[0] = 0;
    for (int i = 1; i <= need; i++) last[i] = cover_no_plan;
    for (int k = d->count-1; k >= 0; k--) {
        const CoverItem *a = &d->paid[k];
        int64_t *out = d->cost+(size_t)k*width;
        const int64_t *tail = out+width;
        for (int i = 0; i <= need; i++) {
            int cap = (i+a->units-1)/a->units;
            if (cap > a->upper) cap = a->upper;
            if (cap > p->supply[a->resource]) cap = p->supply[a->resource];
            int64_t value = tail[i];
            for (int q = 1; q <= cap; q++) {
                int left = i-q*a->units;
                int64_t v = (int64_t)a->cost*q+tail[left > 0 ? left : 0];
                if (v < value) value = v;
            }
            out[i] = value;
        }
    }
    return 1;
}

static int cover_paid_read(CoverProof *q, const CoverModel *p) {
    int kinds = 0;
    for (int i = 0; i < p->demands; i++) {
        const CoverDemand *d = &p->demand[i];
        if (!d->need) continue;
        int yield[COVER_RESOURCES] = {0}, free_supply = 0;
        PaidDemand *row = &q->row[q->n++];
        row->demand = d->need;
        for (int k = 0; k < d->count; k++) {
            CoverItem a = d->item[k];
            if (a.cost > 0.0) row->paid[row->count++] = a;
            else {
                yield[a.resource] = a.units;
                free_supply += a.units*a.upper;
            }
        }
        row->free_supply = free_supply;
        int kind = 0;
        while (kind < kinds && memcmp(yield,q->yield[kind],sizeof(yield))) kind++;
        if (kind >= 2) return 0;
        if (kind == kinds) {
            memcpy(q->yield[kinds],yield,sizeof(yield));
            kinds++;
        }
        row->kind = kind;
        if (!cover_paid_dp(p,row) || !cover_time_left(q->task)) return 0;
    }
    if (!kinds) return 0;
    if (kinds == 1) memcpy(q->yield[1],q->yield[0],sizeof(q->yield[0]));
    for (int r = 0; r < p->resources; r++) {
        if ((q->yield[0][r] > 0) != (q->yield[1][r] > 0)) return 0;
        if (!q->yield[0][r]) continue;
        int i = q->free_count++;
        for (; i > 0; i--) {
            int s = q->free_order[i-1];
            if ((int64_t)q->yield[0][r]*q->yield[1][s] <=
                (int64_t)q->yield[0][s]*q->yield[1][r]) break;
            q->free_order[i] = s;
        }
        q->free_order[i] = r;
    }
    for (int j = 1; j < q->n; j++) {
        PaidDemand row = q->row[j];
        int i = j;
        for (; i > 0 && row.cost[row.width-1] > q->row[i-1].cost[q->row[i-1].width-1]; i--)
            q->row[i] = q->row[i-1];
        q->row[i] = row;
    }
    for (int i = q->n-1; i >= 0; i--) {
        int64_t value = q->row[i].cost[q->row[i].width-1];
        q->suffix[i] = value > q->cutoff ? q->cutoff+1 : q->suffix[i+1]+value;
    }
    memcpy(q->remaining,p->supply,sizeof(q->remaining));
    return 1;
}

/* Allocate the residual stock fractionally between profiles, ignoring
 * individual free-variable caps. Rounding each fractional contribution
 * upward preserves an upper bound on the first profile's supply. */
static int cover_free_possible(const CoverProof *q) {
    int a = q->deficit[0], b = q->deficit[1], total = 0;
    for (int i = 0; i < q->free_count; i++) {
        int r = q->free_order[i];
        total += q->remaining[r]*q->yield[1][r];
    }
    if (b > total) return 0;
    int slack = total-b;
    for (int i = 0; i < q->free_count && a > 0 && slack > 0; i++) {
        int r = q->free_order[i], yield = q->yield[1][r];
        int units = q->remaining[r]*yield;
        if (units > slack) units = slack;
        a -= (int)(((int64_t)units*q->yield[0][r]+yield-1)/yield);
        slack -= units;
    }
    return a <= 0;
}

/* Return 1 only if every cheaper paid assignment is excluded. */
static int cover_paid_search(CoverProof *q, int row, int item, int need, int64_t cost) {
    if (++q->visits > q->limit || !cover_time_left(q->task)) return 0;
    PaidDemand *d = &q->row[row];
    int paid_need = need-d->free_supply;
    if (paid_need < 0) paid_need = 0;
    int64_t lower = d->cost[(size_t)item*d->width+paid_need];
    if (cost+lower+q->suffix[row+1] > q->cutoff) return 1;
    if (item == d->count || need == 0) {
        q->deficit[d->kind] += need;
        /* A feasible aggregate allocation is an open case unless the
         * search can still exclude it using the remaining paid choices. */
        int closed = !cover_free_possible(q);
        if (!closed && row+1 < q->n)
            closed = cover_paid_search(q,row+1,0,q->row[row+1].demand,cost);
        q->deficit[d->kind] -= need;
        return closed;
    }
    const CoverItem *a = &d->paid[item];
    int r = a->resource, cap = (need+a->units-1)/a->units, saved = q->remaining[r];
    if (cap > a->upper) cap = a->upper;
    if (cap > saved) cap = saved;
    for (int n = 0; n <= cap; n++) {
        int64_t spent = cost+n*(int64_t)a->cost;
        if (spent+q->suffix[row+1] > q->cutoff) break;
        int left = need-n*a->units;
        q->remaining[r] = saved-n;
        if (!cover_paid_search(q,row,item+1,left > 0 ? left : 0,spent)) {
            q->remaining[r] = saved;
            return 0;
        }
    }
    q->remaining[r] = saved;
    return 1;
}

static int cover_prove_optimum(PRIMALtask_t t, const CoverModel *p, const double *candidate) {
    if (!cover_time_left(t)) return 0;
    int stock[COVER_RESOURCES] = {0};
    int64_t cost = 0;
    /* Check the incumbent on the exact row lattice used in the proof. */
    for (int r = 0; r < p->demands; r++) {
        const CoverDemand *d = &p->demand[r];
        int amount = 0;
        for (int k = 0; k < d->count; k++) {
            const CoverItem *a = &d->item[k];
            double value = candidate[a->column];
            if (!cover_integer(value,a->upper)) return 0;
            int n = (int)value;
            stock[a->resource] += n;
            amount += n*a->units;
            cost += n*(int64_t)a->cost;
        }
        if (amount < d->need) return 0;
    }
    for (int r = 0; r < p->resources; r++) if (stock[r] > p->supply[r]) return 0;
    if (!cost) return 1;
    CoverProof q = {0};
    q.task = t;
    q.cutoff = cost-1;
    q.limit = t->mip_max_nodes < 100000 ? t->mip_max_nodes : 100000;
    int proven = cover_paid_read(&q,p) && (q.suffix[0] > q.cutoff ||
                  (q.n && cover_paid_search(&q,0,0,q.row[0].demand,0)));
    for (int i = 0; i < q.n; i++) free(q.row[i].cost);
    if (getenv("GMB_DBG"))
        fprintf(stderr,"  [cover-proof] nodes=%d cutoff=%lld proven=%d\n",
                q.visits,(long long)q.cutoff,proven);
    return proven && cover_time_left(t);
}

int mip_cover_incumbent(PRIMALtask_t t, const double *lc, const double *uc,
                         double *lx, double *ux, double *candidate,
                         double *scratch, double *objective) {
    if (t->numcon > COVER_MAX_ROWS || t->numvar > COVER_DEMANDS * COVER_RESOURCES ||
        t->has_qobj || t->has_qcon || t->numcones || t->numbarvar || t->numsos ||
        t->cbfn || t->progcb || !cover_time_left(t)) return 0;
    for (int j = 0; j < t->numvar; j++)
        bound_range(t->bkx[j], t->blx[j], t->bux[j], &lx[j], &ux[j]);
    CoverModel p = {0};
    if (!cover_pattern(t, lx, ux, lc, uc, &p)) return 0;
    size_t width = (size_t)p.max_need + 1;
    double *dp = malloc(2 * width * sizeof(*dp));
    unsigned char *choice = malloc(COVER_RESOURCES * width * sizeof(*choice));
    if (!dp || !choice) { free(dp); free(choice); return 0; }
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
    if (!isfinite(best)) return 0;
    *objective = best;
    tlog(t, "MIP: capacity-priced cover incumbent\n");
    if (!cover_prove_optimum(t,&p,candidate)) return 0;
    tlog(t, "MIP: shared-capacity cover optimality certified\n");
    return 1;
}
