/*
 * PrimalSolver - a convex optimization solver in C99.
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 */
#include "primal_priv.h"
#include <stdint.h>

enum {
    PLAN_ROWS = 256, PLAN_COLS = 320, PLAN_GROUPS = 24, PLAN_OPTIONS = 16,
    PLAN_STOCKS = 3, PLAN_BITS = 8, PLAN_STATES = 32768,
    PLAN_HASH_SIZE = 2 * PLAN_STATES
};

typedef struct {
    int count, col[PLAN_OPTIONS];
    unsigned local, next[PLAN_OPTIONS], after_next[PLAN_OPTIONS];
    int equivalent[PLAN_OPTIONS];
} PlanGroup;

typedef struct {
    int bits, col[PLAN_BITS], cap, reference, last, max_need;
    int rate, need[PLAN_GROUPS], amount[PLAN_COLS];
} PlanStock;

typedef struct {
    int rhs, used, stock, last;
    int *a;
} PlanRow;

typedef struct {
    int ncol, nrow, groups, stocks;
    int group_of[PLAN_COLS], stock_of[PLAN_COLS];
    PlanGroup group[PLAN_GROUPS];
    PlanStock stock[PLAN_STOCKS];
    PlanRow row[2 * PLAN_ROWS];
} PlanModel;

typedef struct {
    uint64_t key, choices_low, choices_high;
    int64_t cost;
} PlanState;

static int plan_time_left(PRIMALtask_t t) {
    return t->mip_deadline < 0.0 || (double)clock() < t->mip_deadline;
}

static int plan_integer(double x, double limit) {
    return fabs(x) <= limit && floor(x) == x;
}

static unsigned option_mask(int count) {
    return (1u << count) - 1u;
}

/* Normalize finite sides to integer inequalities. A genuine equality
 * then has the same representation as two opposite inequalities. */
static int plan_rows(PRIMALtask_t t, PlanModel *p, int *coeff,
                     const double *lc, const double *uc) {
    int lower[PLAN_ROWS], upper[PLAN_ROWS];
    for (int i = 0; i < t->numcon; i++) {
        lower[i] = upper[i] = -1;
        if (isfinite(lc[i])) {
            if (!plan_integer(lc[i], 32767)) return 0;
            lower[i] = p->nrow++;
            p->row[lower[i]].rhs = -(int)lc[i];
        }
        if (isfinite(uc[i])) {
            if (!plan_integer(uc[i], 32767)) return 0;
            upper[i] = p->nrow++;
            p->row[upper[i]].rhs = (int)uc[i];
        }
    }
    for (int i = 0; i < p->nrow; i++) {
        p->row[i].a = coeff + (size_t)i * p->ncol;
        p->row[i].stock = -1;
    }
    for (int j = 0; j < p->ncol; j++) {
        p->group_of[j] = p->stock_of[j] = -1;
        const Col *col = &t->cols[j];
        for (int k = 0; k < col->nz; k++) {
            int i = col->sub[k];
            if (lower[i] < 0 && upper[i] < 0) continue;
            if (!plan_integer(col->val[k], 32767)) return 0;
            int a = (int)col->val[k];
            int slot = upper[i] >= 0 ? upper[i] : lower[i];
            int sum = a + (upper[i] >= 0 ? p->row[slot].a[j] : -p->row[slot].a[j]);
            if (abs(sum) > 32767) return 0;
            if (lower[i] >= 0) p->row[lower[i]].a[j] = -sum;
            if (upper[i] >= 0) p->row[upper[i]].a[j] = sum;
        }
    }
    return 1;
}

/* Find disjoint exactly-one sets. Order them by their first column. */
static int plan_groups(PlanModel *p) {
    for (int i = 0; i < p->nrow; i++) {
        PlanRow *r = &p->row[i];
        if (r->rhs != 1 || r->used) continue;
        PlanGroup group = {0};
        int j = 0;
        for (; j < p->ncol; j++) {
            if (!r->a[j]) continue;
            if (r->a[j] != 1 || group.count == PLAN_OPTIONS) break;
            group.col[group.count++] = j;
        }
        if (j != p->ncol || group.count < 2) continue;
        for (int k = 0; k < p->nrow; k++) {
            PlanRow *opposite = &p->row[k];
            if (opposite->rhs != -1 || opposite->used) continue;
            int m = 0;
            while (m < p->ncol && opposite->a[m] == -r->a[m]) m++;
            if (m != p->ncol) continue;
            if (p->groups == PLAN_GROUPS) return 0;
            int pos = p->groups++;
            while (pos > 0 && p->group[pos-1].col[0] > group.col[0]) {
                p->group[pos] = p->group[pos-1];
                pos--;
            }
            p->group[pos] = group;
            r->used = opposite->used = 1;
            break;
        }
    }
    if (p->groups < 2) return 0;
    for (int g = 0; g < p->groups; g++) {
        PlanGroup *group = &p->group[g];
        group->local = option_mask(group->count);
        unsigned next = g+1 < p->groups ? option_mask(p->group[g+1].count) : 0;
        unsigned after = g+2 < p->groups ? option_mask(p->group[g+2].count) : 0;
        for (int o = 0; o < group->count; o++) {
            int col = group->col[o];
            if (p->group_of[col] >= 0) return 0;
            p->group_of[col] = g;
            group->next[o] = next;
            group->after_next[o] = after;
        }
    }
    return 1;
}

/* The leftover columns must encode bounded nonnegative reserves, with
 * consecutive binary weights and a constant price per unit. */
static int plan_stocks(PRIMALtask_t t, PlanModel *p) {
    for (int i = 0; i < p->nrow; i++) {
        PlanRow *r = &p->row[i];
        if (r->used || r->rhs < 0 || r->rhs > 255) continue;
        PlanStock stock = {0};
        unsigned seen = 0;
        int j = 0;
        for (; j < p->ncol; j++) {
            int a = r->a[j];
            if (!a) continue;
            if (p->group_of[j] >= 0 || p->stock_of[j] >= 0 ||
                a < 1 || a > 128 || (a & (a-1)) || (seen & (unsigned)a))
                break;
            int bit = 0;
            while ((1 << bit) != a) bit++;
            stock.col[bit] = j;
            stock.bits++;
            seen |= (unsigned)a;
        }
        if (j != p->ncol || !stock.bits || seen != option_mask(stock.bits) ||
            r->rhs > (int)seen) continue;
        stock.rate = (int)t->c[stock.col[0]];
        stock.cap = r->rhs;
        stock.reference = stock.last = -1;
        for (int b = 1; b < stock.bits; b++)
            if (t->c[stock.col[b]] != (double)stock.rate * (1 << b)) return 0;
        if (p->stocks == PLAN_STOCKS) return 0;
        for (int b = 0; b < stock.bits; b++) p->stock_of[stock.col[b]] = p->stocks;
        p->stock[p->stocks++] = stock;
        r->used = 1;
    }
    if (!p->stocks) return 0;
    for (int j = 0; j < p->ncol; j++)
        if (p->group_of[j] < 0 && p->stock_of[j] < 0) return 0;
    return 1;
}

/* Record a local inequality as forbidden choices or forbidden pairs. */
static void plan_local(PlanModel *p, const PlanRow *r, int first, int last) {
    PlanGroup *a = &p->group[first], *b = &p->group[last];
    for (int i = 0; i < a->count; i++) {
        int left = r->a[a->col[i]];
        if (first == last) {
            if (left > r->rhs) a->local &= ~(1u << i);
            continue;
        }
        unsigned *mask = last == first+1 ? &a->next[i] : &a->after_next[i];
        for (int j = 0; j < b->count; j++)
            if (left + r->a[b->col[j]] > r->rhs) *mask &= ~(1u << j);
    }
}

/* Each remaining row is local or a lower bound on a reserve plus
 * cumulative contributions from choices up to a given group. */
static int plan_links(PlanModel *p) {
    for (int i = 0; i < p->nrow; i++) {
        PlanRow *r = &p->row[i];
        if (r->used) continue;
        int first = -1, last = -1, distinct = 0, stock = -1;
        unsigned groups = 0;
        for (int j = 0; j < p->ncol; j++) {
            if (!r->a[j]) continue;
            int g = p->group_of[j], s = p->stock_of[j];
            if (g >= 0) groups |= 1u << g;
            else {
                if (stock >= 0 && stock != s) return 0;
                stock = s;
            }
        }
        for (int g = 0; g < p->groups; g++) {
            if (!(groups & (1u << g))) continue;
            if (first < 0) first = g;
            last = g;
            distinct++;
        }
        if (stock < 0) {
            if (first < 0) {
                if (r->rhs < 0) return 0;
            } else {
                if (distinct > 2 || last - first > 2) return 0;
                plan_local(p, r, first, last);
            }
            continue;
        }

        PlanStock *s = &p->stock[stock];
        if (last < 0 || r->rhs > 0 || r->rhs < -255) return 0;
        for (int b = 0; b < s->bits; b++)
            if (r->a[s->col[b]] != -(1 << b)) return 0;
        int need = -r->rhs;
        if (need > s->need[last]) s->need[last] = need;
        if (need > s->max_need) s->max_need = need;
        if (last > s->last) {
            s->last = last;
            s->reference = i;
        }
        r->stock = stock;
        r->last = last;
    }

    for (int s = 0; s < p->stocks; s++) {
        PlanStock *stock = &p->stock[s];
        if (stock->reference < 0) return 0;
        const int *reference = p->row[stock->reference].a;
        for (int j = 0; j < p->ncol; j++) {
            if (p->group_of[j] < 0) continue;
            if (reference[j] > 0) return 0;
            stock->amount[j] = -reference[j];
        }
    }
    for (int i = 0; i < p->nrow; i++) {
        const PlanRow *r = &p->row[i];
        if (r->stock < 0) continue;
        const PlanStock *s = &p->stock[r->stock];
        for (int j = 0; j < p->ncol; j++) {
            int g = p->group_of[j];
            if (g >= 0 && g <= r->last && r->a[j] != -s->amount[j]) return 0;
        }
    }
    for (int g = 0; g < p->groups; g++) {
        PlanGroup *group = &p->group[g];
        for (int o = 0; o < group->count; o++) {
            group->equivalent[o] = o;
            for (int i = 0; i < o; i++) {
                if (group->next[i] == group->next[o] &&
                    group->after_next[i] == group->after_next[o]) {
                    group->equivalent[o] = i;
                    break;
                }
            }
        }
    }
    return 1;
}

/* The state stores the last two compatibility classes (4 bits each)
 * and, for each reserve, cumulative supply and peak deficit (8 bits each).
 * Above max_need, any two cumulative supplies satisfy every future prefix. */
static uint64_t plan_advance(const PlanModel *p, int g, int option, uint64_t key) {
    int col = p->group[g].col[option];
    uint64_t advanced = (uint64_t)p->group[g].equivalent[option] | ((key & 15) << 4);
    for (int r = 0; r < p->stocks; r++) {
        const PlanStock *s = &p->stock[r];
        int shift = 8 + 16*r;
        int count = (int)((key >> shift) & 255) + s->amount[col];
        if (count > s->max_need) count = s->max_need;
        int reserve = (int)((key >> (shift+8)) & 255), deficit = s->need[g] - count;
        if (reserve < deficit) reserve = deficit;
        if (reserve > s->cap) return UINT64_MAX;
        advanced |= (uint64_t)count << shift;
        advanced |= (uint64_t)reserve << (shift+8);
    }
    return advanced;
}

static int plan_insert(PlanState *states, int *slots, int *count,
                       const PlanState *source, uint64_t key, int g, int option, int cost) {
    uint64_t mix = (key ^ (key >> 27)) * UINT64_C(0x9e3779b97f4a7c15);
    unsigned slot = (unsigned)(mix >> 48);
    while (slots[slot] && states[slots[slot]-1].key != key)
        slot = (slot+1) & (PLAN_HASH_SIZE-1);
    int index = slots[slot]-1;
    int64_t new_cost = source->cost + cost;
    if (index >= 0 && states[index].cost <= new_cost) return 1;
    if (index < 0) {
        if (*count == PLAN_STATES) return 0;
        index = (*count)++;
        slots[slot] = index+1;
    }
    PlanState *dest = &states[index];
    *dest = *source;
    dest->key = key;
    dest->cost = new_cost;
    if (g < 16) dest->choices_low |= (uint64_t)option << (4*g);
    else dest->choices_high |= (uint64_t)option << (4*(g-16));
    return 1;
}

static int plan_search(PRIMALtask_t t, const PlanModel *p, double *x, double *obj) {
    PlanState *work = malloc((size_t)2 * PLAN_STATES * sizeof(*work));
    int *slots = malloc((size_t)PLAN_HASH_SIZE * sizeof(*slots));
    int solved = 0;
    if (!work || !slots) goto done;
    PlanState *front = work, *next = work + PLAN_STATES;
    front[0] = (PlanState){0};
    int count = 1;
    for (int g = 0; g < p->groups; g++) {
        int new_count = 0;
        memset(slots, 0, (size_t)PLAN_HASH_SIZE * sizeof(*slots));
        for (int i = 0; i < count; i++) {
            if ((i & 255) == 0 && !plan_time_left(t)) goto done;
            const PlanState *source = &front[i];
            unsigned allowed = p->group[g].local;
            if (g > 0) allowed &= p->group[g-1].next[source->key & 15];
            if (g > 1) allowed &= p->group[g-2].after_next[(source->key >> 4) & 15];
            for (int o = 0; o < p->group[g].count; o++) {
                if (!(allowed & (1u << o))) continue;
                uint64_t key = plan_advance(p, g, o, source->key);
                if (key == UINT64_MAX) continue;
                int cost = (int)t->c[p->group[g].col[o]];
                if (!plan_insert(next, slots, &new_count, source, key, g, o, cost)) goto done;
            }
        }
        PlanState *swap = front;
        front = next;
        next = swap;
        count = new_count;
        if (!count) goto done;
    }

    int selected = -1;
    int64_t best = INT64_MAX;
    for (int i = 0; i < count; i++) {
        int64_t cost = front[i].cost;
        for (int s = 0; s < p->stocks; s++) {
            int reserve = (int)((front[i].key >> (16+16*s)) & 255);
            cost += (int64_t)p->stock[s].rate * reserve;
        }
        if (cost < best) {
            best = cost;
            selected = i;
        }
    }
    const PlanState *winner = &front[selected];
    memset(x, 0, (size_t)p->ncol * sizeof(*x));
    for (int g = 0; g < p->groups; g++) {
        uint64_t choices = g < 16 ? winner->choices_low : winner->choices_high;
        int option = (int)((choices >> (4*(g%16))) & 15);
        x[p->group[g].col[option]] = 1.0;
    }
    for (int s = 0; s < p->stocks; s++) {
        int reserve = (int)((winner->key >> (16+16*s)) & 255);
        for (int b = 0; b < p->stock[s].bits; b++)
            x[p->stock[s].col[b]] = (double)((reserve >> b) & 1);
    }
    *obj = t->cfix + (double)best;
    solved = plan_time_left(t);
done:
    free(work);
    free(slots);
    return solved;
}

/* Exact bounded search for small binary choice sequences with prefix
 * requirements. Unknown structure, an exhausted budget or a rejected
 * witness all leave the solve to the regular MIP path. */
int mip_plan_solve(PRIMALtask_t t) {
    if (t->numvar < 5 || t->numvar > PLAN_COLS || t->numcon > PLAN_ROWS ||
        t->has_qobj || t->has_qcon || t->numcones || t->numbarvar || t->numsos ||
        t->cbfn || t->progcb || t->has_warm || !plan_time_left(t)) return 0;

    double lx[PLAN_COLS], ux[PLAN_COLS], lc[PLAN_ROWS], uc[PLAN_ROWS];
    for (int j = 0; j < t->numvar; j++) {
        bound_range(t->bkx[j], t->blx[j], t->bux[j], &lx[j], &ux[j]);
        if ((t->vartype[j] != PRIMAL_VAR_TYPE_INT && t->vartype[j] != PRIMAL_VAR_TYPE_INT_BIN) ||
            lx[j] != 0.0 || ux[j] != 1.0 || t->c[j] < 0.0 ||
            !plan_integer(t->c[j], 1000000)) return 0;
    }
    for (int i = 0; i < t->numcon; i++) {
        bound_range(t->bkc[i], t->blc[i], t->buc[i], &lc[i], &uc[i]);
        if (isnan(lc[i]) || isnan(uc[i]) || lc[i] > uc[i]) return 0;
    }

    PlanModel p = {0};
    p.ncol = t->numvar;
    int *coeff = calloc((size_t)2 * t->numcon * p.ncol, sizeof(*coeff));
    if (!coeff) return 0;
    int recognized = plan_rows(t, &p, coeff, lc, uc) && plan_groups(&p) &&
                     plan_stocks(t, &p) && plan_links(&p);
    free(coeff);
    if (!recognized) return 0;
    double x[PLAN_COLS], obj;
    if (!plan_search(t, &p, x, &obj) || !isfinite(obj) ||
        !mip_point_measures(t, lx, ux, lc, uc, x, t->mip_tol_feas, t->mip_tol_feas))
        return 0;

    double measured = 0.0;
    for (int j = 0; j < p.ncol; j++) measured += t->c[j] * x[j];
    if (t->cfix + measured != obj || !plan_time_left(t)) return 0;
    memcpy(t->x, x, (size_t)p.ncol * sizeof(*x));
    t->pobj = t->dobj = obj;
    t->has_sol = 1;
    t->solsta = PRIMAL_SOL_STA_INTEGER_OPTIMAL;
    tlog(t, "MIP: optimal choice sequence with prefix reserves\n");
    return 1;
}
