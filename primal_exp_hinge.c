/*
 * PrimalSolver - a convex optimization solver in C99.
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 */
#include "primal_priv.h"

enum {
    HINGE_X = 32, HINGE_TERM = 1024, HINGE_EXTRA = 16,
    HINGE_VARS = 5*HINGE_TERM+HINGE_X+2, HINGE_ROWS = 4*HINGE_TERM+HINGE_EXTRA+1,
    HINGE_DIM = HINGE_X+1, HINGE_LINEAR = HINGE_X+HINGE_EXTRA
};

typedef struct {
    int count, col[3];
    double val[3];
} HingeRow;

typedef struct {
    int head_row, scale_row, arg_row, loss_row, head_alias, loss;
    double weight, rhs, a[HINGE_DIM];
} HingeTerm;

typedef struct {
    int n, dim, extra, sum_row, eta, level;
    int var[HINGE_X], row[HINGE_EXTRA], sign[HINGE_EXTRA];
    double sum_bound, eta_cost, penalty, rhs[HINGE_EXTRA];
    double linear[HINGE_EXTRA][HINGE_DIM];
    HingeTerm term[HINGE_TERM];
} HingeModel;

typedef struct {
    double objective, barrier, logsum;
    double p[HINGE_TERM], loss_slack[HINGE_TERM], linear_slack[HINGE_LINEAR];
} HingeEval;

static int hinge_slot(const HingeRow *row, int col, double val) {
    for (int i = 0; i < row->count && i < 3; i++)
        if (row->col[i] == col) return row->val[i] == val;
    return 0;
}

static int hinge_alias(PRIMALtask_t t, const HingeRow *r, int member) {
    if (t->cols[member].nz != 1 || t->cols[member].val[0] != -1.0) return -1;
    int i = t->cols[member].sub[0];
    if (r[i].count != 2 || t->bkc[i] != PRIMAL_BK_FX || t->blc[i] != 0.0) return -1;
    int k = r[i].col[0] == member;
    return r[i].val[k] == 1.0 ? r[i].col[k] : -1;
}

static int hinge_other(const Col *c, int row, double coefficient) {
    if (c->nz != 2) return -1;
    int k = c->sub[0] == row ? 0 : 1;
    if (c->sub[k] != row || c->val[k] != coefficient) return -1;
    return c->sub[1-k];
}

/* Recognize one fixed-scale exponential sum over private nonnegative
 * losses. A shared positive-cost variable lowers all losses. */
static int hinge_extract(PRIMALtask_t t, HingeModel *m) {
    int nv = t->numvar, nr = t->numcon, n = m->n;
    HingeRow *r = calloc((size_t)nr,sizeof(*r));
    if (!r) return 0;
    unsigned char col_role[HINGE_VARS] = {0};
    int row_role[HINGE_ROWS] = {0}, ok = 0;
    for (int i = 0; i < n; i++) {
        if (t->cone_type[i] != PRIMAL_CT_PEXP || t->cone_nmem[i] != 3) goto done;
        const int *mem = t->cone_mem[i];
        for (int k = 0; k < 3; k++) {
            int j = mem[k];
            if (col_role[j] || t->bkx[j] != PRIMAL_BK_FR || t->c[j] != 0.0) goto done;
            col_role[j] = 1;
        }
    }
    for (int j = 0; j < nv; j++) {
        const Col *c = &t->cols[j];
        if (t->vartype[j] != PRIMAL_VAR_TYPE_CONT || !isfinite(t->c[j])) goto done;
        for (int k = 0; k < c->nz; k++) {
            if (!isfinite(c->val[k]) || c->val[k] == 0.0) goto done;
            HingeRow *row = &r[c->sub[k]];
            int slot = row->count++;
            if (slot >= 3) continue;
            row->col[slot] = j;
            row->val[slot] = c->val[k];
        }
    }
    m->sum_row = m->level = m->eta = -1;
    for (int i = 0; i < n; i++) {
        HingeTerm *q = &m->term[i];
        const int *mem = t->cone_mem[i];
        int a = hinge_alias(t,r,mem[0]);
        if (a < 0 || col_role[a] || t->bkx[a] != PRIMAL_BK_FR || t->c[a] != 0.0) goto done;
        q->head_alias = a;
        q->head_row = t->cols[mem[0]].sub[0];
        int sum = hinge_other(&t->cols[a],q->head_row,1.0);
        if (sum < 0 || (m->sum_row >= 0 && sum != m->sum_row) ||
            t->bkc[sum] != PRIMAL_BK_UP || !(t->buc[sum] > 0.0) || !isfinite(t->buc[sum]))
            goto done;
        m->sum_row = sum; m->sum_bound = t->buc[sum];
        const Col *alias = &t->cols[a];
        q->weight = alias->val[alias->sub[0] == sum ? 0 : 1];
        if (!(q->weight > 0.0) || row_role[q->head_row]) goto done;
        row_role[q->head_row] = -1;
        col_role[a] = 1;
        const Col *sc = &t->cols[mem[1]], *arg = &t->cols[mem[2]];
        if (sc->nz != 1 || sc->val[0] != -1.0 || arg->nz != 1 || arg->val[0] != -1.0) goto done;
        q->scale_row = sc->sub[0]; q->arg_row = arg->sub[0];
        int u = q->scale_row, v = q->arg_row;
        if (u == v || row_role[u] || row_role[v] ||
            t->bkc[u] != PRIMAL_BK_FX || t->blc[u] != -1.0 || r[u].count != 1 ||
            t->bkc[v] != PRIMAL_BK_FX || t->blc[v] != 0.0 || r[v].count != 3) goto done;
        row_role[u] = row_role[v] = -1;
        int loss = -1, level = -1;
        for (int k = 0; k < 3; k++) {
            int j = r[v].col[k];
            if (j == mem[2]) continue;
            if (r[v].val[k] == 1.0) loss = j;
            else if (r[v].val[k] == -1.0) level = j;
            else goto done;
        }
        if (loss < 0 || level < 0 || col_role[loss] || t->c[loss] != 0.0 ||
            t->bkx[loss] != PRIMAL_BK_LO || t->blx[loss] != 0.0) goto done;
        if (m->level < 0) {
            if (col_role[level] || t->bkx[level] != PRIMAL_BK_FR ||
                !(t->c[level] > 0.0) || t->cols[level].nz != n) goto done;
            m->level = level; m->penalty = t->c[level]; col_role[level] = 1;
        } else if (m->level != level) goto done;
        int lr = hinge_other(&t->cols[loss],v,1.0);
        if (lr < 0 || row_role[lr] || t->bkc[lr] != PRIMAL_BK_LO || !isfinite(t->blc[lr]))
            goto done;
        q->loss = loss; q->loss_row = lr; q->rhs = t->blc[lr];
        col_role[loss] = 1; row_role[lr] = i+1;
    }
    if (row_role[m->sum_row] || r[m->sum_row].count != n) goto done;
    row_role[m->sum_row] = -1;
    for (int j = 0; j < nv; j++) if (!col_role[j] && t->bkx[j] == PRIMAL_BK_FR) {
        const Col *c = &t->cols[j];
        if (m->eta >= 0 || !(t->c[j] > 0.0) || c->nz != n) goto done;
        for (int k = 0; k < c->nz; k++)
            if (row_role[c->sub[k]] <= 0 || c->val[k] != 1.0) goto done;
        m->eta = j; m->eta_cost = t->c[j]; col_role[j] = 1;
    }
    if (m->eta < 0 || !(m->eta_cost < m->penalty)) goto done;
    for (int i = 0; i < nr; i++) if (!row_role[i]) {
        if (m->extra == HINGE_EXTRA || (t->bkc[i] != PRIMAL_BK_LO && t->bkc[i] != PRIMAL_BK_UP))
            goto done;
        int u = m->extra++, sign = t->bkc[i] == PRIMAL_BK_LO ? 1 : -1;
        m->row[u] = i; m->sign[u] = sign;
        m->rhs[u] = sign*(sign > 0 ? t->blc[i] : t->buc[i]);
        if (!isfinite(m->rhs[u])) goto done;
        row_role[i] = -2-u;
    }
    for (int j = 0; j < nv; j++) if (!col_role[j]) {
        if (m->dim == HINGE_X || t->bkx[j] != PRIMAL_BK_LO || t->blx[j] != 0.0 ||
            t->c[j] != 0.0) goto done;
        int u = m->dim++;
        m->var[u] = j;
        const Col *c = &t->cols[j];
        for (int k = 0; k < c->nz; k++) {
            int v = row_role[c->sub[k]];
            if (v > 0) m->term[v-1].a[u] += c->val[k];
            else if (v < -1) m->linear[-v-2][u] += m->sign[-v-2]*c->val[k];
            else goto done;
        }
    }
    if (!m->dim) goto done;
    for (int i = 0; i < n; i++) {
        if (!hinge_slot(&r[m->term[i].arg_row],m->level,-1.0)) goto done;
        m->term[i].a[m->dim] = 1.0;
    }
    ok = 1;
done:
    free(r);
    return ok;
}

/* A small phase I LP supplies a strictly feasible point in the common
 * coordinates. The private loss variables can then be set independently. */
static int hinge_start(PRIMALtask_t t, const HingeModel *m, double *x, double *z) {
    PRIMALtask_t lp = NULL;
    if (PRIMAL_maketask(t->env,0,0,&lp)) return 0;
    int d = m->dim, ok = 0;
    if (PRIMAL_appendvars(lp,d+1) || PRIMAL_appendcons(lp,d+m->extra) ||
        PRIMAL_putvarbound(lp,d,PRIMAL_BK_RA,0.0,1.0) || PRIMAL_putcj(lp,d,-1.0)) goto done;
    lp->log = 0;
    for (int j = 0; j < d; j++)
        if (PRIMAL_putvarbound(lp,j,PRIMAL_BK_LO,0.0,0.0) ||
            PRIMAL_putconbound(lp,j,PRIMAL_BK_LO,0.0,0.0) ||
            PRIMAL_putaij(lp,j,j,1.0)) goto done;
    for (int i = 0; i < m->extra; i++) {
        if (PRIMAL_putconbound(lp,d+i,PRIMAL_BK_LO,m->rhs[i],0.0)) goto done;
        for (int j = 0; j < d; j++)
            if (m->linear[i][j] != 0.0 && PRIMAL_putaij(lp,d+i,j,m->linear[i][j])) goto done;
    }
    for (int i = 0; i < lp->numcon; i++) if (PRIMAL_putaij(lp,i,d,-1.0)) goto done;
    if (PRIMAL_optimize(lp) || !lp->has_sol || lp->solsta != PRIMAL_SOL_STA_OPTIMAL ||
        !(lp->x[d] > 1e-10)) goto done;
    for (int j = 0; j < d; j++) x[j] = lp->x[j];
    x[d] = 0.0;
    for (int i = 0; i < m->n; i++)
        z[i] = fmax(0.0,m->term[i].rhs-dvec_dot(d+1,m->term[i].a,x))+0.01;
    ok = 1;
done:
    PRIMAL_deletetask(&lp);
    return ok;
}

static int hinge_eval(const HingeModel *m, const double *x, const double *z, HingeEval *v) {
    int d = m->dim;
    double peak = 0.0, sum = 0.0, barrier = 0.0;
    for (int j = 0; j < d; j++) {
        if (!(x[j] > 0.0)) return 0;
        v->linear_slack[j] = x[j];
        barrier += log(x[j]);
    }
    for (int i = 0; i < m->extra; i++) {
        double slack = dvec_dot(d,m->linear[i],x)-m->rhs[i];
        if (!(slack > 0.0)) return 0;
        v->linear_slack[d+i] = slack;
        barrier += log(slack);
    }
    for (int i = 0; i < m->n; i++) {
        const HingeTerm *t = &m->term[i];
        double slack = z[i]+(dvec_dot(d+1,t->a,x)-t->rhs);
        if (!(z[i] > 0.0 && slack > 0.0)) return 0;
        v->loss_slack[i] = slack;
        barrier += log(z[i])+log(slack);
        peak = fmax(peak,z[i]);
    }
    for (int i = 0; i < m->n; i++) {
        v->p[i] = m->term[i].weight*exp(z[i]-peak);
        sum += v->p[i];
    }
    if (!(sum > 0.0) || !isfinite(sum) || !isfinite(barrier)) return 0;
    for (int i = 0; i < m->n; i++) v->p[i] /= sum;
    v->logsum = (log(sum)-log(m->sum_bound))+peak;
    v->objective = m->eta_cost*x[d]+m->penalty*v->logsum;
    v->barrier = barrier;
    return isfinite(v->objective);
}

/* The loss Hessian is diagonal minus a rank-one covariance term.
 * Eliminate it with Sherman-Morrison, accumulating a small Schur system. */
static int hinge_step(const HingeModel *m, const HingeEval *v, const double *z,
                      double mu, double *dx, double *dz, double *decrease) {
    int d = m->dim, dim = d+1;
    double mat[HINGE_DIM][HINGE_DIM] = {{0}};
    double *rows[HINGE_DIM], g[HINGE_DIM] = {0}, a[HINGE_DIM] = {0}, b[HINGE_DIM] = {0};
    double gz[HINGE_TERM], diag[HINGE_TERM], u[HINGE_TERM], cross[HINGE_TERM];
    double denom = 0.0, ug = 0.0, root = sqrt(m->penalty);
    g[d] = m->eta_cost;
    for (int j = 0; j < d; j++) {
        double l = mu/v->linear_slack[j];
        g[j] = -l;
        mat[j][j] = l/v->linear_slack[j];
    }
    for (int i = 0; i < m->extra; i++) {
        const double *vrow = m->linear[i];
        double slack = v->linear_slack[d+i], q = mu/slack, weight = q/slack;
        for (int j = 0; j < d; j++) {
            g[j] -= q*vrow[j];
            double w = weight*vrow[j];
            for (int k = 0; k <= j; k++) mat[j][k] += w*vrow[k];
        }
    }
    for (int i = 0; i < m->n; i++) {
        const double *a_i = m->term[i].a;
        double slack = v->loss_slack[i], q = mu/slack, r = mu/z[i];
        double w = q/slack, k = r/z[i], p = v->p[i], tail = m->penalty*p+k;
        u[i] = root*p;
        cross[i] = w;
        gz[i] = m->penalty*p-r-q;
        diag[i] = 1.0/(tail+w);
        denom += p*(w+k)*diag[i];
        ug += u[i]*gz[i]*diag[i];
        double ratio = w*diag[i], schur = ratio*tail;
        for (int j = 0; j < dim; j++) {
            g[j] -= q*a_i[j];
            a[j] += ratio*u[i]*a_i[j];
            b[j] += ratio*gz[i]*a_i[j];
            double h = schur*a_i[j];
            for (int l = 0; l <= j; l++) mat[j][l] += h*a_i[l];
        }
    }
    if (!(denom > 0.0) || !isfinite(denom)) return 0;
    for (int j = 0; j < dim; j++) {
        rows[j] = mat[j];
        dx[j] = -g[j]+b[j]+a[j]*(ug/denom);
        for (int l = 0; l <= j; l++) mat[j][l] -= a[j]*a[l]/denom;
    }
    DMat factor = {dim,dim,rows};
    if (dmat_cholesky(&factor,dim) || dmat_chol_solve(&factor,dx,dim)) return 0;
    double uq = 0.0;
    for (int i = 0; i < m->n; i++) {
        dz[i] = gz[i]+cross[i]*dvec_dot(dim,m->term[i].a,dx);
        uq += u[i]*diag[i]*dz[i];
    }
    for (int i = 0; i < m->n; i++) dz[i] = -diag[i]*(dz[i]+u[i]*(uq/denom));
    *decrease = -(dvec_dot(dim,g,dx)+dvec_dot(m->n,gz,dz));
    return isfinite(*decrease) && *decrease >= 0.0;
}

static int hinge_publish(PRIMALtask_t t, const HingeModel *m, const HingeEval *ev,
                         const double *x, const double *z, double mu) {
    int nv = t->numvar, nr = t->numcon, d = m->dim;
    double *work = calloc((size_t)nv*2+(size_t)nr*2,sizeof(double));
    if (!work) return 0;
    double *point = work, *grad = point+nv, *y = grad+nv, *activity = y+nr;
    unsigned char cone_member[HINGE_VARS] = {0};
    double pf = fmin(1e-8,t->tol_co_pfeas), df = fmin(1e-8,t->tol_co_dfeas);
    int ok = 0;
    for (int j = 0; j < d; j++) point[m->var[j]] = x[j];
    point[m->eta] = x[d];
    point[m->level] = ev->logsum;
    y[m->sum_row] = m->penalty/m->sum_bound;
    for (int i = 0; i < m->extra; i++)
        y[m->row[i]] = -m->sign[i]*mu/ev->linear_slack[d+i];
    double price_sum = 0.0;
    for (int i = 0; i < m->n; i++) {
        const HingeTerm *q = &m->term[i];
        const int *mem = t->cone_mem[i];
        double tail = z[i]-ev->logsum, head = exp(tail);
        if (!(head > 0.0) || !isfinite(head)) goto done;
        point[q->loss] = z[i];
        point[q->head_alias] = point[mem[0]] = head;
        point[mem[1]] = 1.0;
        point[mem[2]] = tail;
        for (int k = 0; k < 3; k++) cone_member[mem[k]] = 1;
        double slope = m->penalty*ev->p[i];
        double scale = slope*(tail-1.0);
        y[q->head_row] = -y[m->sum_row]*q->weight;
        y[q->arg_row] = slope;
        y[q->scale_row] = -(scale+8.0*DBL_EPSILON*(fabs(scale)+slope));
        /* The ratio preserves the nonnegative loss's dual bound even
         * near a kink where both primal slacks are small. */
        y[q->loss_row] = -slope*(z[i]/(z[i]+ev->loss_slack[i]));
        price_sum -= y[q->loss_row];
    }
    /* Impose eta stationarity without crossing either dual bound. */
    if (!isfinite(price_sum)) goto done;
    if (price_sum > m->eta_cost) {
        double ratio = m->eta_cost/price_sum;
        for (int i = 0; i < m->n; i++) y[m->term[i].loss_row] *= ratio;
    } else {
        double ratio = (m->eta_cost-price_sum)/(m->penalty-price_sum);
        for (int i = 0; i < m->n; i++) {
            int r = m->term[i].loss_row;
            y[r] -= ratio*(m->penalty*ev->p[i]+y[r]);
        }
    }
    double primal = t->cfix, dual = t->cfix, worst = 0.0;
    for (int j = 0; j < nv; j++) {
        if (!isfinite(point[j])) goto done;
        primal += t->c[j]*point[j];
        double g = t->c[j];
        const Col *c = &t->cols[j];
        for (int k = 0; k < c->nz; k++) {
            int i = c->sub[k];
            activity[i] += c->val[k]*point[j];
            g += c->val[k]*y[i];
        }
        if (!isfinite(g)) goto done;
        grad[j] = g;
        if (cone_member[j]) continue;
        if (t->bkx[j] == PRIMAL_BK_FR) worst = fmax(worst,fabs(g));
        else {
            if (point[j] < -pf) goto done;
            worst = fmax(worst,-g);
        }
    }
    for (int r = 0; r < nr; r++) {
        if (!isfinite(activity[r]) || !isfinite(y[r])) goto done;
        double rhs = t->bkc[r] == PRIMAL_BK_UP ? t->buc[r] : t->blc[r];
        double v = activity[r]-rhs;
        if (t->bkc[r] == PRIMAL_BK_LO) {
            if (y[r] > 0.0) goto done;
            v = -v;
        } else if (t->bkc[r] == PRIMAL_BK_UP) {
            if (y[r] < 0.0) goto done;
        } else v = fabs(v);
        if (!(v <= pf)) goto done;
        dual -= rhs*y[r];
    }
    for (int i = 0; i < m->n; i++) {
        const int *mem = t->cone_mem[i];
        double u[3] = {point[mem[0]],point[mem[1]],point[mem[2]]};
        double v[3] = {grad[mem[0]],grad[mem[1]],grad[mem[2]]};
        if (!(cone_signed_slack(PRIMAL_CT_PEXP,0.0,u,3) >= -pf) ||
            !(cone_dual_signed_slack(PRIMAL_CT_PEXP,0.0,v,3) >= -df)) goto done;
    }
    double gap = fabs(primal-dual)/(1.0+fabs(primal)+fabs(dual));
    if (getenv("GMB_DBG"))
        fprintf(stderr,"  [exp-hinge] mu=%.3g primal=%.17g dual=%.17g gap=%.3g stationary=%.3g\n",
                mu,primal,dual,gap,worst);
    if (!isfinite(gap) || worst > df || gap > fmin(1e-8,t->tol_co_gap)) goto done;
    memcpy(t->x,point,(size_t)nv*sizeof(double));
    for (int j = 0; j < nv; j++)
        if (t->bkx[j] == PRIMAL_BK_LO) t->slx[j] = -fmax(0.0,grad[j]);
    for (int r = 0; r < nr; r++) {
        t->y[r] = y[r];
        t->slc[r] = fmin(0.0,y[r]);
        t->suc[r] = fmax(0.0,y[r]);
    }
    t->pobj = primal;
    t->dobj = dual;
    t->has_sol = 1;
    t->solsta = PRIMAL_SOL_STA_OPTIMAL;
    tlog(t,"optimal solution found (reduced exponential loss epigraphs)\n");
    ok = 1;
done:
    free(work);
    return ok;
}

static int hinge_solve(PRIMALtask_t t, const HingeModel *m, int limit) {
    int d = m->dim, n = m->n;
    double x[HINGE_DIM], z[HINGE_TERM], dx[HINGE_DIM], dz[HINGE_TERM];
    double tx[HINGE_DIM], tz[HINGE_TERM];
    HingeEval value, trial;
    if (!hinge_start(t,m,x,z) || !hinge_eval(m,x,z,&value)) return 0;
    double mu = 0.04*fmin(m->eta_cost,m->penalty/n);
    double target = 0.82*fmin(1e-8,t->tol_co_gap)/(2*n+d+m->extra);
    if (!(mu > 0.0 && target > 0.0) || !isfinite(mu)) return 0;
    int steps = 0;
    for (int stage = 0; stage < 18 && steps < limit; stage++) {
        for (int k = 0; k < 22 && steps < limit; k++, steps++) {
            double dec;
            if (!hinge_step(m,&value,z,mu,dx,dz,&dec) || dec <= fmax(2e-19,1e-8*mu))
                break;
            double alpha = 1.0;
            int accepted = 0;
            for (int ls = 0; ls < 30; ls++, alpha *= 0.5) {
                for (int j = 0; j <= d; j++) tx[j] = x[j]+alpha*dx[j];
                for (int i = 0; i < n; i++) tz[i] = z[i]+alpha*dz[i];
                if (!hinge_eval(m,tx,tz,&trial)) continue;
                double exp_change = 0.0;
                for (int i = 0; i < n; i++) exp_change += value.p[i]*expm1(alpha*dz[i]);
                double obj_change = trial.objective-value.objective;
                if (isfinite(exp_change) && exp_change > -1.0)
                    obj_change = m->eta_cost*alpha*dx[d]+m->penalty*log1p(exp_change);
                if (!(obj_change-mu*(trial.barrier-value.barrier) <= -0.01*alpha*dec))
                    continue;
                memcpy(x,tx,(size_t)(d+1)*sizeof(double));
                memcpy(z,tz,(size_t)n*sizeof(double));
                value = trial;
                accepted = 1;
                break;
            }
            if (!accepted) break;
        }
        if (mu <= target && hinge_publish(t,m,&value,x,z,mu)) return 1;
        mu = mu > target ? fmax(target,0.1*mu) : 0.75*mu;
    }
    return 0;
}

int exp_hinge_solve(PRIMALtask_t t) {
    int n = t->numcones, limit = iter_cap(t->max_iter_intpnt);
    if (n < 1 || n > HINGE_TERM || t->numvar > HINGE_VARS || t->numcon > HINGE_ROWS ||
        t->numvar < 5*n+3 || t->numcon < 4*n+1 || t->numbarvar ||
        t->has_warm || t->cbfn || t->progcb || limit <= 1 ||
        t->optimizer_max_time >= 0.0 || t->lower_obj_cut > -0.5*DBL_MAX ||
        t->upper_obj_cut < 0.5*DBL_MAX || getenv("GMB_NO_EXP_REDUCED") ||
        getenv("GMB_NO_EXP_ABSORB") || getenv("GMB_EXP_DA") || getenv("GMB_NO_EXP_HIGHER") ||
        getenv("GMB_SDP_DENSE") || getenv("GMB_SDP_SPARSE"))
        return 0;
    HingeModel *m = calloc(1,sizeof(*m));
    if (!m) return 0;
    m->n = n;
    int ok = hinge_extract(t,m) && hinge_solve(t,m,limit);
    free(m);
    return ok;
}
