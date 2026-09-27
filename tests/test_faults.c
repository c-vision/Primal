/* Sweep each allocation site reached by representative sparse factorizations,
 * standard-form construction and AFE/SOS mutations. SPDX-License-Identifier: Apache-2.0 */
#define _POSIX_C_SOURCE 200809L
#define PRIMAL_FAULT_IMPLEMENTATION
#include "fault_hooks.h"
#include "primal_priv.h"
#ifdef _WIN32
#define setenv(n, v, o) _putenv_s((n), (v))
#endif
#include "linalg.h"
#include "stdform.h"
#include <math.h>
#include <stdio.h>
static int checks, failures, injected;
#define CHECK(c) do { checks++; if (!(c)) { failures++; fprintf(stderr,"%s:%d: %s (allocation %d)\n",__FILE__,__LINE__,#c,primal_fault_nth); } } while (0)
#define OK(c) CHECK((c) == PRIMAL_RES_OK)
static void arm(int n) { primal_fault_calls = primal_fault_hit = 0; primal_fault_nth = n; }
static int disarm(void) { int hit = primal_fault_hit; injected += hit; primal_fault_nth = 0; return hit; }
static void sparse_case(int kind) {
    enum { N = 8 };
    int ptr[N+1], sub[N*N], nz = 0; double val[N*N];
    for (int j = 0; j < N; j++) {
        ptr[j] = nz;
        for (int i = (kind == 2 ? 0 : j); i < N; i++) {
            sub[nz] = i; val[nz++] = i == j ? 10 : 1;
        }
    }
    ptr[N] = nz;
    for (int nth = 1; nth < 1000; nth++) {
        long before = primal_fault_live;
        double rhs[N]; for (int i = 0; i < N; i++) rhs[i] = 17;
        arm(nth);
        SpChol *c = NULL; SpluFact *u = NULL; SpLdl *d = NULL;
        if (kind < 2) c = kind ? spchol_factor_ord(N,ptr,sub,val) : spchol_factor(N,ptr,sub,val);
        if (kind == 2) u = splu_factor(N,ptr,sub,val);
        if (kind == 3) d = spldl_factor(N,ptr,sub,val);
        int hit = disarm();
        if (c) { CHECK(spchol_solve_ord(c,rhs) == 0); spchol_free(c); }
        if (u) { CHECK(splu_solve(u,rhs) == 0); splu_free(u); }
        if (d) { CHECK(spldl_solve(d,rhs) == 0); spldl_free(d); }
        if (c || u || d) for (int i = 0; i < N; i++) CHECK(fabs(rhs[i]-1) < 1e-12);
        if (!hit) CHECK(c || u || d);
        CHECK(primal_fault_live == before);
        if (!hit) break;
        CHECK(nth < 999);
    }
}
static void standard_case(void) {
    enum { N = 96, M = 8 };
    int ptr[N+1], sub[N*M], qi[N]; double val[N*M], qv[N], c[N], lo[N], up[N], lc[M], uc[M];
    for (int j = 0; j < N; j++) {
        ptr[j] = j*M; qi[j] = j; qv[j] = 1; c[j] = j+1; lo[j] = 0; up[j] = 10;
        for (int i = 0; i < M; i++) { sub[j*M+i] = i; val[j*M+i] = 1+i; }
    }
    ptr[N] = N*M;
    for (int i = 0; i < M; i++) { lc[i] = 1; uc[i] = 100; }
    for (int nth = 1; nth < 1000; nth++) {
        long before = primal_fault_live; arm(nth);
        StdForm *sf = stdform_build(N,M,c,qi,qi,qv,N,lo,up,lc,uc,ptr,sub,val);
        int hit = disarm();
        if (!hit) CHECK(sf != NULL);
        stdform_free(sf); CHECK(primal_fault_live == before);
        if (!hit) break;
        CHECK(nth < 999);
    }
}
static void mutation_case(PRIMALenv_t env, int kind) {
    for (int nth = 1; nth < 100; nth++) {
        long before = primal_fault_live;
        PRIMALtask_t t = NULL; OK(PRIMAL_maketask(env,0,8,&t));
        OK(PRIMAL_appendafes(t,1)); OK(PRIMAL_putafeg(t,0,17));
        OK(PRIMAL_putafefentry(t,0,0,3));
        int idx[8]; double v[8]; for (int i = 0; i < 8; i++) { idx[i] = i; v[i] = i+1; }
        OK(PRIMAL_appendcons(t,1));
        OK(PRIMAL_putqconk(t,0,1,(int[]){7},(int[]){7},(double[]){2}));
        PRIMALint64t dom;
        if (kind == 5) { OK(PRIMAL_appendrzerodomain(t,1,&dom)); OK(PRIMAL_appenddjcs(t,1)); }
        arm(nth);
        int rc = kind == 0 ? PRIMAL_appendafes(t,100) : kind == 1 ? PRIMAL_putafefrow(t,0,8,idx,v) :
            kind == 2 ? PRIMAL_appendsos2(t,8,idx,v) : kind == 3 ? PRIMAL_appendvars(t,8) :
            kind == 4 ? PRIMAL_appendcons(t,9) : PRIMAL_putdjc(t,0,2,(PRIMALint64t[]){dom,dom},
                2,(PRIMALint64t[]){0,0},(double[]){0,1},2,(PRIMALint64t[]){1,1});
        int hit = disarm(); CHECK(rc == PRIMAL_RES_OK || (hit && rc == PRIMAL_RES_ERR_ALLOC));
        double g; OK(PRIMAL_getafeg(t,0,&g)); CHECK(g == 17);
        if (hit && kind == 1) {
            int n, ids[8]; double vv[8]; OK(PRIMAL_getafefrow(t,0,&n,ids,vv));
            CHECK(n == 1 && ids[0] == 0 && vv[0] == 3);
        }
        int nv, nc; OK(PRIMAL_getnumvar(t,&nv)); OK(PRIMAL_getnumcon(t,&nc));
        if (hit) CHECK(nv == 8 && nc == 1);
        double q; OK(PRIMAL_getqconkij(t,0,7,7,&q)); CHECK(q == 2);
        OK(PRIMAL_deletetask(&t)); CHECK(primal_fault_live == before);
        if (!hit) break;
        CHECK(nth < 99);
    }
}
static void thread_case(PRIMALenv_t env) {
    /* Two binary packing rows have a fractional root. Enumerate all
     * assignments to supply an independent optimum. Disable optional cuts so
     * probing and strong branching both reach pthread_create. */
    setenv("GMB_NO_MIP_CUTS","1",1);
    setenv("GMB_NO_MIP_RINS","1",1); setenv("GMB_NO_MIP_RENS","1",1);
    setenv("GMB_NO_MIP_FPUMP","1",1); setenv("GMB_NO_MIP_LOCALSEARCH","1",1);
    enum { N = 4, M = 2 };
    const double A[M][N] = {{1,1,0,0},{0,0,1,1}};
    const double b[M] = {1.5,1.5}, c[N] = {1,1,1,1};
    double best = -INFINITY;
    for (int mask = 0; mask < (1<<N); mask++) {
        double obj = 0; int feasible = 1;
        for (int j = 0; j < N; j++) if ((mask>>j)&1) obj += c[j];
        for (int i = 0; i < M; i++) { double v = 0; for (int j = 0; j < N; j++) if ((mask>>j)&1) v += A[i][j]; if (v > b[i]) feasible = 0; }
        if (feasible) best = fmax(best,obj);
    }
    PRIMALtask_t t = NULL; OK(PRIMAL_maketask(env,M,N,&t));
    for (int j = 0; j < N; j++) { OK(PRIMAL_putvarbound(t,j,PRIMAL_BK_RA,0,1)); OK(PRIMAL_putvartype(t,j,PRIMAL_VAR_TYPE_INT_BIN)); OK(PRIMAL_putcj(t,j,c[j])); }
    int idx[N]; for (int j = 0; j < N; j++) idx[j] = j;
    for (int i = 0; i < M; i++) { OK(PRIMAL_putarow(t,i,N,idx,A[i])); OK(PRIMAL_putconbound(t,i,PRIMAL_BK_UP,0,b[i])); }
    OK(PRIMAL_putobjsense(t,PRIMAL_OPTIMIZE_MAXIMIZE)); OK(PRIMAL_putintparam(t,PRIMAL_IPAR_NUM_THREADS,4));
    primal_fault_threads = 1; int rc = PRIMAL_optimize(t);
    if (rc) fprintf(stderr,"thread fallback solve returned %d\n",rc);
    CHECK(rc == PRIMAL_RES_OK); primal_fault_threads = 0;
    double obj = NAN, x[N] = {0}; OK(PRIMAL_getprimalobj(t,PRIMAL_SOL_ITR,&obj)); OK(PRIMAL_getxx(t,PRIMAL_SOL_ITR,x));
    CHECK(fabs(obj-best) < 1e-7);
    for (int i = 0; i < M; i++) { double v = 0; for (int j = 0; j < N; j++) v += A[i][j]*x[j]; CHECK(v <= b[i]+1e-7); }
    for (int j = 0; j < N; j++) CHECK(fabs(x[j]-round(x[j])) < 1e-7);
    CHECK(primal_fault_probe > 0);
    double lo[N] = {0}, up[N] = {1,1,1,1}, lc[M] = {-INFINITY,-INFINITY};
    double fractional[N] = {0.75,0.75,0.75,0.75}, score[2] = {NAN,NAN};
    int cand[2] = {0,2}; SBJob jobs[2];
    for (int k = 0; k < 2; k++) {
        jobs[k].trelax = t; jobs[k].s = -1; jobs[k].nvar = N;
        jobs[k].lx = lo; jobs[k].ux = up; jobs[k].lc = lc; jobs[k].uc = b;
        jobs[k].x = fractional; jobs[k].cand = cand; jobs[k].start = k; jobs[k].end = k+1; jobs[k].score = score;
    }
    primal_fault_threads = 1; mip_run_jobs(2,sizeof(SBJob),jobs,sb_worker); primal_fault_threads = 0;
    CHECK(primal_fault_sb == 2);
    for (int k = 0; k < 2; k++) CHECK(fabs(score[k]+3) < 1e-7);
    OK(PRIMAL_deletetask(&t));
}
int main(void) {
    for (int kind = 0; kind < 4; kind++) sparse_case(kind);
    standard_case();
    PRIMALenv_t env = NULL; OK(PRIMAL_makeenv(&env,NULL));
    for (int kind = 0; kind < 6; kind++) mutation_case(env,kind);
    thread_case(env); OK(PRIMAL_deleteenv(&env));
    printf("Fault checks: %d, failures: %d; injected allocations: %d; failed probe/strong-branch threads: %d/%d\n",checks,failures,injected,primal_fault_probe,primal_fault_sb);
    return failures ? 1 : 0;
}
