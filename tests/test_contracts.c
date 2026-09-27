/* Regression checks for numerical and result contracts.
 * Expectations come from the original models, not solver residual getters.
 * SPDX-License-Identifier: Apache-2.0
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "primal.h"
#ifdef _WIN32
#define setenv(n, v, o) _putenv_s((n), (v))
#define unsetenv(n) _putenv_s((n), "")
#endif

static int failures, checks;
#define CHECK(c) do { checks++; if (!(c)) { failures++; \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)
#define OK(c) CHECK((c) == PRIMAL_RES_OK)
static PRIMALenv_t env;

static PRIMALtask_t model(int n, int m) {
    PRIMALtask_t t = NULL;
    OK(PRIMAL_maketask(env, 0, 0, &t));
    OK(PRIMAL_appendvars(t, n)); OK(PRIMAL_appendcons(t, m));
    return t;
}
static void exp_faces(int ct) {
    const int mem[] = {0,1,2}, sub[] = {0};
    for (int sign = -1; sign <= 1; sign++) {

        PRIMALtask_t t = model(3, 0);
        double x[3] = {1, 0, sign};
        if (ct == PRIMAL_CT_DEXP) { x[1] = sign; x[2] = 0; }
        for (int j = 0; j < 3; j++) OK(PRIMAL_putvarbound(t,j,PRIMAL_BK_FX,x[j],x[j]));
        OK(PRIMAL_optimize(t));
        OK(PRIMAL_appendcone(t, ct, 0, 3, mem));
        double v = NAN;
        OK(PRIMAL_getpviolcones(t, PRIMAL_SOL_ITR, 1, sub, &v));
        int valid = ct == PRIMAL_CT_PEXP ? sign <= 0 : sign >= 0;
        CHECK(valid ? v == 0 : v > 0);
        PRIMALrescodee rc = PRIMAL_optimize(t);
        PRIMALsolstae ss; OK(PRIMAL_getsolsta(t, PRIMAL_SOL_ITR, &ss));
        if (!valid) CHECK(rc != PRIMAL_RES_OK && ss != PRIMAL_SOL_STA_OPTIMAL);
        else CHECK(rc == PRIMAL_RES_OK);
        OK(PRIMAL_deletetask(&t));
    }
}
static void dexp_optimum(int cuts, int acc) {

    PRIMALtask_t t = model(3, 0);
    OK(PRIMAL_putcj(t,0,1));
    OK(PRIMAL_putvarbound(t,0,PRIMAL_BK_FR,0,0));
    OK(PRIMAL_putvarbound(t,1,PRIMAL_BK_FX,0,0));
    OK(PRIMAL_putvarbound(t,2,PRIMAL_BK_FX,-1,-1));
    if (acc) {
        PRIMALint64t domain;
        OK(PRIMAL_appendafes(t,3));
        for (int j=0;j<3;j++) OK(PRIMAL_putafefentry(t,j,j,1));
        OK(PRIMAL_appenddualexpconedomain(t,&domain));
        OK(PRIMAL_appendacc(t,domain,3,(PRIMALint64t[]){0,1,2},NULL));
    } else OK(PRIMAL_appendcone(t,PRIMAL_CT_DEXP,0,3,(int[]){0,1,2}));
    if (cuts) setenv("GMB_NO_EXP_IPM","1",1);
    OK(PRIMAL_optimize(t));
    if (cuts) unsetenv("GMB_NO_EXP_IPM");
    double x[3], p, d;
    OK(PRIMAL_getxxslice(t,PRIMAL_SOL_ITR,0,3,x));
    OK(PRIMAL_getprimalobj(t,PRIMAL_SOL_ITR,&p));
    OK(PRIMAL_getdualobj(t,PRIMAL_SOL_ITR,&d));
    CHECK(fabs(x[0]-exp(-1)) < 1e-6);
    CHECK(x[0] >= -x[2]*exp(x[1]/x[2]-1)-1e-6);
    CHECK(fabs(p-d) < 1e-6);
    OK(PRIMAL_deletetask(&t));
}
static void quadratic_rows(void) {
    const int key[] = {PRIMAL_BK_UP, PRIMAL_BK_LO, PRIMAL_BK_FX, PRIMAL_BK_RA};
    for (int k = 0; k < 4; k++) {

        PRIMALtask_t t = model(1,1);
        double q = k == 0 ? 2 : -2;
        OK(PRIMAL_putvarbound(t,0,PRIMAL_BK_RA,0,3));
        OK(PRIMAL_putcj(t,0,k < 2 ? -1 : 1));
        OK(PRIMAL_putconbound(t,0,key[k],k == 3 ? -4 : -1,k == 0 ? 1 : -1));
        OK(PRIMAL_putqconk(t,0,1,(int[]){0},(int[]){0},&q));
        PRIMALrescodee rc = PRIMAL_optimize(t);
        if (k >= 2) CHECK(rc == PRIMAL_RES_ERR_ARG);
        else {
            CHECK(rc == PRIMAL_RES_OK);
            double x, y, lo, up, d;
            OK(PRIMAL_getxx(t,PRIMAL_SOL_ITR,&x));
            OK(PRIMAL_gety(t,PRIMAL_SOL_ITR,&y));
            OK(PRIMAL_getslx(t,PRIMAL_SOL_ITR,&lo));
            OK(PRIMAL_getsux(t,PRIMAL_SOL_ITR,&up));
            OK(PRIMAL_getdualobj(t,PRIMAL_SOL_ITR,&d));
            CHECK(fabs(x-1) < 1e-6);
            CHECK(x*x <= 1+1e-6);
            CHECK(fabs(-1+y*q*x+lo+up) < 1e-6);
            CHECK(d <= -1+1e-6 && d >= -1-1e-6);
        }
        OK(PRIMAL_deletetask(&t));
    }
}
static void mip_repair(int max, int threads, int warm, int cuts) {

    PRIMALtask_t t = model(2,2);
    OK(PRIMAL_putvarbound(t,0,PRIMAL_BK_RA,0,1));
    OK(PRIMAL_putvarbound(t,1,PRIMAL_BK_LO,0,0));
    OK(PRIMAL_putvartype(t,0,PRIMAL_VAR_TYPE_INT));
    OK(PRIMAL_putcj(t,1,max ? -1 : 1));
    OK(PRIMAL_putcfix(t,3.5));
    OK(PRIMAL_putobjsense(t,max ? PRIMAL_OPTIMIZE_MAXIMIZE : PRIMAL_OPTIMIZE_MINIMIZE));
    OK(PRIMAL_putarow(t,0,2,(int[]){0,1},(double[]){1.0/0.999995,1}));
    OK(PRIMAL_putarow(t,1,2,(int[]){0,1},(double[]){-1e6,1}));
    OK(PRIMAL_putconbound(t,0,PRIMAL_BK_LO,1,0));
    OK(PRIMAL_putconbound(t,1,PRIMAL_BK_LO,-999995,0));
    OK(PRIMAL_putintparam(t,PRIMAL_IPAR_NUM_THREADS,threads));
    if (warm) OK(PRIMAL_putxx(t,PRIMAL_SOL_ITR,(double[]){0,1}));
    if (!cuts) setenv("GMB_NO_MIP_CUTS","1",1);
    OK(PRIMAL_optimize(t));
    if (!cuts) unsetenv("GMB_NO_MIP_CUTS");
    double x[2], p, b, gap; int defined;
    OK(PRIMAL_getxx(t,PRIMAL_SOL_ITR,x));
    OK(PRIMAL_getprimalobj(t,PRIMAL_SOL_ITR,&p));
    OK(PRIMAL_getintinf(t,PRIMAL_IINF_MIO_OBJ_BOUND_DEFINED,&defined));
    CHECK(defined);
    OK(PRIMAL_getdouinf(t,PRIMAL_DINF_MIO_OBJ_BOUND,&b));
    OK(PRIMAL_getdouinf(t,PRIMAL_DINF_MIO_OBJ_ABS_GAP,&gap));
    CHECK(fabs(x[0]) < 1e-7 && fabs(x[1]-1) < 1e-7);
    CHECK(fabs(p-(max ? 2.5 : 4.5)) < 1e-7);
    CHECK(max ? b >= p-1e-7 : b <= p+1e-7);
    CHECK(gap < 1e-6);
    OK(PRIMAL_deletetask(&t));
}
static void capped_bound(int max, int threads, int warm) {

    PRIMALtask_t t = model(1,1);
    OK(PRIMAL_putvarbound(t,0,PRIMAL_BK_LO,0,0));
    OK(PRIMAL_putvartype(t,0,PRIMAL_VAR_TYPE_INT));
    OK(PRIMAL_putcj(t,0,max ? 1 : -1));
    OK(PRIMAL_putcfix(t,-7));
    OK(PRIMAL_putobjsense(t,max ? PRIMAL_OPTIMIZE_MAXIMIZE : PRIMAL_OPTIMIZE_MINIMIZE));
    OK(PRIMAL_putarow(t,0,1,(int[]){0},(double[]){2}));
    OK(PRIMAL_putconbound(t,0,PRIMAL_BK_UP,0,9));
    if (warm) OK(PRIMAL_putxx(t,PRIMAL_SOL_ITR,(double[]){2}));
    OK(PRIMAL_putintparam(t,PRIMAL_IPAR_MIP_MAX_NODES,1));
    OK(PRIMAL_putintparam(t,PRIMAL_IPAR_NUM_THREADS,threads));
    setenv("GMB_NO_MIP_CUTS","1",1);
    PRIMALrescodee rc = PRIMAL_optimize(t);
    unsetenv("GMB_NO_MIP_CUTS");
    CHECK(rc == PRIMAL_RES_OK || rc == PRIMAL_RES_TRM_MAX_ITER);
    double b = NAN; int defined = 0;
    OK(PRIMAL_getintinf(t,PRIMAL_IINF_MIO_OBJ_BOUND_DEFINED,&defined));
    CHECK(defined);
    OK(PRIMAL_getdouinf(t,PRIMAL_DINF_MIO_OBJ_BOUND,&b));
    CHECK(max ? b >= -3-1e-7 : b <= -11+1e-7);
    OK(PRIMAL_deletetask(&t));
}
static void bounded_qp(void) {

    PRIMALtask_t t = model(1,0);
    OK(PRIMAL_putvarbound(t,0,PRIMAL_BK_LO,0,0));
    OK(PRIMAL_putcj(t,0,-1));
    OK(PRIMAL_putqobj(t,1,(int[]){0},(int[]){0},(double[]){1}));
    OK(PRIMAL_optimize(t));
    double x, p, ray;
    OK(PRIMAL_getxx(t,PRIMAL_SOL_ITR,&x));
    OK(PRIMAL_getprimalobj(t,PRIMAL_SOL_ITR,&p));
    CHECK(fabs(x-1)<1e-6 && fabs(p+0.5)<1e-6);
    CHECK(PRIMAL_getprimalray(t,&ray) == PRIMAL_RES_ERR_ARG);
    OK(PRIMAL_deletetask(&t));
}
static void mutation_and_availability(void) {

    PRIMALtask_t t = model(1,0);
    OK(PRIMAL_putvarbound(t,0,PRIMAL_BK_RA,0,1)); OK(PRIMAL_putcj(t,0,-1));
    OK(PRIMAL_optimize(t));
    double p = 123, oldx; OK(PRIMAL_getxx(t,PRIMAL_SOL_ITR,&oldx));
    OK(PRIMAL_putvarbound(t,0,PRIMAL_BK_RA,0,0.5));
    PRIMALsolstae ss; OK(PRIMAL_getsolsta(t,PRIMAL_SOL_ITR,&ss)); CHECK(ss == PRIMAL_SOL_STA_UNKNOWN);
    CHECK(PRIMAL_getprimalobj(t,PRIMAL_SOL_ITR,&p) == PRIMAL_RES_ERR_ARG && p == 123);
    double v; OK(PRIMAL_getprimalinfeas(t,PRIMAL_SOL_ITR,&v)); CHECK(v > 0.49);
    CHECK(PRIMAL_getdouinf(t,PRIMAL_DINF_MIO_CLIQUE_SELECTION_TIME,&p) == PRIMAL_RES_ERR_ARG && p == 123);
    OK(PRIMAL_optimize(t));
    OK(PRIMAL_getprimalobj(t,PRIMAL_SOL_ITR,&p)); CHECK(fabs(p+0.5)<1e-7);
    OK(PRIMAL_deletetask(&t));
}
int main(void) {
    OK(PRIMAL_makeenv(&env,NULL));
    exp_faces(PRIMAL_CT_PEXP); exp_faces(PRIMAL_CT_DEXP);
    dexp_optimum(0,0); dexp_optimum(1,0); dexp_optimum(0,1); dexp_optimum(1,1); quadratic_rows(); bounded_qp(); mutation_and_availability();
    for (int max=0;max<2;max++) for (int threads=1;threads<=2;threads++)
        for (int warm=0;warm<2;warm++) {
            for (int cuts=0;cuts<2;cuts++) mip_repair(max,threads,warm,cuts);
            capped_bound(max,threads,warm);
        }
    OK(PRIMAL_deleteenv(&env));
    printf("Contract checks: %d, failures: %d\n", checks, failures);
    return failures ? 1 : 0;
}
