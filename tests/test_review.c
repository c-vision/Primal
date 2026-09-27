/* Regression checks for original-model feasibility and result handling.
 * SPDX-License-Identifier: Apache-2.0 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "primal_priv.h"

static int checks, failures;
static PRIMALenv_t env;
#define CHECK(c) do { checks++; if (!(c)) { failures++; \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)
#define OK(c) CHECK((c) == PRIMAL_RES_OK)

static PRIMALtask_t model(int n) {
    PRIMALtask_t t = NULL;
    OK(PRIMAL_maketask(env, 0, n, &t));
    return t;
}

/* The oracle counts support, then checks the ranks of its members; it does
 * not use the solver's violation or branching helpers. */
static int sos_feasible(int type, int n, const double *weights, const double *x) {
    int count = 0, ranks[2] = {0, 0};
    for (int j = 0; j < n; j++) {
        if (!isfinite(x[j])) return 0;
        if (fabs(x[j]) <= 1e-6) continue;
        if (count == type) return 0;
        for (int k = 0; k < n; k++) if (weights[k] < weights[j]) ranks[count]++;
        count++;
    }
    return count != 2 || abs(ranks[0] - ranks[1]) == 1;
}

static void sos_fixed(int type, int n, int integer, const double *x, const double *w) {
    PRIMALtask_t t = model(n);
    int *mem = malloc((size_t)n * sizeof(*mem));
    if (!mem) exit(2);
    for (int j = 0; j < n; j++) {
        mem[j] = j;
        OK(PRIMAL_putvarbound(t, j, PRIMAL_BK_FX, x[j], x[j]));
        if (integer) OK(PRIMAL_putvartype(t, j, PRIMAL_VAR_TYPE_INT));
    }
    OK(type == 1 ? PRIMAL_appendsos1(t, n, mem, w) : PRIMAL_appendsos2(t, n, mem, w));
    int rc = PRIMAL_optimize(t);
    PRIMALsolstae ss = PRIMAL_SOL_STA_UNKNOWN;
    OK(PRIMAL_getsolsta(t, PRIMAL_SOL_ITR, &ss));
    if (sos_feasible(type, n, w, x)) {
        double *result = calloc((size_t)n, sizeof(*result));
        if (!result) exit(2);
        CHECK(rc == PRIMAL_RES_OK && ss == PRIMAL_SOL_STA_INTEGER_OPTIMAL);
        OK(PRIMAL_getxx(t, PRIMAL_SOL_ITR, result));
        CHECK(sos_feasible(type, n, w, result));
        for (int j = 0; j < n; j++) CHECK(fabs(result[j] - x[j]) < 1e-7);
        free(result);
    } else {
        PRIMALprostae ps = PRIMAL_PRO_STA_UNKNOWN;
        OK(PRIMAL_getprosta(t, PRIMAL_SOL_ITR, &ps));
        CHECK(rc == PRIMAL_RES_ERR_INFEASIBLE);
        CHECK(ss == PRIMAL_SOL_STA_UNKNOWN && ps == PRIMAL_PRO_STA_PRIM_INFEAS);
    }
    free(mem);
    OK(PRIMAL_deletetask(&t));
}

static void sos_tests(void) {
    for (int type = 1; type <= 2; type++) for (int integer = 0; integer < 2; integer++) {
        /* Every signed support on three variables, including permuted weights. */
        for (int v = 0; v < 27; v++) {
            double x[] = {v % 3 - 1, (v / 3) % 3 - 1, v / 9 - 1};
            sos_fixed(type, 3, integer, x, (double[]){30, 10, 20});
        }
        for (int n = 63; n <= 65; n++) {
            double x[65] = {0}, w[65];
            for (int j = 0; j < n; j++) w[j] = n - j;
            x[0] = -1; x[n-1] = 1;
            sos_fixed(type, n, integer, x, w);
            x[n-1] = 0; x[1] = -1;
            sos_fixed(type, n, integer, x, w);
        }
    }
    for (int type = 1; type <= 2; type++) for (int threads = 1; threads <= 2; threads++) {
        PRIMALtask_t t = model(3), copy = NULL;
        for (int j = 0; j < 3; j++) {
            OK(PRIMAL_putvarbound(t, j, PRIMAL_BK_RA, -1, 1));
            OK(PRIMAL_putcj(t, j, (double[]){1, 3, 2}[j]));
        }
        OK(type == 1 ? PRIMAL_appendsos1(t, 3, (int[]){0,1,2}, (double[]){0,1,2})
                     : PRIMAL_appendsos2(t, 3, (int[]){0,1,2}, (double[]){0,1,2}));
        OK(PRIMAL_clonetask(t, &copy));
        OK(PRIMAL_deletetask(&t));
        OK(PRIMAL_putintparam(copy, PRIMAL_IPAR_NUM_THREADS, threads));
        OK(PRIMAL_optimize(copy));
        double x[3], obj = NAN;
        OK(PRIMAL_getxx(copy, PRIMAL_SOL_ITR, x));
        OK(PRIMAL_getprimalobj(copy, PRIMAL_SOL_ITR, &obj));
        CHECK(sos_feasible(type, 3, (double[]){0,1,2}, x));
        CHECK(fabs(obj - (type == 1 ? -3 : -5)) < 1e-6);
        OK(PRIMAL_deletetask(&copy));
    }
    PRIMALtask_t t = model(3);
    for (int type = 1; type <= 2; type++) {
        int mem[][3] = {{0,0,2},{0,1,2},{0,1,2},{0,1,2}};
        double weights[][3] = {{0,1,2},{0,0,2},{0,NAN,2},{0,1,INFINITY}};
        for (int i = 0; i < 4; i++) {
            CHECK((type == 1 ? PRIMAL_appendsos1(t,3,mem[i],weights[i])
                             : PRIMAL_appendsos2(t,3,mem[i],weights[i])) == PRIMAL_RES_ERR_ARG);
            int count = -1; OK(PRIMAL_getnumsos(t, &count)); CHECK(count == 0);
        }
    }
    OK(PRIMAL_deletetask(&t));
}

static void write_bytes(const char *path, const void *data, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) exit(2);
    CHECK(fwrite(data, 1, n, f) == n); CHECK(fclose(f) == 0);
}
static unsigned char *read_bytes(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) exit(2);
    CHECK(fseek(f, 0, SEEK_END) == 0); long count = ftell(f);
    if (count < 0) exit(2);
    *n = (size_t)count;
    unsigned char *data = malloc(*n+1); if (!data) exit(2);
    rewind(f); CHECK(fread(data, 1, *n, f) == *n); CHECK(fclose(f) == 0);
    data[*n] = 0; return data;
}
static void unchanged_result(PRIMALtask_t t) {
    double x = NAN, p = NAN;
    PRIMALsolstae ss = PRIMAL_SOL_STA_UNKNOWN;
    OK(PRIMAL_getxx(t, PRIMAL_SOL_ITR, &x)); CHECK(x == 1);
    OK(PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &p)); CHECK(p == 1);
    OK(PRIMAL_getsolsta(t, PRIMAL_SOL_ITR, &ss)); CHECK(ss == PRIMAL_SOL_STA_OPTIMAL);
}
static void file_callback(void *handle, const char *data, int len) {
    CHECK(fwrite(data, 1, (size_t)len, (FILE *)handle) == (size_t)len);
}
static void solution_io_tests(void) {
    char path[] = "/tmp/primal-review-XXXXXX";
    int fd = mkstemp(path); if (fd < 0) exit(2); close(fd);
    PRIMALtask_t t = model(1);
    OK(PRIMAL_putvarbound(t,0,PRIMAL_BK_FX,1,1)); OK(PRIMAL_putcj(t,0,1)); OK(PRIMAL_optimize(t));
    CHECK(PRIMAL_writebsolution(t,path,1) == PRIMAL_RES_ERR_ARG);
    CHECK(PRIMAL_readbsolution(t,path,1) == PRIMAL_RES_ERR_ARG);
    CHECK(PRIMAL_writebsolutionhandle(t,file_callback,NULL,1) == PRIMAL_RES_ERR_ARG);
    OK(PRIMAL_writebsolution(t,path,0));
    size_t n; unsigned char *data = read_bytes(path,&n);
    /* Every strict prefix must fail, including header-only and mid-double EOF. */
    for (size_t i = 0; i < n; i++) {
        write_bytes(path,data,i);
        CHECK(PRIMAL_readbsolution(t,path,0) == PRIMAL_RES_ERR_FILE);
        unchanged_result(t);
    }
    for (size_t i = 16+5*sizeof(int); i+sizeof(double) <= n; i += sizeof(double)) {
        double old, bad = NAN; memcpy(&old,data+i,sizeof old); memcpy(data+i,&bad,sizeof bad);
        write_bytes(path,data,n); CHECK(PRIMAL_readbsolution(t,path,0) == PRIMAL_RES_ERR_FILE);
        unchanged_result(t); memcpy(data+i,&old,sizeof old);
    }
    unsigned char saved = data[12]; data[12] = '9';
    write_bytes(path,data,n); CHECK(PRIMAL_readbsolution(t,path,0) == PRIMAL_RES_ERR_FILE);
    unchanged_result(t); data[12] = saved;
    FILE *f = fopen(path,"wb"); if (!f) exit(2);
    OK(PRIMAL_writebsolutionhandle(t,file_callback,f,0)); CHECK(fclose(f) == 0);
    size_t emitted; unsigned char *copy = read_bytes(path,&emitted);
    CHECK(emitted == n && memcmp(copy,data,n) == 0); free(copy); free(data);
    const char *bad_text[] = {
        "PRIMAL-SOLUTION 1\nsolsta 0\n", "PRIMAL-SOLUTION 9\n",
        "PRIMAL-SOLUTION 1\nsolsta 1 nan\n", "PRIMAL-SOLUTION 1\nxx 999999999999999999999999 1\n",
        "PRIMAL-SOLUTION 1\nsolsta 1 1\nsolsta 1 1\n"
    };
    for (size_t i = 0; i < sizeof bad_text / sizeof *bad_text; i++) {
        write_bytes(path,bad_text[i],strlen(bad_text[i]));
        CHECK(PRIMAL_readsolution(t,PRIMAL_SOL_ITR,path) == PRIMAL_RES_ERR_FILE); unchanged_result(t);
    }
    const char *bad_json[] = {"{}", "{\"pobj\":1,\"xx\":[1]}", "{\"pobj\":nan}",
        "{\"pobj\":1e999}", "{\"xx\":[1,]}", "{\"pobj\":1,\"pobj\":2}"};
    for (size_t i = 0; i < sizeof bad_json / sizeof *bad_json; i++) {
        CHECK(PRIMAL_readjsonstring(t,bad_json[i]) == PRIMAL_RES_ERR_FILE); unchanged_result(t);
    }
    OK(PRIMAL_deletetask(&t));
    /* Long text lines exceed the former 64-byte buffer. All three formats keep
     * every coordinate and label imports as unverified, even for the same model. */
    t = model(20);
    for (int j = 0; j < 20; j++) OK(PRIMAL_putvarbound(t,j,PRIMAL_BK_FX,1000+j,1000+j));
    for (int format = 0; format < 3; format++) {
        OK(PRIMAL_optimize(t));
        if (format == 0) OK(PRIMAL_writesolution(t,PRIMAL_SOL_ITR,path));
        if (format == 1) OK(PRIMAL_writebsolution(t,path,0));
        if (format == 2) OK(PRIMAL_writejsonsol(t,path));
        if (format == 0) OK(PRIMAL_readsolution(t,PRIMAL_SOL_ITR,path));
        if (format == 1) OK(PRIMAL_readbsolution(t,path,0));
        if (format == 2) OK(PRIMAL_readjsonsol(t,path));
        double x[20], obj = 123;
        OK(PRIMAL_getxx(t,PRIMAL_SOL_ITR,x));
        for (int j = 0; j < 20; j++) CHECK(x[j] == 1000+j);
        PRIMALsolstae ss; PRIMALprostae ps;
        OK(PRIMAL_getsolsta(t,PRIMAL_SOL_ITR,&ss)); CHECK(ss == PRIMAL_SOL_STA_UNKNOWN);
        OK(PRIMAL_getprosta(t,PRIMAL_SOL_ITR,&ps)); CHECK(ps == PRIMAL_PRO_STA_UNKNOWN);
        CHECK(PRIMAL_getprimalobj(t,PRIMAL_SOL_ITR,&obj) == PRIMAL_RES_ERR_ARG && obj == 123);
        CHECK(PRIMAL_getdualobj(t,PRIMAL_SOL_ITR,&obj) == PRIMAL_RES_ERR_ARG && obj == 123);
    }
    OK(PRIMAL_optimize(t));
    double objective = NAN;
    OK(PRIMAL_getprimalobj(t,PRIMAL_SOL_ITR,&objective)); CHECK(objective == 0);
    OK(PRIMAL_deletetask(&t)); CHECK(remove(path) == 0);
}

static void bar_buffer_tests(void) {
    PRIMALtask_t t = model(0);
    OK(PRIMAL_appendbarvars(t,2,(int[]){2,3}));
    /* This test exercises user-supplied buffers, not an optimization verdict. */
    OK(PRIMAL_putsolution(t, PRIMAL_SOL_ITR, NULL, NULL, NULL, NULL,
                         NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    double x2[] = {1,0,0,2}, x3[] = {3,0,0,0,4,0,0,0,5};
    OK(PRIMAL_putbarxj(t,PRIMAL_SOL_ITR,0,x2)); OK(PRIMAL_putbarxj(t,PRIMAL_SOL_ITR,1,x3));
    OK(PRIMAL_putbarsj(t,PRIMAL_SOL_ITR,0,x2)); OK(PRIMAL_putbarsj(t,PRIMAL_SOL_ITR,1,x3));
    for (int dual = 0; dual < 2; dual++) {
        double guarded[15]; for (int i = 0; i < 15; i++) guarded[i] = 12345;
        OK(dual ? PRIMAL_getbarsj(t,PRIMAL_SOL_ITR,1,guarded+1)
                : PRIMAL_getbarxj(t,PRIMAL_SOL_ITR,1,guarded+1));
        CHECK(guarded[0] == 12345 && guarded[10] == 12345);
        for (int i = 0; i < 9; i++) CHECK(guarded[i+1] == x3[i]);
        for (int i = 0; i < 15; i++) guarded[i] = 12345;
        CHECK((dual ? PRIMAL_getbarsslice(t,PRIMAL_SOL_ITR,0,2,12,guarded+1)
                    : PRIMAL_getbarxslice(t,PRIMAL_SOL_ITR,0,2,12,guarded+1)) == PRIMAL_RES_ERR_ARG);
        for (int i = 0; i < 15; i++) CHECK(guarded[i] == 12345);
        OK(dual ? PRIMAL_getbarsslice(t,PRIMAL_SOL_ITR,0,2,13,guarded+1)
                : PRIMAL_getbarxslice(t,PRIMAL_SOL_ITR,0,2,13,guarded+1));
        CHECK(guarded[0] == 12345 && guarded[14] == 12345);
        for (int i = 0; i < 13; i++) CHECK(guarded[i+1] == (i < 4 ? x2[i] : x3[i-4]));
    }
    char path[] = "/tmp/primal-bar-review-XXXXXX";
    int fd = mkstemp(path); if (fd < 0) exit(2); close(fd);
    for (int format = 0; format < 3; format++) {
        if (format == 0) { OK(PRIMAL_writesolution(t,PRIMAL_SOL_ITR,path)); OK(PRIMAL_readsolution(t,PRIMAL_SOL_ITR,path)); }
        if (format == 1) { OK(PRIMAL_writebsolution(t,path,0)); OK(PRIMAL_readbsolution(t,path,0)); }
        if (format == 2) { OK(PRIMAL_writejsonsol(t,path)); OK(PRIMAL_readjsonsol(t,path)); }
        double x[13]; OK(PRIMAL_getbarxslice(t,PRIMAL_SOL_ITR,0,2,13,x));
        for (int i = 0; i < 13; i++) CHECK(x[i] == (i < 4 ? x2[i] : x3[i-4]));
    }
    CHECK(remove(path) == 0); OK(PRIMAL_deletetask(&t));
}

static PRIMALtask_t affine_model(int conic, int shared, double coefficient, double g, double offset) {
    PRIMALtask_t t = model(1);
    PRIMALint64t dom;
    OK(PRIMAL_putvarbound(t,0,PRIMAL_BK_FR,0,0)); OK(PRIMAL_putcj(t,0,1));
    OK(PRIMAL_appendafes(t,2)); OK(PRIMAL_putafefentry(t,0,0,coefficient)); OK(PRIMAL_putafeg(t,0,g));
    if (conic) OK(PRIMAL_appendquadraticconedomain(t,2,&dom));
    else OK(PRIMAL_appendrplusdomain(t,1,&dom));
    OK(PRIMAL_appendacc(t,dom,conic ? 2 : 1,(PRIMALint64t[]){0,1},(double[]){offset,0}));
    if (shared) OK(PRIMAL_appendacc(t,dom,conic ? 2 : 1,(PRIMALint64t[]){0,1},(double[]){offset+1,0}));
    return t;
}
static double solve_x(PRIMALtask_t t, double expected) {
    OK(PRIMAL_optimize(t));
    double x = NAN; OK(PRIMAL_getxxslice(t,PRIMAL_SOL_ITR,0,1,&x));
    CHECK(isfinite(x) && fabs(x-expected) < 2e-6);
    return x;
}
static void affine_mutation_tests(void) {
    for (int conic = 0; conic < 2; conic++) for (int shared = 0; shared < 2; shared++) {
        PRIMALtask_t t = affine_model(conic,shared,1,-1,0);
        solve_x(t,1+shared);
        OK(PRIMAL_putafeg(t,0,-4)); OK(PRIMAL_putafefentry(t,0,0,2));
        OK(PRIMAL_putaccb(t,0,conic ? 2 : 1,(double[]){1,0}));
        if (shared) OK(PRIMAL_putaccbj(t,1,0,2));
        double expected = (5+shared)/2.0;
        double edited = solve_x(t,expected);
        PRIMALtask_t fresh = affine_model(conic,shared,2,-4,1);
        CHECK(fabs(edited-solve_x(fresh,expected)) < 2e-6);
        PRIMALtask_t clone = NULL; OK(PRIMAL_clonetask(t,&clone));
        int nv,nc,cv,cc;
        OK(PRIMAL_getnumvar(t,&nv)); OK(PRIMAL_getnumcon(t,&nc));
        OK(PRIMAL_getnumvar(clone,&cv)); OK(PRIMAL_getnumcon(clone,&cc)); CHECK(nv == cv && nc == cc);
        OK(PRIMAL_emptyafefrow(clone,0));
        OK(PRIMAL_putafefrow(clone,0,1,(int[]){0},(double[]){1}));
        solve_x(clone,5+shared);
        /* The clone's edits do not affect the source model. */
        solve_x(t,expected);
        OK(PRIMAL_deletetask(&clone)); OK(PRIMAL_deletetask(&fresh)); OK(PRIMAL_deletetask(&t));
    }
    PRIMALtask_t t = model(1);
    PRIMALint64t dom; int sym;
    OK(PRIMAL_putvarbound(t,0,PRIMAL_BK_FR,0,0)); OK(PRIMAL_putcj(t,0,1));
    OK(PRIMAL_appendcons(t,1)); OK(PRIMAL_putconbound(t,0,PRIMAL_BK_FX,2,2));
    OK(PRIMAL_appendbarvars(t,2,(int[]){1,1}));
    OK(PRIMAL_appendsparsesymmat(t,1,1,(int[]){0},(int[]){0},(double[]){1},&sym));
    OK(PRIMAL_putbaraij(t,0,1,1,&sym,(double[]){1}));
    OK(PRIMAL_appendafes(t,2)); OK(PRIMAL_putafefentry(t,0,0,1)); OK(PRIMAL_putafeg(t,1,1));
    OK(PRIMAL_putafebarfentry(t,0,1,1,(PRIMALint64t[]){sym},(double[]){1}));
    OK(PRIMAL_appendquadraticconedomain(t,2,&dom));
    OK(PRIMAL_appendacc(t,dom,2,(PRIMALint64t[]){0,1},NULL));
    solve_x(t,-1); /* x + B >= 1 with B = 2 */
    OK(PRIMAL_putafebarfentry(t,0,1,1,(PRIMALint64t[]){sym},(double[]){2}));
    solve_x(t,-3);
    PRIMALtask_t clone = NULL; OK(PRIMAL_clonetask(t,&clone)); solve_x(clone,-3);
    OK(PRIMAL_removebarvars(clone,1,(int[]){0})); solve_x(clone,-3);
    OK(PRIMAL_emptyafebarfrow(clone,0)); solve_x(clone,1);
    solve_x(t,-3);
    OK(PRIMAL_deletetask(&clone)); OK(PRIMAL_deletetask(&t));
}

static void affine_removal_tests(void) {
    PRIMALtask_t t = model(2);
    PRIMALint64t dom;
    OK(PRIMAL_putvarbound(t,0,PRIMAL_BK_FX,0,0));
    OK(PRIMAL_putvarbound(t,1,PRIMAL_BK_FR,0,0)); OK(PRIMAL_putcj(t,1,1));
    OK(PRIMAL_appendcons(t,1)); OK(PRIMAL_putconbound(t,0,PRIMAL_BK_FR,0,0));
    OK(PRIMAL_appendafes(t,2)); OK(PRIMAL_putafefentry(t,0,1,1)); OK(PRIMAL_putafeg(t,0,-2));
    OK(PRIMAL_appendquadraticconedomain(t,2,&dom));
    OK(PRIMAL_appendacc(t,dom,2,(PRIMALint64t[]){0,1},NULL));
    OK(PRIMAL_removevars(t,1,(int[]){0})); OK(PRIMAL_removecons(t,1,(int[]){0}));
    solve_x(t,2);
    CHECK(PRIMAL_removevars(t,1,(int[]){1}) == PRIMAL_RES_ERR_ARG);
    CHECK(PRIMAL_removecons(t,1,(int[]){0}) == PRIMAL_RES_ERR_ARG);
    CHECK(PRIMAL_removecones(t,1,(int[]){0}) == PRIMAL_RES_ERR_ARG);
    solve_x(t,2); OK(PRIMAL_deletetask(&t));
}

static PRIMALtask_t disjunction(double bound, int singleton) {
    PRIMALtask_t t = model(1);
    PRIMALint64t dm, dp;
    /* Deliberately irrelevant finite bl/bu on a free variable. */
    OK(PRIMAL_putvarbound(t,0,singleton ? PRIMAL_BK_FR : PRIMAL_BK_RA,0,bound));
    OK(PRIMAL_putcj(t,0,1)); OK(PRIMAL_putobjsense(t,PRIMAL_OBJECTIVE_SENSE_MAXIMIZE));
    OK(PRIMAL_appendafes(t,1)); OK(PRIMAL_putafefentry(t,0,0,1));
    OK(PRIMAL_appendrminusdomain(t,1,&dm)); OK(PRIMAL_appendrplusdomain(t,1,&dp));
    OK(PRIMAL_appenddjcs(t,1));
    OK(PRIMAL_putdjc(t,0,2,(PRIMALint64t[]){dm,dp},2,(PRIMALint64t[]){0,0},
                    (double[]){0,3e6},2,(PRIMALint64t[]){1,1}));
    if (singleton == 1) {
        int r; OK(PRIMAL_getnumcon(t,&r)); OK(PRIMAL_appendcons(t,1));
        OK(PRIMAL_putarow(t,r,1,(int[]){0},(double[]){1}));
        OK(PRIMAL_putconbound(t,r,PRIMAL_BK_FX,3e6,3e6));
    }
    return t;
}
static void disjunction_tests(void) {
    for (int threads = 1; threads <= 2; threads++) {
        PRIMALtask_t t = disjunction(10,1);
        OK(PRIMAL_putintparam(t,PRIMAL_IPAR_NUM_THREADS,threads));
        double x = solve_x(t,3e6); CHECK(x <= 1e-6 || x >= 3e6-1e-6);
        double v; OK(PRIMAL_getpvioldjc(t,PRIMAL_SOL_ITR,1,(PRIMALint64t[]){0},&v)); CHECK(v <= 1e-6);
        PRIMALtask_t c = NULL; OK(PRIMAL_clonetask(t,&c));
        int nv, nc, cv, cc; OK(PRIMAL_getnumvar(t,&nv)); OK(PRIMAL_getnumcon(t,&nc));
        OK(PRIMAL_getnumvar(c,&cv)); OK(PRIMAL_getnumcon(c,&cc)); CHECK(nv == cv && nc == cc);
        solve_x(c,3e6); OK(PRIMAL_deletetask(&c)); OK(PRIMAL_deletetask(&t));
    }
    PRIMALtask_t t = disjunction(10,0);
    solve_x(t,0);
    OK(PRIMAL_putvarbound(t,0,PRIMAL_BK_RA,0,4e6)); solve_x(t,4e6);
    OK(PRIMAL_putafeg(t,0,3e6)); solve_x(t,4e6);
    OK(PRIMAL_putafefentry(t,0,0,-1)); solve_x(t,4e6); /* -x+3e6 <= 0 */
    OK(PRIMAL_putafeg(t,0,5e6)); solve_x(t,2e6); /* -x+5e6 >= 3e6 */
    OK(PRIMAL_emptyafefrow(t,0)); solve_x(t,4e6); /* constant true term */
    CHECK(PRIMAL_removevars(t,1,(int[]){1}) == PRIMAL_RES_ERR_ARG);
    CHECK(PRIMAL_removecons(t,1,(int[]){0}) == PRIMAL_RES_ERR_ARG);
    OK(PRIMAL_deletetask(&t));
    t = disjunction(10,2);
    CHECK(PRIMAL_optimize(t) == PRIMAL_RES_ERR_ARG);
    PRIMALsolstae ss; OK(PRIMAL_getsolsta(t,PRIMAL_SOL_ITR,&ss)); CHECK(ss == PRIMAL_SOL_STA_UNKNOWN);
    double obj; CHECK(PRIMAL_getprimalobj(t,PRIMAL_SOL_ITR,&obj) == PRIMAL_RES_ERR_ARG);
    OK(PRIMAL_putvarbound(t,0,PRIMAL_BK_RA,0,4e6)); solve_x(t,4e6);
    OK(PRIMAL_putvarbound(t,0,PRIMAL_BK_FR,0,10)); CHECK(PRIMAL_optimize(t) == PRIMAL_RES_ERR_ARG);
    OK(PRIMAL_deletetask(&t));
    /* Invalid descriptions cannot consume the slot or extend the model. */
    t = model(1); PRIMALint64t dom;
    OK(PRIMAL_appendrminusdomain(t,1,&dom)); OK(PRIMAL_appendafes(t,1)); OK(PRIMAL_appenddjcs(t,1));
    CHECK(PRIMAL_putdjc(t,0,1,&dom,1,(PRIMALint64t[]){0},(double[]){INFINITY},1,(PRIMALint64t[]){1}) == PRIMAL_RES_ERR_ARG);
    CHECK(PRIMAL_putdjcslice(t,0,1,1,&dom,1,(PRIMALint64t[]){0},NULL,1,(PRIMALint64t[]){2},(PRIMALint64t[]){1}) == PRIMAL_RES_ERR_ARG);
    CHECK(PRIMAL_putdjcslice(t,0,1,1,&dom,1,(PRIMALint64t[]){0},NULL,1,NULL,(PRIMALint64t[]){1}) == PRIMAL_RES_ERR_NULL);
    PRIMALint64t nt; OK(PRIMAL_getdjcnumterm(t,0,&nt)); CHECK(nt == 0);
    int nv, nc; OK(PRIMAL_getnumvar(t,&nv)); OK(PRIMAL_getnumcon(t,&nc)); CHECK(nv == 1 && nc == 0);
    OK(PRIMAL_putafefentry(t,0,0,1));
    OK(PRIMAL_putdjc(t,0,1,&dom,1,(PRIMALint64t[]){0},(double[]){5},1,(PRIMALint64t[]){1}));
    OK(PRIMAL_putcj(t,0,1)); OK(PRIMAL_putobjsense(t,PRIMAL_OBJECTIVE_SENSE_MAXIMIZE));
    solve_x(t,5); /* A single clause needs no big-M even on a free variable. */
    OK(PRIMAL_deletetask(&t));
}

static void bounded_psd_face(PRIMALtask_t t) {
    /* Trace(B)=1 and B positive semidefinite imply x0=B00<=1.
     * B=diag(1,0,...,0) attains the objective -1. Entry bounds
     * [-1,1] are redundant and keep the LP away from its artificial cap. */
    enum { D = 20 };
    int si[D], sj[D];
    double sv[D];
    for (int a = 0; a < D; a++) { si[a] = a; sj[a] = a; sv[a] = 1.0; }
    int m00, mI;
    OK(PRIMAL_appendcons(t, 2));
    OK(PRIMAL_appendsparsesymmat(t, D, 1, (int[]){0}, (int[]){0}, (double[]){1.0}, &m00));
    OK(PRIMAL_appendsparsesymmat(t, D, D, si, sj, sv, &mI));
    int dim = D;
    OK(PRIMAL_appendbarvars(t, 1, &dim));
    OK(PRIMAL_appendvars(t, 1));
    OK(PRIMAL_putvarbound(t, 0, PRIMAL_BK_LO, 0.0, INFINITY));
    OK(PRIMAL_putcj(t, 0, -1.0));
    OK(PRIMAL_putarow(t, 0, 1, (int[]){0}, (double[]){1.0}));
    OK(PRIMAL_putbaraij(t, 0, 0, 1, &m00, (double[]){-1.0}));
    OK(PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 0.0, 0.0));
    OK(PRIMAL_putbaraij(t, 1, 0, 1, &mI, (double[]){1.0}));
    OK(PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 1.0, 1.0));
    for (int i=0;i<D;i++) for (int j=i;j<D;j++) {
      int r,m; OK(PRIMAL_getnumcon(t,&r)); OK(PRIMAL_appendcons(t,1));
      OK(PRIMAL_appendsparsesymmat(t,D,1,&i,&j,(double[]){i == j ? 1 : 0.5},&m));
      OK(PRIMAL_putbaraij(t,r,0,1,&m,(double[]){1}));
      OK(PRIMAL_putconbound(t,r,PRIMAL_BK_RA,i == j ? 0 : -1,1));
    }

}

static void psd_stall_test(void) {
    PRIMALtask_t t = model(0);
    bounded_psd_face(t);
    /* Exercise the same cut entry point used for MIP relaxations, without
     * the native SDP solve or the public dispatcher's later quality gate. */
    OK(opt_prepare(t));
    CHECK(optimize_sdp(t,1) == PRIMAL_RES_TRM_MAX_ITER);
    PRIMALsolstae ss;
    OK(PRIMAL_getsolsta(t,PRIMAL_SOL_ITR,&ss));
    CHECK(ss == PRIMAL_SOL_STA_UNKNOWN);
    double x = 123, obj = 123;
    CHECK(PRIMAL_getxx(t,PRIMAL_SOL_ITR,&x) == PRIMAL_RES_ERR_ARG);
    CHECK(PRIMAL_getprimalobj(t,PRIMAL_SOL_ITR,&obj) == PRIMAL_RES_ERR_ARG);
    CHECK(x == 123 && obj == 123);
    OK(PRIMAL_deletetask(&t));
}

static void rejected_edit_tests(void) {
    PRIMALtask_t t = model(1);
    OK(PRIMAL_putvarbound(t,0,PRIMAL_BK_FX,1,1));
    OK(PRIMAL_putcj(t,0,1)); OK(PRIMAL_optimize(t));
#define REJECT(call) do { OK(PRIMAL_optimize(t)); CHECK((call) == PRIMAL_RES_ERR_ARG); unchanged_result(t); } while (0)
    REJECT(PRIMAL_putcj(t,99,1));
    REJECT(PRIMAL_putclist(t,1,(int[]){99},(double[]){1}));
    REJECT(PRIMAL_putcslice(t,0,2,(double[]){1,2}));
    REJECT(PRIMAL_putobjsense(t,(PRIMALobjsensee)99));
    REJECT(PRIMAL_putvarbound(t,0,PRIMAL_BK_RA,2,1));
    REJECT(PRIMAL_putconbound(t,0,PRIMAL_BK_FX,1,1));
    REJECT(PRIMAL_putvarboundlist(t,1,(int[]){99},(PRIMALboundkeye[]){PRIMAL_BK_FX},(double[]){1},(double[]){1}));
    REJECT(PRIMAL_putacol(t,99,0,NULL,NULL));
    REJECT(PRIMAL_putarow(t,99,0,NULL,NULL));
    REJECT(PRIMAL_putaij(t,99,0,1));
    REJECT(PRIMAL_putqobj(t,1,(int[]){99},(int[]){0},(double[]){1}));
    REJECT(PRIMAL_putqobjij(t,0,0,NAN));
    REJECT(PRIMAL_putqconk(t,99,0,NULL,NULL,NULL));
    REJECT(PRIMAL_putqcon(t,1,(int[]){99},(int[]){0},(int[]){0},(double[]){1}));
    REJECT(PRIMAL_putvartype(t,0,(PRIMALvariabletypee)99));
    REJECT(PRIMAL_putvartypelist(t,1,(int[]){99},(PRIMALvariabletypee[]){PRIMAL_VAR_TYPE_INT}));
    REJECT(PRIMAL_appendvars(t,-1)); REJECT(PRIMAL_appendcons(t,-1));
    REJECT(PRIMAL_appendafes(t,-1)); REJECT(PRIMAL_appenddjcs(t,-1));
    REJECT(PRIMAL_appendbarvars(t,1,(int[]){0}));
    REJECT(PRIMAL_putafeg(t,99,1));
    REJECT(PRIMAL_putafefentry(t,99,0,1));
    REJECT(PRIMAL_putafefrow(t,99,0,NULL,NULL));
    REJECT(PRIMAL_putafefcol(t,99,0,NULL,NULL));
    REJECT(PRIMAL_emptyafefrow(t,99)); REJECT(PRIMAL_emptyafefcol(t,99));
    REJECT(PRIMAL_emptyafebarfrow(t,99));
    REJECT(PRIMAL_appendcone(t,(PRIMALconetypee)99,0,1,(int[]){0}));
    REJECT(PRIMAL_appendsos1(t,2,(int[]){0,0},(double[]){0,1}));
    REJECT(PRIMAL_appendsos2(t,1,(int[]){99},(double[]){0}));
    REJECT(PRIMAL_appendacc(t,99,0,NULL,NULL));
    REJECT(PRIMAL_putaccbj(t,99,0,1));
    REJECT(PRIMAL_putaccb(t,99,0,NULL));
    REJECT(PRIMAL_putbaraij(t,99,0,0,NULL,NULL));
    REJECT(PRIMAL_putbarablockij(t,99,0,0,NULL,NULL));
    REJECT(PRIMAL_putbarcj(t,99,0,NULL,NULL));
    REJECT(PRIMAL_removevars(t,1,(int[]){99}));
    REJECT(PRIMAL_removecons(t,1,(int[]){99}));
    REJECT(PRIMAL_removecones(t,1,(int[]){99}));
    REJECT(PRIMAL_removebarvars(t,1,(int[]){99}));
#undef REJECT
    OK(PRIMAL_appendvars(t,0)); OK(PRIMAL_appendcons(t,0)); unchanged_result(t);
    OK(PRIMAL_putcj(t,0,2));
    double objective=123;
    CHECK(PRIMAL_getprimalobj(t,PRIMAL_SOL_ITR,&objective)==PRIMAL_RES_ERR_ARG && objective==123);
    OK(PRIMAL_optimize(t)); OK(PRIMAL_getprimalobj(t,PRIMAL_SOL_ITR,&objective)); CHECK(objective==2);
    OK(PRIMAL_deletetask(&t));
}

static void legacy_json_and_mip_write_tests(void) {
    char path[]="/tmp/primal-sol-XXXXXX";
    int fd=mkstemp(path); if(fd<0) exit(2); close(fd);
    PRIMALtask_t t=model(1);
    /* Exact ten-key schema emitted by the original JSON writer. */
    const char *legacy="{\"pobj\":1,\"dobj\":1,\"solsta\":1,\"prosta\":1,"
        "\"xx\":[1],\"y\":[],\"slc\":[],\"suc\":[],\"slx\":[-1],\"sux\":[0]}";
    write_bytes(path,legacy,strlen(legacy));
    OK(PRIMAL_readjsonsol(t,path));
    double x=0, objective=123;
    OK(PRIMAL_getxx(t,PRIMAL_SOL_ITR,&x)); CHECK(x==1);
    CHECK(PRIMAL_getprimalobj(t,PRIMAL_SOL_ITR,&objective)==PRIMAL_RES_ERR_ARG && objective==123);
    PRIMALsolstae ss; OK(PRIMAL_getsolsta(t,PRIMAL_SOL_ITR,&ss)); CHECK(ss==PRIMAL_SOL_STA_UNKNOWN);
    const char *short_array="{\"pobj\":1,\"dobj\":1,\"solsta\":1,\"prosta\":1,"
        "\"xx\":[],\"y\":[],\"slc\":[],\"suc\":[],\"slx\":[-1],\"sux\":[0]}";
    CHECK(PRIMAL_readjsonstring(t,short_array)==PRIMAL_RES_ERR_FILE);
    OK(PRIMAL_getxx(t,PRIMAL_SOL_ITR,&x)); CHECK(x==1);
    OK(PRIMAL_putvarbound(t,0,PRIMAL_BK_RA,0,10));
    OK(PRIMAL_putvartype(t,0,PRIMAL_VAR_TYPE_INT)); OK(PRIMAL_putcj(t,0,1));
    OK(PRIMAL_putxx(t,PRIMAL_SOL_ITR,(double[]){5}));
    OK(PRIMAL_putdouparam(t,PRIMAL_DPAR_MIO_MAX_TIME,0));
    CHECK(PRIMAL_optimize(t)==PRIMAL_RES_TRM_MAX_TIME);
    OK(PRIMAL_getxx(t,PRIMAL_SOL_ITG,&x)); CHECK(x==5);
    int defined=-1; OK(PRIMAL_getintinf(t,PRIMAL_IINF_MIO_OBJ_BOUND_DEFINED,&defined)); CHECK(!defined);
    CHECK(PRIMAL_getdualobj(t,PRIMAL_SOL_ITG,&objective)==PRIMAL_RES_ERR_ARG && objective==123);
    for(int format=0;format<3;format++) {
        PRIMALtask_t copy=model(1);
        if(format==0) { OK(PRIMAL_writesolution(t,PRIMAL_SOL_ITG,path)); OK(PRIMAL_readsolution(copy,PRIMAL_SOL_ITR,path)); }
        if(format==1) { OK(PRIMAL_writebsolution(t,path,0)); OK(PRIMAL_readbsolution(copy,path,0)); }
        if(format==2) { OK(PRIMAL_writejsonsol(t,path)); OK(PRIMAL_readjsonsol(copy,path)); }
        x=0; OK(PRIMAL_getxx(copy,PRIMAL_SOL_ITR,&x)); CHECK(x==5);
        OK(PRIMAL_deletetask(&copy));
    }
    /* The summaries must not turn an unavailable dual into a numeric zero. */
    FILE *capture=tmpfile(); if(!capture) exit(2);
    fflush(stdout); int saved=dup(STDOUT_FILENO); CHECK(saved>=0);
    CHECK(dup2(fileno(capture),STDOUT_FILENO)>=0);
    OK(PRIMAL_solutionsummary(t,0)); OK(PRIMAL_analyzesolution(t,0,PRIMAL_SOL_ITG));
    fflush(stdout); CHECK(dup2(saved,STDOUT_FILENO)>=0); close(saved);
    rewind(capture); char text[1024]={0}; size_t count=fread(text,1,sizeof text-1,capture); text[count]=0; fclose(capture);
    CHECK(strstr(text,"Dual objective: unavailable")!=NULL && strstr(text,"dobj=unavailable")!=NULL);
    OK(PRIMAL_deletetask(&t)); CHECK(remove(path)==0);
}

static void quadratic_lower_activity_test(void) {
    PRIMALtask_t t=model(2); OK(PRIMAL_appendcons(t,1));
    OK(PRIMAL_putvarbound(t,0,PRIMAL_BK_RA,0,2)); OK(PRIMAL_putvarbound(t,1,PRIMAL_BK_FX,1,1));
    OK(PRIMAL_putcj(t,0,1)); OK(PRIMAL_putaij(t,0,0,1));
    OK(PRIMAL_putconbound(t,0,PRIMAL_BK_LO,0,0));
    OK(PRIMAL_putqconk(t,0,1,(int[]){1},(int[]){1},(double[]){-2}));
    OK(PRIMAL_optimize(t));
    double x[2], activity=123, violation=123;
    OK(PRIMAL_getxx(t,PRIMAL_SOL_ITR,x)); CHECK(fabs(x[0]-1)<1e-6 && fabs(x[1]-1)<1e-6);
    OK(PRIMAL_getxc(t,PRIMAL_SOL_ITR,&activity)); CHECK(fabs(activity-(x[0]-x[1]*x[1]))<1e-9);
    OK(PRIMAL_putxxslice(t,PRIMAL_SOL_ITR,0,2,(double[]){0,1}));
    OK(PRIMAL_getxc(t,PRIMAL_SOL_ITR,&activity)); CHECK(fabs(activity+1)<1e-9);
    OK(PRIMAL_getpviolcon(t,PRIMAL_SOL_ITR,1,(int[]){0},&violation)); CHECK(fabs(violation-1)<1e-9);
    OK(PRIMAL_deletetask(&t));
}

static void disjunction_export_clone_tests(void) {
    PRIMALtask_t t=disjunction(4e6,0), copy=NULL;
    char temporary[]="/tmp/primal-export-XXXXXX";
    int fd=mkstemp(temporary); if(fd<0) exit(2); close(fd);
    char path[128]; snprintf(path,sizeof path,"%s.lp",temporary);
    CHECK(rename(temporary,path)==0);
    OK(PRIMAL_writedata(t,path));
    copy=model(0); OK(PRIMAL_readdata(copy,path));
    /* The LP expansion must retain x<=0 OR x>=3e6 before any optimize call. */
    OK(PRIMAL_putobjsense(copy,PRIMAL_OPTIMIZE_MINIMIZE));
    OK(PRIMAL_putvarbound(copy,0,PRIMAL_BK_LO,1,0)); solve_x(copy,3e6);
    OK(PRIMAL_deletetask(&copy));
    solve_x(t,4e6); OK(PRIMAL_putvarbound(t,0,PRIMAL_BK_FR,0,0));
    double before[3], after[3]; OK(PRIMAL_getxx(t,PRIMAL_SOL_ITR,before));
    OK(PRIMAL_clonetask(t,&copy));
    OK(PRIMAL_getxx(t,PRIMAL_SOL_ITR,after)); CHECK(memcmp(before,after,sizeof before)==0);
    OK(PRIMAL_putvarbound(copy,0,PRIMAL_BK_RA,0,5e6)); solve_x(copy,5e6);
    OK(PRIMAL_deletetask(&copy)); OK(PRIMAL_deletetask(&t));
    CHECK(remove(path)==0);
}

static void node_witness_test(void) {
    for (int infeasible=0; infeasible<2; infeasible++) {
        PRIMALtask_t t=model(3);
        /* (x0,1,1) in QUAD needs x0>=sqrt(2). The linear cone faces
         * prove x0<=0.5 impossible; x0<=3 leaves a feasible problem. */
        double lo[3]={0,1,1}, up[3]={infeasible ? 0.5 : 3,1,1};
        for (int j=0;j<3;j++) OK(PRIMAL_putvarbound(t,j,PRIMAL_BK_RA,lo[j],up[j]));
        OK(PRIMAL_putcj(t,0,1));
        OK(PRIMAL_appendcone(t,PRIMAL_CT_QUAD,0,3,(int[]){0,1,2}));
        OK(PRIMAL_putintparam(t,PRIMAL_IPAR_INTPNT_MAX_ITERATIONS,1));
        double x[3]={123,123,123}, objective=123;
        int status=mip_relax_conic(t,1,env,lo,up,NULL,NULL,x,&objective,NULL);
        CHECK(status==(infeasible ? 1 : 3));
        CHECK(x[0]==123 && x[1]==123 && x[2]==123 && objective==123);
        OK(PRIMAL_deletetask(&t));
    }
}

int main(void) {
    OK(PRIMAL_makeenv(&env, NULL));
    sos_tests();
    solution_io_tests();
    bar_buffer_tests();
    affine_mutation_tests();
    affine_removal_tests();
    disjunction_tests();
    rejected_edit_tests();
    legacy_json_and_mip_write_tests();
    quadratic_lower_activity_test();
    disjunction_export_clone_tests();
    node_witness_test();
    psd_stall_test();
    OK(PRIMAL_deleteenv(&env));
    printf("Review checks: %d, failures: %d\n", checks, failures);
    return failures ? 1 : 0;
}
