/* Persistent worker for tools/socp_compare.py. Model construction and public
 * quality getters are outside the PRIMAL_optimize timer. Build under out/. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "primal.h"

static double now(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

int main(void) {
    int n, variant;
    while (scanf("%d %d", &n, &variant) == 2) {
        if (n < 2 || n % 2 || variant < 0 || variant > 2) return 2;
        int K = n/2;
        PRIMALenv_t env = NULL;
        PRIMALtask_t t = NULL;
        PRIMAL_makeenv(&env, NULL);
        PRIMAL_maketask(env, 0, 0, &t);
        PRIMAL_appendvars(t, n+K);
        for (int j=0; j<n; j++)
            PRIMAL_putvarbound(t,j,PRIMAL_BK_FR,-INFINITY,INFINITY);
        for (int k=0; k<K; k++) {
            PRIMAL_putvarbound(t,n+k,PRIMAL_BK_LO,0,INFINITY);
            PRIMAL_putcj(t,n+k,variant ? 1.0+0.05*(k%7) : 1.0);
            int mem[3] = {n+k,2*k,2*k+1};
            if (variant == 2) {
                mem[1]=k; mem[2]=K+k;
                PRIMAL_putvarbound(t,k,PRIMAL_BK_FX,0.5,0.5);
            }
            PRIMAL_appendcone(t,variant==2 ? PRIMAL_CT_RQUAD : PRIMAL_CT_QUAD,0,3,mem);
        }
        int rows = variant==1 ? 2 : 1;
        PRIMAL_appendcons(t,rows);
        for (int r=0; r<rows; r++) {
            for (int j=0; j<n; j++) {
                int use = variant==0 || (variant==1 && j%2==r) || (variant==2 && j>=K);
                if (use) PRIMAL_putaij(t,r,j,1.0);
            }
            double b = variant==1 ? (r==0 ? 0.6 : 0.4) : 1.0;
            PRIMAL_putconbound(t,r,PRIMAL_BK_FX,b,b);
        }
        double start=now();
        int rc=PRIMAL_optimize(t);
        double elapsed=now()-start, obj=0,dobj=0,pri=0,dual=0;
        PRIMALsolstae status=PRIMAL_SOL_STA_UNKNOWN;
        int valid = !rc && !PRIMAL_getsolsta(t,PRIMAL_SOL_ITR,&status)
            && !PRIMAL_getprimalobj(t,PRIMAL_SOL_ITR,&obj)
            && !PRIMAL_getdualobj(t,PRIMAL_SOL_ITR,&dobj)
            && !PRIMAL_getprimalinfeas(t,PRIMAL_SOL_ITR,&pri)
            && !PRIMAL_getdualinfeas(t,PRIMAL_SOL_ITR,&dual);
        printf("{\"n\":%d,\"variant\":%d,\"seconds\":%.9g,\"rc\":%d,"
               "\"status\":%d,\"valid\":%d,\"objective\":%.17g,\"dual_objective\":%.17g,"
               "\"primal\":%.17g,\"dual\":%.17g,\"gap\":%.17g}\n",
               n,variant,elapsed,rc,(int)status,valid,obj,dobj,pri,dual,
               fabs(obj-dobj)/(1+fabs(obj)+fabs(dobj)));
        fflush(stdout);
        PRIMAL_deletetask(&t);
        PRIMAL_deleteenv(&env);
    }
    return 0;
}
