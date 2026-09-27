/* SPDX-License-Identifier: Apache-2.0
 * Deterministic original-model checks: binary knapsack enumeration and box QP.
 * This is a small random instance corpus, not file-format fuzzing. */
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "primal.h"
static uint32_t seed=137;
static int next(void) { seed=1664525*seed+1013904223; return (int)(seed>>16); }
#define OK(x) do { if((x)!=PRIMAL_RES_OK) { fprintf(stderr,"API failure at %d\n",__LINE__); return 1; } } while(0)
int main(void) {
    PRIMALenv_t e; OK(PRIMAL_makeenv(&e,NULL));
    for(int k=0;k<100;k++) {
        PRIMALtask_t t; OK(PRIMAL_maketask(e,1,8,&t));
        double c[8],w[8],x[8],cap=10+(next()%10),oracle=-INFINITY,p;
        int sub[8];
        OK(PRIMAL_putobjsense(t,PRIMAL_OPTIMIZE_MAXIMIZE));
        for(int j=0;j<8;j++) {
            sub[j]=j;c[j]=1+next()%15;w[j]=1+next()%9;
            OK(PRIMAL_putcj(t,j,c[j]));OK(PRIMAL_putvarbound(t,j,PRIMAL_BK_RA,0,1));
            OK(PRIMAL_putvartype(t,j,PRIMAL_VAR_TYPE_INT_BIN));
        }
        OK(PRIMAL_putarow(t,0,8,sub,w));OK(PRIMAL_putconbound(t,0,PRIMAL_BK_UP,0,cap));
        for(int mask=0;mask<256;mask++) {
            double cost=0,weight=0;
            for(int j=0;j<8;j++) if(mask&(1<<j)) { cost+=c[j];weight+=w[j]; }
            if(weight<=cap && cost>oracle) oracle=cost;
        }
        OK(PRIMAL_optimize(t));OK(PRIMAL_getxx(t,PRIMAL_SOL_ITR,x));OK(PRIMAL_getprimalobj(t,PRIMAL_SOL_ITR,&p));
        double obj=0,weight=0;
        for(int j=0;j<8;j++) { if(!isfinite(x[j])||fabs(x[j]-round(x[j]))>1e-6||x[j]<-1e-6||x[j]>1+1e-6) return 1; obj+=c[j]*x[j];weight+=w[j]*x[j]; }
        if(weight>cap+1e-6||fabs(obj-oracle)>1e-6||fabs(p-obj)>1e-6) { fprintf(stderr,"knapsack case %d failed\n",k);return 1; }
        OK(PRIMAL_deletetask(&t));
        OK(PRIMAL_maketask(e,0,8,&t));oracle=0;
        for(int j=0;j<8;j++) {
            c[j]=(next()%21)-10; w[j]=1+next()%5;
            OK(PRIMAL_putcj(t,j,c[j]));OK(PRIMAL_putqobjij(t,j,j,w[j]));
            OK(PRIMAL_putvarbound(t,j,PRIMAL_BK_RA,0,2));
            double best=fmin(2,fmax(0,-c[j]/w[j]));oracle+=c[j]*best+0.5*w[j]*best*best;
        }
        OK(PRIMAL_optimize(t));OK(PRIMAL_getxx(t,PRIMAL_SOL_ITR,x));OK(PRIMAL_getprimalobj(t,PRIMAL_SOL_ITR,&p));obj=0;
        for(int j=0;j<8;j++) { if(!isfinite(x[j])||x[j]<-1e-6||x[j]>2+1e-6) return 1;obj+=c[j]*x[j]+0.5*w[j]*x[j]*x[j]; }
        if(fabs(obj-oracle)>1e-5||fabs(p-obj)>1e-6) { fprintf(stderr,"QP case %d failed\n",k);return 1; }
        OK(PRIMAL_deletetask(&t));
    }
    OK(PRIMAL_deleteenv(&e));puts("200 deterministic instances passed (100 enumerated MILP, 100 analytic QP)");return 0;
}
