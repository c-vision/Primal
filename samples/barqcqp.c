/*
 * PrimalSolver - a convex optimization solver in C99 (LP/QP/SOCP/SDP/exp-power/MIP).
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License.  A copy of the License
 * is in the repository root (LICENSE) and at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

/* barqcqp.c — barre simmetriche e parte quadratica (vincolo e obiettivo) nello
 * stesso modello, risolti dalla strada QCQP→conica attraverso la sola API
 * pubblica; e il dominio non convesso, che deve essere rifiutato.
 *
 * I valori ottimi sono derivati a mano qui sotto (nessun numero copiato da un
 * altro solver). A e B sono le due forme che la via barre perdeva in silenzio
 * (T86); C e' la forma che l'encoder QCQP tagliava in silenzio (T87).
 *
 * A) min -x0 - x1 - 4 <E00,B>
 *    s.t. x0^2 + x1^2 + <E00,B> <= 3,  <I,B> = 1,  B in S^2_+,  x >= 0
 *    Con tr(B)=1 e B psd si ha b = B00 in [0,1]; per un b fissato la sfera e'
 *    attiva in x0=x1=sqrt((3-b)/2) e resta da massimizzare
 *    g(b) = 2 sqrt((3-b)/2) + 4b, con g'(b) = 4 - 1/(2 sqrt((3-b)/2)) >= 3.5
 *    su tutto [0,1]. Quindi b*=1 e s*=1, e
 *      x* = (1,1),  B = diag(1,0) (B00=1 e tr(B)=1 forzano B11=0 e, per
 *      det(B)>=0, B01=0),  obj = -(2 + 4) = -6.
 *
 * B) min x0^2 + x1^2 - 2 x0 - 2 x1 - 3 <E00,B>
 *    s.t. <I,B> = 1,  B in S^2_+,  x >= 0
 *    Le due parti sono separate: la quadratica e' (x0-1)^2 + (x1-1)^2 - 2,
 *    minima in x*=(1,1) con valore -2; il termine di obiettivo sulla barra vale -3 b con
 *    b = B00 in [0,1], minimo a b=1, cioe' di nuovo B = diag(1,0).
 *      obj = -2 - 3 = -5.
 *    Qui e' il termine quadratico dell'obiettivo che deve spostare la risposta:
 *    se la via barre lo scarta resta -x0-x1 su x >= 0, cioe' un modello
 *    illimitato. Il numero checkato sotto e' quindi un certificato
 *    dell'encoding intero, non un dettaglio.
 *
 * C) come A ma con il vincolo x0^2 - 2 x1^2 + <E00,B> <= 3 e 0 <= x <= 2.
 *    La matrice quadratica diag(2,-4) ha un autovalore dal lato sbagliato: il
 *    dominio non e' convesso e il cono non puo' rappresentarlo. Un encoder che
 *    taglia lascia la sola x0^2 + <E00,B> <= 3 e risponde un numero sul
 *    modello di qualcun altro. Qui si misura il rifiuto: rc = ERR_ARG, e i
 *    getter soluzione dicono "no" con lo stesso codice.
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

/* E00 e l'identita' 2x2, condivisi dai tre casi */
static void put_syms(PRIMALtask_t t, int *m00, int *mI) {
    PRIMAL_appendsparsesymmat(t, 2, 1, (int[]){0}, (int[]){0}, (double[]){1.0}, m00);
    PRIMAL_appendsparsesymmat(t, 2, 2, (int[]){0, 1}, (int[]){0, 1},
                              (double[]){1.0, 1.0}, mI);
}

int main(void) {
    PRIMALenv_t env;
    PRIMAL_makeenv(&env, NULL);
    int pass = 1;

    /* ---- A) riga quadratica e barra nella stessa riga, barra anche in obiettivo ---- */
    {
        PRIMALtask_t t;
        PRIMAL_maketask(env, 0, 0, &t);
        int m00, mI;
        PRIMAL_appendcons(t, 2);
        put_syms(t, &m00, &mI);
        int dim = 2;
        PRIMAL_appendbarvars(t, 1, &dim);
        PRIMAL_appendvars(t, 2);
        PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
        for (int j = 0; j < 2; j++)
            PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putcj(t, 0, -1.0);
        PRIMAL_putcj(t, 1, -1.0);
        double bobj = -4.0;
        PRIMAL_putbarcj(t, 0, 1, &m00, &bobj);
        PRIMAL_putconbound(t, 0, PRIMAL_BK_UP, -INFINITY, 3.0);
        PRIMAL_putqconk(t, 0, 2, (int[]){0, 1}, (int[]){0, 1}, (double[]){2.0, 2.0});
        PRIMAL_putbaraij(t, 0, 0, 1, &m00, (double[]){1.0});
        PRIMAL_putbaraij(t, 1, 0, 1, &mI, (double[]){1.0});
        PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 1.0, 1.0);

        double x[2] = {0, 0}, B[4] = {0, 0, 0, 0}, po = 0.0;
        int ok = (PRIMAL_optimize(t) == PRIMAL_RES_OK) &&
                 (PRIMAL_getxx(t, PRIMAL_SOL_ITR, x) == PRIMAL_RES_OK) &&
                 (PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &po) == PRIMAL_RES_OK) &&
                 (PRIMAL_getbarxj(t, PRIMAL_SOL_ITR, 0, B) == PRIMAL_RES_OK);
        ok = ok && fabs(x[0] - 1.0) < 1e-4 && fabs(x[1] - 1.0) < 1e-4 &&
             fabs(po + 6.0) < 1e-5 &&
             fabs(B[0] - 1.0) < 1e-5 && fabs(B[1]) < 1e-5 && fabs(B[3]) < 1e-5;
        printf("barqcqp A  obj=%.6f (atteso -6)  x=(%.6f, %.6f)  "
               "B=[[%.4f,%.4f],[%.4f,%.4f]]  %s\n",
               po, x[0], x[1], B[0], B[1], B[2], B[3], ok ? "OK" : "FAIL");
        pass &= ok;
        PRIMAL_deletetask(&t);
    }

    /* ---- B) obiettivo quadratico convesso + barra ---- */
    {
        PRIMALtask_t t;
        PRIMAL_maketask(env, 0, 0, &t);
        int m00, mI;
        PRIMAL_appendcons(t, 1);
        put_syms(t, &m00, &mI);
        int dim = 2;
        PRIMAL_appendbarvars(t, 1, &dim);
        PRIMAL_appendvars(t, 2);
        PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
        for (int j = 0; j < 2; j++)
            PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
        PRIMAL_putcj(t, 0, -2.0);
        PRIMAL_putcj(t, 1, -2.0);
        PRIMAL_putqobj(t, 2, (int[]){0, 1}, (int[]){0, 1}, (double[]){2.0, 2.0});
        double bobj = -3.0;
        PRIMAL_putbarcj(t, 0, 1, &m00, &bobj);
        PRIMAL_putbaraij(t, 0, 0, 1, &mI, (double[]){1.0});
        PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);

        double x[2] = {0, 0}, B[4] = {0, 0, 0, 0}, po = 0.0;
        int ok = (PRIMAL_optimize(t) == PRIMAL_RES_OK) &&
                 (PRIMAL_getxx(t, PRIMAL_SOL_ITR, x) == PRIMAL_RES_OK) &&
                 (PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &po) == PRIMAL_RES_OK) &&
                 (PRIMAL_getbarxj(t, PRIMAL_SOL_ITR, 0, B) == PRIMAL_RES_OK);
        ok = ok && fabs(x[0] - 1.0) < 1e-4 && fabs(x[1] - 1.0) < 1e-4 &&
             fabs(po + 5.0) < 1e-5 &&
             fabs(B[0] - 1.0) < 1e-5 && fabs(B[1]) < 1e-5 && fabs(B[3]) < 1e-5;
        printf("barqcqp B  obj=%.6f (atteso -5)  x=(%.6f, %.6f)  "
               "B=[[%.4f,%.4f],[%.4f,%.4f]]  %s\n",
               po, x[0], x[1], B[0], B[1], B[2], B[3], ok ? "OK" : "FAIL");
        pass &= ok;
        PRIMAL_deletetask(&t);
    }

    /* ---- C) dominio non convesso: rifiuto, non risposta ---- */
    {
        PRIMALtask_t t;
        PRIMAL_maketask(env, 0, 0, &t);
        int m00, mI;
        PRIMAL_appendcons(t, 2);
        put_syms(t, &m00, &mI);
        int dim = 2;
        PRIMAL_appendbarvars(t, 1, &dim);
        PRIMAL_appendvars(t, 2);
        PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
        for (int j = 0; j < 2; j++)
            PRIMAL_putvarbound(t, j, PRIMAL_BK_RA, 0.0, 2.0);
        PRIMAL_putcj(t, 0, -1.0);
        PRIMAL_putcj(t, 1, -1.0);
        PRIMAL_putconbound(t, 0, PRIMAL_BK_UP, -INFINITY, 3.0);
        PRIMAL_putqconk(t, 0, 2, (int[]){0, 1}, (int[]){0, 1}, (double[]){2.0, -4.0});
        PRIMAL_putbaraij(t, 0, 0, 1, &m00, (double[]){1.0});
        PRIMAL_putbaraij(t, 1, 0, 1, &mI, (double[]){1.0});
        PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 1.0, 1.0);

        double x[2] = {0, 0}, po = 0.0;
        int rc = (int)PRIMAL_optimize(t);
        int ok = (rc == (int)PRIMAL_RES_ERR_ARG) &&
                 (PRIMAL_getxx(t, PRIMAL_SOL_ITR, x) == (int)PRIMAL_RES_ERR_ARG) &&
                 (PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &po) == (int)PRIMAL_RES_ERR_ARG);
        printf("barqcqp C  rc=%d (atteso %d = ERR_ARG)  obj non pubblicato  %s\n",
               rc, (int)PRIMAL_RES_ERR_ARG, ok ? "OK" : "FAIL");
        pass &= ok;
        PRIMAL_deletetask(&t);
    }

    PRIMAL_deleteenv(&env);
    return pass ? 0 : 1;
}
