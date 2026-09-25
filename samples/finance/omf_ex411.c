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

/* omf_ex411.c - Exercise 4.11 of "Optimization Methods in Finance"
 * (Cornuejols & Tutuncu, 2007), Section 4.3 "Arbitrage in the currency market".
 *
 * Given the exchange rates traded on February 14, 2002 (units of the "into"
 * currency per unit of the "from" currency):
 *
 *            into:   Dollar   Euro    Pound    Yen
 *   from Dollar          -   1.1486   0.7003   133.38
 *   from Euro       0.8706        -   1.6401   116.12
 *   from Pound      1.4279   0.6097        -   190.45
 *   from Yen        0.00750  0.00861  0.00525        -
 *
 * an arbitrage is a sequence of conversions that turns one unit of a currency
 * into more than one. The book formulates an LP over the 12 conversion
 * quantities (DE, DP, DY, ED, EP, EY, PD, PE, PY, YD, YE, YP) plus the
 * arbitrage amount D, with one balance equation per currency:
 *
 *   max D
 *   s.t. Dollar: D + DE + DP + DY - 0.8706 ED - 1.4279 PD - 0.00750 YD = 1
 *        Euro:   ED + EP + EY - 1.1486 DE - 1.6401 PE - 0.00861 YE      = 0
 *        Pound:  PD + PE + PY - 0.7003 DP - 0.6097 EP - 0.00525 YP      = 0
 *        Yen:    YD + YE + YP - 133.38 DY - 116.12 EY - 190.45 PY      = 0
 *        0 <= variables, D <= 10000.
 *
 * The book reports that the dollar -> euro -> yen -> dollar cycle makes about
 * $0.0003 per dollar (1.1486 * 116.12 * 0.00750 = 1.000316) while the reverse
 * order loses about $0.0002 (133.38 * 0.00861 * 0.8706 = 0.99979). Solving the
 * LP drives D to its cap of 10000 (any positive arbitrage does), converting
 * about $34 million into euros in the book's particular solution; the book
 * notes there are other optimal solutions, and the opportunity is so small
 * that some solvers miss it on numerical grounds. Indeed the direct
 * dollar -> yen -> dollar cycle is also an arbitrage (133.38 * 0.00750 =
 * 1.00035), and this solver finds that one instead.
 *
 * The sample checks that the two cycle products have the stated signs and
 * magnitude and that the LP attains the arbitrage cap.
 *
 * Usage: omf_ex411   (no arguments)
 */
#include <stdio.h>
#include <math.h>
#include "primal.h"

#define NV 13
enum { DE, DP, DY, ED, EP, EY, PD, PE, PY, YD, YE, YP, D_ };

int main(void) {
    /* cycle products: dollar->euro->yen->dollar and the reverse */
    double c_dey = 1.1486 * 116.12 * 0.00750;   /* > 1 : arbitrage */
    double c_dye = 133.38 * 0.00861 * 0.8706;   /* < 1 : loss */
    int cycles_ok = c_dey > 1.0 && c_dye < 1.0 &&
                    fabs((c_dey - 1.0) - 0.0003) < 5e-5 &&
                    fabs((1.0 - c_dye) - 0.0002) < 5e-5;

    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL);
    PRIMAL_maketask(env, 4, NV, &t);
    for (int j = 0; j < NV; j++) PRIMAL_putvarbound(t, j, PRIMAL_BK_LO, 0.0, INFINITY);
    PRIMAL_putvarbound(t, D_, PRIMAL_BK_UP, 0.0, 10000.0);
    PRIMAL_putcj(t, D_, -1.0);                       /* max D */

    {   /* Dollar */
        int i[] = {D_, DE, DP, DY, ED, PD, YD};
        double v[] = {1.0, 1.0, 1.0, 1.0, -0.8706, -1.4279, -0.00750};
        PRIMAL_putarow(t, 0, 7, i, v); PRIMAL_putconbound(t, 0, PRIMAL_BK_FX, 1.0, 1.0);
    }
    {   /* Euro */
        int i[] = {ED, EP, EY, DE, PE, YE};
        double v[] = {1.0, 1.0, 1.0, -1.1486, -1.6401, -0.00861};
        PRIMAL_putarow(t, 1, 6, i, v); PRIMAL_putconbound(t, 1, PRIMAL_BK_FX, 0.0, 0.0);
    }
    {   /* Pound */
        int i[] = {PD, PE, PY, DP, EP, YP};
        double v[] = {1.0, 1.0, 1.0, -0.7003, -0.6097, -0.00525};
        PRIMAL_putarow(t, 2, 6, i, v); PRIMAL_putconbound(t, 2, PRIMAL_BK_FX, 0.0, 0.0);
    }
    {   /* Yen */
        int i[] = {YD, YE, YP, DY, EY, PY};
        double v[] = {1.0, 1.0, 1.0, -133.38, -116.12, -190.45};
        PRIMAL_putarow(t, 3, 6, i, v); PRIMAL_putconbound(t, 3, PRIMAL_BK_FX, 0.0, 0.0);
    }

    PRIMALrescodee rc = PRIMAL_optimize(t);
    int ok = (rc == PRIMAL_RES_OK);
    if (ok) {
        double z, x[NV];
        PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &z);
        PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
        double d = -z;
        ok = cycles_ok && fabs(d - 10000.0) < 1.0;
        printf("omf_ex411  arbitrage D=%.4f (cap 10000)  %s\n", d, ok ? "OK" : "FAIL");
        printf("  cycle $->E->Yen->$ = %.6f (>1, +%.4f/dollar)  "
               "$->Yen->E->$ = %.6f (<1, %.4f/dollar)  %s\n",
               c_dey, c_dey - 1.0, c_dye, c_dye - 1.0, cycles_ok ? "OK" : "FAIL");
        printf("  flows (one of several optimal cycles): DE=%.6g DP=%.4g DY=%.4g "
               "ED=%.4g YD=%.4g\n", x[DE], x[DP], x[DY], x[ED], x[YD]);
    } else {
        printf("omf_ex411  rc=%d FAIL\n", (int)rc);
    }
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return ok ? 0 : 1;
}
