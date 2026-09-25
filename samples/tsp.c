/*
 * PrimalSolver - a convex optimization solver in C99 (LP/QP/SOCP/SDP/exp-power/MIP).
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 * Licensed under the Apache License, Version 2.0 ... (see LICENSE).
 */

/* tsp.c - a directed TSP on four cities by iterative subtour elimination
 * (MOSEK Cookbook 11.8, tsp.cc).
 *
 * Binary x[i][j] = 1 iff the tour goes i -> j.  The base model is the
 * assignment relaxation
 *     minimize   sum_ij C_ij x_ij
 *     s.t.       sum_j x_ij = 1,  sum_i x_ij = 1     (one out, one in)
 *                x_ij <= A_ij                        (only the listed arcs)
 *                x_ii = 0                            (no 1-hop loops)
 * (and, optionally, x_ij + x_ji <= 1 for no 2-hop loops; without it the
 * first cover is a set of disjoint cycles and subtour elimination is needed).  A tour that is a cover
 * of disjoint cycles is then repaired by adding, for every cycle found,
 *     sum_{i->j in the cycle} x_ij <= (cycle size) - 1,
 * and re-solving until one Hamiltonian cycle remains.
 *
 * Instance: four cities with the Cookbook's arc set and costs
 *     (0,1)=1  (1,2)=1  (2,3)=1  (3,0)=1  (the unit square)
 *     (1,0)=.1 (0,2)=.1 (2,1)=.1 (0,3)=.1 (the cheap arcs)
 * The only Hamiltonian cycle is 0->1->2->3->0, cost 4 (the cheap arcs form the
 * 3-cycle 1->0->2->1, which subtour elimination removes).
 *
 * Usage: tsp   (no arguments)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "primal.h"

#define N 4
#define IDX(i, j) ((i) * N + (j))

/* arcs present in A and their cost (absent arcs have A_ij = 0) */
static const int AI[8] = {0, 1, 2, 3, 1, 0, 2, 0};
static const int AJ[8] = {1, 2, 3, 0, 0, 2, 1, 3};
static const double CV[8] = {1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1};

/* one solve of the model built so far; returns the successor of each city. */
static void solve(const PRIMALtask_t t, int succ[N]) {
    PRIMAL_optimize(t);
    double x[N * N] = {0};
    PRIMAL_getxx(t, PRIMAL_SOL_ITR, x);
    for (int i = 0; i < N; i++) {
        succ[i] = -1;
        for (int j = 0; j < N; j++) if (x[IDX(i, j)] > 0.5) succ[i] = j;
    }
}

int main(void) {
    const int two_hop = 0;   /* without 2-hop removal the first cover is disjoint cycles */
    PRIMALenv_t env; PRIMALtask_t t;
    PRIMAL_makeenv(&env, NULL); PRIMAL_maketask(env, 0, 0, &t);
    PRIMAL_appendvars(t, N * N);
    PRIMAL_appendcons(t, 2 * N + (two_hop ? N : 0));
    PRIMAL_putobjsense(t, PRIMAL_OPTIMIZE_MINIMIZE);
    double C[N * N] = {0};
    for (int e = 0; e < 8; e++) C[IDX(AI[e], AJ[e])] = CV[e];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double a = C[IDX(i, j)] > 0.0 ? 1.0 : 0.0;
            PRIMAL_putvarbound(t, IDX(i, j), PRIMAL_BK_RA, 0.0, a);   /* x_ij <= A_ij */
            PRIMAL_putvartype(t, IDX(i, j), PRIMAL_VAR_TYPE_INT_BIN);
            PRIMAL_putcj(t, IDX(i, j), C[IDX(i, j)]);
        }
    int row = 0;
    for (int i = 0; i < N; i++) {                       /* one arc out of each city */
        int sub[N]; double v[N];
        for (int j = 0; j < N; j++) { sub[j] = IDX(i, j); v[j] = 1.0; }
        PRIMAL_putarow(t, row, N, sub, v);
        PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 1.0, 1.0); row++;
    }
    for (int j = 0; j < N; j++) {                       /* one arc into each city */
        int sub[N]; double v[N];
        for (int i = 0; i < N; i++) { sub[i] = IDX(i, j); v[i] = 1.0; }
        PRIMAL_putarow(t, row, N, sub, v);
        PRIMAL_putconbound(t, row, PRIMAL_BK_FX, 1.0, 1.0); row++;
    }
    for (int i = 0; i < N; i++) {                       /* no 2-hop loops */
        PRIMAL_putarow(t, row, 2, (int[]){IDX(i, (i + 1) % N), IDX((i + 1) % N, i)},
                       (double[]){1.0, 1.0});
        PRIMAL_putconbound(t, row, PRIMAL_BK_UP, -INFINITY, 1.0); row++;
    }

    /* iterative subtour elimination */
    int it = 0, succ[N];
    for (;;) {
        solve(t, succ);
        it++;
        /* walk the cycle containing city 0, count how many cities it covers */
        int cycle[N], len = 0, cur = 0;
        do { cycle[len++] = cur; cur = succ[cur]; } while (cur != 0 && len < N);
        if (len == N && cur == 0) break;                /* one Hamiltonian cycle */
        /* add sum_{i in cycle} x_{i,succ[i]} <= len - 1 */
        int sub[N]; double v[N];
        for (int k = 0; k < len; k++) { sub[k] = IDX(cycle[k], succ[cycle[k]]); v[k] = 1.0; }
        PRIMAL_appendcons(t, 1);
        int last = 0;
        { PRIMAL_getnumcon(t, &last); }
        PRIMAL_putarow(t, last - 1, len, sub, v);
        PRIMAL_putconbound(t, last - 1, PRIMAL_BK_UP, -INFINITY, (double)(len - 1));
    }

    double po = 0.0;
    PRIMAL_getprimalobj(t, PRIMAL_SOL_ITR, &po);
    /* the tour is 0 -> 1 -> 2 -> 3 -> 0 */
    int good = fabs(po - 4.0) < 1e-6 && succ[0] == 1 && succ[1] == 2 &&
               succ[2] == 3 && succ[3] == 0 && it >= 2;   /* subtours were eliminated */
    printf("tsp: obj=%.6f  tour 0->%d->%d->%d->%d  (atteso 4, 0->1->2->3->0)  iterazioni=%d  %s\n",
           po, succ[0], succ[1], succ[2], succ[3], it, good ? "OK" : "FAIL");
    PRIMAL_deletetask(&t); PRIMAL_deleteenv(&env);
    return good ? 0 : 1;
}
