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

/* simplex.h - two-phase primal simplex on the standard form
 *   min c'x  s.t.  A x = b, x >= 0, b >= 0
 * statuses: 0 optimal, 1 infeasible (Farkas witness in dray),
 *           2 unbounded (recession direction in pray),
 *           3 max iterations, 4 out of memory.
 * dray (m) and pray (n) may be NULL and are written only by the status that
 * owns them, so on any other status they hold nothing the caller may read.
 * dray is unnormalized (y'A <= 0, y'b > 0); pray is normalized to 1 in the
 * entering column (rho >= 0, A rho = 0, c'rho < 0). */
#ifndef SIMPLEX_H
#define SIMPLEX_H

int simplex_solve_std(const double *A, int m, int n,
                      const double *b, const double *c,
                      int max_iter, double *x /*n*/, double *y /*m*/,
                      double *dray /*m, status 1*/, double *pray /*n, status 2*/);
/* Come sopra, ma restituisce anche la base finale (`basis_out`, m indici di
 * colonna) e il tableau (`tab_out`, m righe di stride n+m+1: coefficienti
 * ridotti nelle prime n+m colonne, RHS in [n+m]). Servono ai tagli di Gomory.
 * Entrambi opzionali (NULL). */
int simplex_solve_std_tab(const double *A, int m, int n,
                          const double *b, const double *c,
                          int max_iter, double *x, double *y,
                          double *dray, double *pray,
                          int *basis_out, double *tab_out);

/* Dual simplex: min c'x s.t. A x = b, x >= 0, da una base DUAL-ammissibile
 * (`basis[m]`, indici di colonna); `b` puo' avere componenti negative. Costruisce
 * B^-1 con Gauss-Jordan su [B|I], sceglie la riga con RHS piu' negativo e fa il
 * ratio test sui costi ridotti. Status come sopra (0 ottimo, 1 inammissibile,
 * 2 illimitato, 3 max iter, 4 memoria). x[n]. Portato da gmbortools
 * `gor_lp_dual_simplex` (a sua volta da apurvasijaria/Operation_Research_Lab). */
int simplex_dual_solve_std(const double *A, int m, int n,
                           const double *b, const double *c,
                           const int *basis, int max_iter, double *x /*n*/,
                           int *basis_out /*m, base finale, opzionale*/,
                           double *y /*m, duali cB^T B^-1 via LU, opzionale*/);

/* Revised simplex: min c'x s.t. A x = b, x >= 0, da una base PRIMAL-ammissibile
 * (`basis[m]`). Mantiene B^-1 con aggiornamento eta (una fattorizzazione per
 * iterazione, non un tableau denso). Status come sopra; 1 = base primal
 * inammissibile. x[n]. Portato da gmbortools `gor_lp_revised_simplex`
 * (da athityakumar/or_lab). */
int simplex_revised_solve_std(const double *A, int m, int n,
                              const double *b, const double *c,
                              const int *basis, int max_iter, double *x /*n*/,
                              double *y /*m, duali, opzionale (NULL ok)*/);

/* Costi ridotti c - A'y (n). Portato da gmbortools `gor_lp_reduced_costs`. */
void simplex_reduced_costs(const double *A, int m, int n, const double *c,
                           const double *y, double *red /*n*/);

/* Crash basis: m colonne di A linearmente indipendenti (eliminazione di Gauss
 * con pivoting di colonna). Ritorna 1 e riempie `basis[m]` (indici di colonna)
 * se rank(A) == m, 0 altrimenti. Serve a dare al simplesso revised una base
 * valida senza fase 1 (concetto di MSK_IPAR_SIM_PRIMAL_CRASH). */
int simplex_crash_basis(const double *A, int m, int n, int *basis /*m*/);

#endif /* SIMPLEX_H */
