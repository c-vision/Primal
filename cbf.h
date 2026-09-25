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

/* cbf.h - Conic Benchmark Format (CBF v4) I/O
 * Sottoinsieme supportato:
 *   VER (1..4), OBJSENSE, VAR (F,L+,L-,L=,Q,QR,EXP,EXP*,@k:POW con alpha len 2),
 *   INT, PSDVAR, PSDCON, CON, POWCONES,
 *   OBJACOORD, OBJFCOORD, OBJBCOORD, ACOORD, FCOORD, BCOORD, HCOORD, DCOORD.
 * CHANGE termina il file (hotstart-sequences non supportate).
 * Non supportati (errore): L-, ONENORM, INFNORM, SVECPSD, GMEAN*, GMEANABS*,
 *   POWH, POW*, @k:POW*, POWCONES con vettori alpha di lunghezza != 2.
 * Deviazione documentata: un vincolo LMI (PSDCON) viene ricostruito come
 * variabile bar + vincoli di uguaglianza elemento-per-elemento. */
#ifndef CBF_H
#define CBF_H
#include <stdio.h>
#include "primal.h"

PRIMALrescodee cbf_write(PRIMALtask_t t, FILE *f);
PRIMALrescodee cbf_read(PRIMALtask_t t, FILE *f);

#endif