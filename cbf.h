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
 * Supported subset:
 *   VER (1..4), OBJSENSE, VAR (F,L+,L-,L=,Q,QR,EXP,EXP*,@k:POW with alpha len 2),
 *   INT, PSDVAR, PSDCON, CON, POWCONES,
 *   OBJACOORD, OBJFCOORD, OBJBCOORD, ACOORD, FCOORD, BCOORD, HCOORD, DCOORD.
 * CHANGE terminates the file (hotstart sequences are not supported).
 * Not supported (error): L-, ONENORM, INFNORM, SVECPSD, GMEAN*, GMEANABS*,
 *   POWH, POW*, @k:POW*, POWCONES with alpha vectors of length != 2.
 * Documented deviation: an LMI constraint (PSDCON) is rebuilt as a bar
 * variable plus element-by-element equality constraints. */
#ifndef CBF_H
#define CBF_H
#include <stdio.h>
#include "primal.h"

/* Write the task in CBF v4; RPOW cones make the writer answer ERR_ARG.
 * Returns PRIMAL_RES_OK, ERR_NULL, ERR_ALLOC or ERR_ARG. */
PRIMALrescodee cbf_write(PRIMALtask_t t, FILE *f);
/* Read a CBF v4 file into an EMPTY task (numvar==0 && numcon==0).
 * Returns PRIMAL_RES_OK, ERR_NULL, ERR_FILE, ERR_ARG or ERR_ALLOC. */
PRIMALrescodee cbf_read(PRIMALtask_t t, FILE *f);

#endif