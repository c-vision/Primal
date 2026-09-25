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

/* mpsio.h - file I/O: free-format MPS reader/writer + CPLEX LP reader/writer.
 * Dispatch on filename extension (.mps / .lp, case-insensitive; default MPS).
 */
#ifndef MPSIO_H
#define MPSIO_H

#include "primal.h"

/* Write task to file. .lp -> LP format, anything else -> free MPS.
 * Cones are written in MPS (CSECTION); LP writes only the linear part and
 * refuses a quadratic model (no cone section there). */
PRIMALrescodee primalio_write(PRIMALtask_t t, const char *filename);

/* Read file into an EMPTY task (numvar==0 && numcon==0, else ERR_ARG). */
PRIMALrescodee primalio_read(PRIMALtask_t t, const char *filename);

/* Read file with a declared data format (PRIMALdataformate). format 0
 * dispatches by extension (same as primalio_read); 1/4 force MPS, 2 forces LP,
 * 3 forces OPF, 7 forces CBF; 5/6/8 (task dump, PTF, JSON) are not read. */
PRIMALrescodee primalio_read_format(PRIMALtask_t t, const char *filename, int format);

/* OPF format (supported file format 16.3). opf_read parses OPF text into an
 * EMPTY task; opf_write emits the OPF problem (objective/constraints/bounds/
 * cones/integers). Both return ERR_ARG on unsupported constructs. */
PRIMALrescodee opf_read(PRIMALtask_t t, const char *data);
PRIMALrescodee opf_write(PRIMALtask_t t, FILE *f);

/* Internal hook implemented in primal.c: fire a generic callback code. */
void primal_cb_notify(PRIMALtask_t t, int code);

#endif
