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

/* expcone.c - see expcone.h. */
/* Compile the shared SDP implementation with the PEXP corrector.  Keeping
 * this in its own TU preserves optimization of the original engine. */
#define SDP_IPM_CORRECTED
#define sdp_ipm_csr sdp_ipm_csr_corrected
#define sdp_ipm sdp_ipm_corrected
#include "expcone_corrector.h"
#include "sdp.c"
