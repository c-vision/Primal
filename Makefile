# PrimalSolver - a convex optimization solver in C99 (LP/QP/SOCP/SDP/exp-power/MIP).
# Copyright 2026 Gaetano Minardi
# SPDX-License-Identifier: Apache-2.0
# 
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  A copy of the License
# is in the repository root (LICENSE) and at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations
# under the License.

# Makefile - PrimalSolver
# All build artifacts (objects, binaries, compiled samples) go into out/,
# which is gitignored. Never write binaries into the repo root or samples/.
CC      = gcc
CFLAGS  = -std=c99 -Wall -Wextra -pedantic -O2
LDLIBS  = -lm

OUT     = out
HEADERS = primal.h linalg.h stdform.h simplex.h ipm.h socp.h sdp.h expcone.h \
          mpsio.h cbf.h scaling.h presolve.h
OBJS    = $(addprefix $(OUT)/,linalg.o stdform.o simplex.o ipm.o socp.o sdp.o \
          expcone.o mpsio.o cbf.o scaling.o presolve.o primal.o)

all: $(OUT)/example_lp $(OUT)/run_tests

lib: $(OBJS)

$(OUT):
	mkdir -p $(OUT) $(OUT)/samples

$(OUT)/%.o: %.c $(HEADERS) | $(OUT)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUT)/example_lp: example_lp.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ example_lp.c $(OBJS) $(LDLIBS)

$(OUT)/run_tests: test_primal.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ test_primal.c $(OBJS) $(LDLIBS)

test: $(OUT)/run_tests
	./$(OUT)/run_tests

example: $(OUT)/example_lp
	./$(OUT)/example_lp

# ---- sanitizer (ASan+UBSan): objects and binaries under out/san/ ----
SANFLAGS = -std=c99 -Wall -Wextra -pedantic -O2 -g -fno-omit-frame-pointer
SAN      = -fsanitize=address,undefined
SANOBJS  = $(addprefix $(OUT)/san/,linalg.o stdform.o simplex.o ipm.o socp.o \
           sdp.o expcone.o mpsio.o cbf.o scaling.o presolve.o primal.o)
SAN_SAMPLES = samples/logistic_large.c samples/finance/market_impact.c \
           samples/cvx_regression.c samples/maxcut_sdp.c samples/socp_robust.c \
           samples/lp_large.c samples/finance/portfolio_mgmt.c \
           samples/mosek_comparison/logistic.c samples/mosek_comparison/sdo2.c \
           samples/mosek_comparison/qcqo1.c

$(OUT)/san:
	mkdir -p $(OUT)/san $(OUT)/san/samples

$(OUT)/san/%.o: %.c $(HEADERS) | $(OUT)/san
	$(CC) $(SANFLAGS) $(SAN) -c $< -o $@

$(OUT)/san/run_tests: test_primal.c $(SANOBJS) | $(OUT)/san
	$(CC) $(SANFLAGS) $(SAN) -o $@ test_primal.c $(SANOBJS) $(LDLIBS)

# The leak check does not exist on macOS ("LeakSanitizer is not supported"):
# ASAN_OPTIONS=detect_leaks=0 is required or the process aborts at exit.
sanitize: $(OUT)/san/run_tests
	ASAN_OPTIONS=detect_leaks=0 ./$(OUT)/san/run_tests

sanitize-samples: $(SANOBJS) | $(OUT)/san
	@for f in $(SAN_SAMPLES); do \
	  b=$$(basename $$f .c); \
	  $(CC) $(SANFLAGS) $(SAN) -I. -o $(OUT)/san/samples/$$b $$f $(SANOBJS) $(LDLIBS) || exit 1; \
	  ASAN_OPTIONS=detect_leaks=0 $(OUT)/san/samples/$$b || { echo "SANITIZE FAIL: $$b"; exit 1; }; \
	  echo "  sanitize ok: $$b"; \
	done

# ---- fuzz (deterministic randomized cross-validation) ----
$(OUT)/fuzz: bench/fuzz.c $(OBJS) | $(OUT)
	$(CC) $(CFLAGS) -I. -o $@ bench/fuzz.c $(OBJS) $(LDLIBS)

fuzz: $(OUT)/fuzz
	./$(OUT)/fuzz

$(OUT)/san/fuzz: bench/fuzz.c $(SANOBJS) | $(OUT)/san
	$(CC) $(SANFLAGS) $(SAN) -I. -o $@ bench/fuzz.c $(SANOBJS) $(LDLIBS)

sanitize-fuzz: $(OUT)/san/fuzz
	ASAN_OPTIONS=detect_leaks=0 ./$(OUT)/san/fuzz

clean:
	rm -rf $(OUT)

.PHONY: all lib test example clean sanitize sanitize-samples fuzz sanitize-fuzz

# ---- samples ----
# mosek_comparison/: ports of public MOSEK examples (matching optimal values)
# samples/ (root):   larger, more complex examples that stress scaling
MOSEK_SAMPLES = lo1 qo1 mi1 sched sdo1 ceo1 pow1 hello lo2 pinfeas mil1 milo1 acc1_max \
          mioinfeas1 portfolio_1 portfolio_2 sdo2 logistic response cqo1 \
          qcqo1 callback solvebasis sos1 sos2 mil2 portfolio_3 portfolio_5 \
          mioinitsol mico1 portfolio_4 portfolio_6 nearestcorrelation \
          sensitivity acc1 djc1 simple solvelinear reoptimization parameters \
          solutionquality feasrepairex1 gp1 sdo_lmi dual_sdo_l1 \
          sparsecholesky concurrent1 parallel
COMPLEX_SAMPLES = transport assignment knapsack facility portfolio_large \
          network_flow socp_robust maxcut_sdp logistic_large cvx_regression \
          lp_large lp_ineq_large qp_sparse barqcqp max_volume_cuboid \
          unit_commitment sinr_balancing truss_design max_flow_min_cut \
          sos_m1_certificate lyapunov_roa secure_ee_sdma logcontrast_lasso \
          quantum_separability lovasz_theta min_enclosing_ball \
          binary_quadratic exact_cover pwl_convex equilibrium dist_robust \
          facility_location wasserstein mle_density rank_one kmeans \
          filter_design f_sparc hard_uncertain diet nesting_dotted_board eco_driving_qp gate_sizing_gp lp_mincut poly_date limit_analysis_dp lmi_lyapunov transformer_design surface_cycles gp_toolbox min_circle welzl steering_robustness bounded_real_lmi primal_svm sudoku total_variation lownerjohn_ellipsoid tsp lpt qcqp_sdo_relaxation mpc_linear secure_beamforming
FINANCE_SAMPLES = markowitz_conic cvar_portfolio risk_parity market_impact \
          regression_ls regression_regularized portfolio_mgmt \
          transaction_cost lsq_pos lsq_l1_penalty risk_budgeting_ls sharpe_ratio sharpe_ratio_sectors factor_model robust_cvar risk_budgeting \
          option_pricing cardinality_portfolio mean_cvar multi_period index_tracking \
          omf_ex410 omf_ex217 omf_bb omf_qp_grg omf_ex2001 omf_ex25 omf_dedication omf_ex21 omf_ex312 omf_financing omf_mvo omf_bl omf_workforce omf_ex411 omf_ex118 \
          omf_capital omf_appendix_d omf_duality omf_ex207 omf_nearestcorr \
          big_portfolio evar_portfolio market_neutral
# books/: examples/exercises taken from the optimization books (AiMathWiki 09-Mosek)
BOOK_SAMPLES = intlo_simplex intlo_bigm intlo_transport hdb_kkt \
          boyd_lp_unique boyd_qp_box boyd_qp_box2 boyd_l1 boyd_detector \
          pca_alloc oa_feed npo_lad npo_minimax pca_cashflow \
          boyd_corr_sdp lmco_lovasz
SAMPLES = $(MOSEK_SAMPLES) $(COMPLEX_SAMPLES) $(FINANCE_SAMPLES) $(BOOK_SAMPLES)
LIBSRCS = linalg.c stdform.c simplex.c ipm.c socp.c sdp.c expcone.c mpsio.c cbf.c scaling.c presolve.c primal.c

samples: $(MOSEK_SAMPLES:%=$(OUT)/samples/%) $(COMPLEX_SAMPLES:%=$(OUT)/samples/%) $(FINANCE_SAMPLES:%=$(OUT)/samples/%) $(BOOK_SAMPLES:%=$(OUT)/samples/%)

$(MOSEK_SAMPLES:%=$(OUT)/samples/%): $(OUT)/samples/%: samples/mosek_comparison/%.c $(LIBSRCS) | $(OUT)
	$(CC) $(CFLAGS) -I. $< $(LIBSRCS) -lm -o $@

$(COMPLEX_SAMPLES:%=$(OUT)/samples/%): $(OUT)/samples/%: samples/%.c $(LIBSRCS) | $(OUT)
	$(CC) $(CFLAGS) -I. $< $(LIBSRCS) -lm -o $@

$(FINANCE_SAMPLES:%=$(OUT)/samples/%): $(OUT)/samples/%: samples/finance/%.c $(LIBSRCS) | $(OUT)
	$(CC) $(CFLAGS) -I. $< $(LIBSRCS) -lm -o $@

$(BOOK_SAMPLES:%=$(OUT)/samples/%): $(OUT)/samples/%: samples/books/%.c $(LIBSRCS) | $(OUT)
	$(CC) $(CFLAGS) -I. $< $(LIBSRCS) -lm -o $@

.PHONY: run-samples
run-samples: samples
	@for s in $(SAMPLES); do echo "== $$s =="; ./$(OUT)/samples/$$s || exit 1; done

# ---- benchmark harness (see bench/) ----
bench: $(OUT)/bench/solve_mps $(OUT)/bench/conic_bench $(OUT)/bench/expcone_route_probe $(OUT)/bench/expcone_ipm_probe

$(OUT)/bench/solve_mps: bench/solve_mps.c $(LIBSRCS) | $(OUT)
	mkdir -p $(OUT)/bench
	$(CC) $(CFLAGS) -I. bench/solve_mps.c $(LIBSRCS) -lm -o $@

$(OUT)/bench/conic_bench: bench/conic_bench.c $(LIBSRCS) | $(OUT)
	mkdir -p $(OUT)/bench
	$(CC) $(CFLAGS) -I. bench/conic_bench.c $(LIBSRCS) -lm -o $@

$(OUT)/bench/expcone_route_probe: bench/expcone_route_probe.c $(LIBSRCS) | $(OUT)
	mkdir -p $(OUT)/bench
	$(CC) $(CFLAGS) -I. bench/expcone_route_probe.c $(LIBSRCS) -lm -o $@

$(OUT)/bench/expcone_ipm_probe: bench/expcone_ipm_probe.c $(LIBSRCS) | $(OUT)
	mkdir -p $(OUT)/bench
	$(CC) $(CFLAGS) -I. bench/expcone_ipm_probe.c $(LIBSRCS) -lm -o $@

.PHONY: bench
