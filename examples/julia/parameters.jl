#!/usr/bin/env julia
# parameters.jl — MOSEK port "parameters": reading/writing parameters
# (putintparam/putdouparam and the getters).
#
# Checks: every parameter set is read back unchanged; an invalid parameter is
# rejected (ERR_ARG); INTPNT_MAX_ITERATIONS really limits the IPM (TRM).
# Port of examples/c/parameters.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model
const B = Primal.Bindings

const IPAR_LOG = 4
const IPAR_SIMPLEX_MAX_ITERATIONS = 10
const IPAR_INTPNT_MAX_ITERATIONS = 11
const DPAR_INTPNT_TOL_PFEAS = 0
const DPAR_INTPNT_TOL_DFEAS = 1
const DPAR_INTPNT_TOL_REL_GAP = 2

function main()
    m = Model(0, 0)
    q = Model(0, 0)
    try
        t = m.task[]
        rc = 0                                      # int params
        B.putintparam(t, IPAR_LOG, 0)
        B.putintparam(t, IPAR_SIMPLEX_MAX_ITERATIONS, 5000)
        B.putintparam(t, IPAR_INTPNT_MAX_ITERATIONS, 123)
        vi = Ref{Cint}(-1)
        B.getintparam(t, IPAR_LOG, vi); rc += (vi[] != 0)
        B.getintparam(t, IPAR_SIMPLEX_MAX_ITERATIONS, vi); rc += (vi[] != 5000)
        B.getintparam(t, IPAR_INTPNT_MAX_ITERATIONS, vi); rc += (vi[] != 123)

        B.putdouparam(t, DPAR_INTPNT_TOL_PFEAS, 1e-7)   # double params
        B.putdouparam(t, DPAR_INTPNT_TOL_DFEAS, 2e-7)
        B.putdouparam(t, DPAR_INTPNT_TOL_REL_GAP, 3e-7)
        vd = Ref{Cdouble}(-1.0)
        B.getdouparam(t, DPAR_INTPNT_TOL_PFEAS, vd); rc += (abs(vd[] - 1e-7) > 1e-15)
        B.getdouparam(t, DPAR_INTPNT_TOL_DFEAS, vd); rc += (abs(vd[] - 2e-7) > 1e-15)
        B.getdouparam(t, DPAR_INTPNT_TOL_REL_GAP, vd); rc += (abs(vd[] - 3e-7) > 1e-15)

        # invalid parameters rejected: 9999 exists in neither namespace, and a
        # tolerance outside its [0,1] range is an API error
        rc += (B.putintparam(t, 9999, 1) == 0 ? 1 : 0)
        rc += (B.putdouparam(t, 9999, 1.0) == 0 ? 1 : 0)
        rc += (B.putdouparam(t, DPAR_INTPNT_TOL_PFEAS, -1.0) == 0 ? 1 : 0)
        rc += (B.getintparam(t, 9999, vi) == 0 ? 1 : 0)

        # INTPNT_MAX_ITERATIONS as a real limit: QP with 1 iteration -> TRM_MAX_ITER
        add_vars(q, 2)
        add_cons(q, 1)
        var_bounds(q, 0, BK_LO; lo=0.0, up=Inf)
        var_bounds(q, 1, BK_LO; lo=0.0, up=Inf)
        qobj(q, [(0, 0, 1.0), (1, 1, 1.0)])
        a_row(q, 0, [0, 1], [1.0, 1.0])
        con_bounds(q, 0, BK_LO; lo=1.0, up=Inf)
        int_param(q, IPAR_INTPNT_MAX_ITERATIONS, 1)
        rq = Int(B.optimize(q.task[]))
        @printf("QP with max_iter=1: rc=%d (expected non-OK: 1007 TRM or 1002)\n", rq)
        rc_trm = rq != 0                            # the limit really interrupts

        @printf("parameter error count: %d\n", rc)
        ok = rc == 0 && rc_trm
        @printf("%s (parameters round-trip + real iteration limit)\n", ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(q)
        free(m)
    end
end

exit(main())
