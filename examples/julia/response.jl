#!/usr/bin/env julia
# response.jl — MOSEK port "response": capture the optimizer's log output
# through the stream callback PRIMAL_linkfunctotaskstream(PRIMAL_STREAM_LOG).
#
# LP: min -2x0 - 3x1  s.t.  x0 + x1 <= 4,  x0 + 3x1 <= 6,  x >= 0
# -> x = (3, 1), obj = -9. Checks that the callback really receives log
# messages and that the result is correct. Port of examples/c/response.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model
const B = Primal.Bindings

const STREAM_LOG = 0
const msg_count = Ref(0)
const first_msg = Ref("")

# Log callback recording the message count and the first message.
function logcb(handle::Ptr{Cvoid}, msg::Cstring)::Cvoid
    if msg_count[] == 0 && msg != C_NULL
        s = unsafe_string(msg)
        sizeof(s) < 64 && (first_msg[] = s)
    end
    msg_count[] += 1
    return nothing
end

function main()
    m = Model(0, 0)
    try
        add_vars(m, 2)
        add_cons(m, 2)
        sense(m, SENSE_MIN)
        cj(m, 0, -2.0)
        cj(m, 1, -3.0)
        a_row(m, 0, [0, 1], [1.0, 1.0]); con_bounds(m, 0, BK_UP; lo=-Inf, up=4.0)
        a_row(m, 1, [0, 1], [1.0, 3.0]); con_bounds(m, 1, BK_UP; lo=-Inf, up=6.0)
        var_bounds(m, 0, BK_RA; lo=0.0, up=Inf)
        var_bounds(m, 1, BK_RA; lo=0.0, up=Inf)

        cb = @cfunction(logcb, Cvoid, (Ptr{Cvoid}, Cstring))
        rc = Int(B.linkfunctotaskstream(m.task[], STREAM_LOG, C_NULL, cb))
        if rc != 0
            @printf("link rc=%d\n", rc)
            return 1
        end
        r = solve(m)
        if r.rc != 0
            @printf("optimize rc=%d\n", r.rc)
            return 1
        end
        x, po = r.x, r.objective
        @printf("x = (%.6f, %.6f), obj = %.6f\n", x[1], x[2], po)
        @printf("log received: %d messages (first: \"%s\")\n", msg_count[], first_msg[])
        ok = msg_count[] > 0 && abs(x[1] - 3.0) < 1e-6 && abs(x[2] - 1.0) < 1e-6 && abs(po + 9.0) < 1e-6
        @printf("%s (callback stream + optimum)\n", ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
