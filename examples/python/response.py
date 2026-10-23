#!/usr/bin/env python3
"""MOSEK port "response" - capture the optimizer's log output.

The log stream is linked to a Python function through PRIMAL_linkfunctotaskstream
(PRIMAL_STREAM_LOG). LP model identical to lo1: min -2x0-3x1 s.t. x0+x1<=4,
x0+3x1<=6, optimum x=(3,1), obj=-9. The test checks that the callback really
receives log messages and that the result is correct. Port of examples/c/response.c.
"""

import ctypes
import math

import primalsolver as ps
from primalsolver import Model, BK, SENSE

STREAM_LOG = 0
STREAMFUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_char_p)

msg_count = 0
first_msg = ""


@STREAMFUNC
def logcb(handle, msg):
    """Log callback recording the message count and the first message."""
    global msg_count, first_msg
    text = msg.decode() if msg else ""
    if msg_count == 0 and msg and len(text) < 64:
        first_msg = text
    msg_count += 1


def main() -> int:
    with Model() as m:
        m.add_vars(2)
        m.add_cons(2)
        m.sense(SENSE.MIN)
        m.cj(0, -2.0); m.cj(1, -3.0)
        m.a_row(0, [0, 1], [1.0, 1.0]); m.con_bounds(0, BK.UP, -math.inf, 4.0)
        m.a_row(1, [0, 1], [1.0, 3.0]); m.con_bounds(1, BK.UP, -math.inf, 6.0)
        m.var_bounds(0, BK.RA, 0.0, math.inf)
        m.var_bounds(1, BK.RA, 0.0, math.inf)
        rc = ps.PRIMAL_linkfunctotaskstream(m.task, STREAM_LOG, None,
                                            ctypes.cast(logcb, ctypes.c_void_p))
        if rc != 0:
            print(f"link rc={rc}")
            return 1
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    x, po = r.x, r.objective
    print(f"x = ({x[0]:.6f}, {x[1]:.6f}), obj = {po:.6f}")
    print(f'log received: {msg_count} messages (first: "{first_msg}")')
    ok = msg_count > 0 and abs(x[0] - 3.0) < 1e-6 and abs(x[1] - 1.0) < 1e-6 and abs(po + 9.0) < 1e-6
    print(f"{'OK' if ok else 'FAIL'} (callback stream + optimum)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
