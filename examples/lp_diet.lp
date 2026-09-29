\ PrimalSolver example -- a small linear program in CPLEX LP format, the
\ easiest format to write by hand.
\
\   maximize    3 x0 + 2 x1
\   subject to  x0 + x1 <= 4
\               x0 + 3 x1 <= 6
\               0 <= x0 <= 3,  0 <= x1 <= 3
\
\ Solution: x = (3, 1), objective 11.
\
\ Run:  primal examples/lp_diet.lp
Maximize
 obj: 3 x0 + 2 x1
Subject To
 c0: x0 + x1 <= 4
 c1: x0 + 3 x1 <= 6
Bounds
 x0 <= 3
 x1 <= 3
End
