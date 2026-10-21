\ PrimalSolver example -- a binary knapsack problem (MIP) in LP format.
\ The `Binaries` section makes the three variables 0/1; the solver branches.
\
\   maximize    8 x0 + 5 x1 + 4 x2
\   subject to  5 x0 + 4 x1 + 3 x2 <= 7
\               x0, x1, x2 in {0, 1}
\
\ Solution: x = (0, 1, 1), objective 9.
\
\ Run:  primal lp_examples/mip_knapsack.lp
Maximize
 obj: 8 x0 + 5 x1 + 4 x2
Subject To
 cap: 5 x0 + 4 x1 + 3 x2 <= 7
Binaries
 x0 x1 x2
End
