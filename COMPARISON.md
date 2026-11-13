# How it compares — fonte della sezione del README

> Tracciato dal 2026-11-13. La tabella di `README.md` (sezione "How it compares")
> è la versione pubblica di questo file: se una riga cambia qui, cambia anche lì.
> Fonte dei dati: `AiWiki/01-Projects/PrimalSolver/Knowledge/Posizionamento-Rispetto-Ai-Solver.md`
> (§3 tabella di capacità, §8-bis conico vs Clarabel/SCS). Nessun numero nuovo, nessuna misura.

---

## How it compares

PrimalSolver is a research-grade convex solver, not a drop-in replacement for the
industrial LP/MIP engines — but the **combination** of what it solves is rare
among the permissively-licensed open solvers, and that combination is its niche.

| | PrimalSolver | MOSEK | Gurobi / CPLEX | HiGHS | Clarabel | SCS |
|---|---|---|---|---|---|---|
| LP / QP | ✅ | ✅ | ✅ | ✅ | LP (QP via cones) | ✅ |
| SOCP | ✅ | ✅ | Gurobi only | — | ✅ | ✅ |
| **SDP** (PSD cone) | ✅ | ✅ | — | — | ✅ | ✅ first-order |
| **Exp / power cones** | ✅ native | ✅ | — | — | ✅ | ✅ |
| MIP / MIQP | ✅ | ✅ | ✅ | ✅ | — | — |
| Method | interior-point | interior-point | simplex + IPM | simplex + IPM | interior-point | **first-order (ADMM)** |
| Certificates | ✅ | ✅ | ✅ | ✅ | ✅ | partial |
| Dependencies | **libm only (C99)** | — | — | C++ | Rust / C++ | C |
| I/O | MPS / LP / OPF / CBF | MPS / LP | MPS / LP | MPS / LP | native API | native API |
| Licence | **Apache-2.0** | commercial | commercial | MIT | Apache-2.0 | MIT |

**What no other permissively-licensed open solver covers** — said once:
**SDP + exponential/power cones + MIP + an interior-point (not first-order)
method, in dependency-free C99.**

- **Clarabel** (Apache-2.0) is the closest open conic IPM — PSD, exp and power
  cones — but it has **no integer variables** and it is Rust/C++.
- **SCS** (MIT) has PSD and exp cones, but is **first-order** — accuracy around
  `1e-4`/`1e-5`, not the `1e-8` an interior-point method reaches.
- **CVXOPT / SDPA / Sedumi** have SDP, but are **GPL** (viral for commercial use).
- **HiGHS** is fast on LP/MIP, but has **no SDP and no non-symmetric cones**.

The niche that follows: embedded/enterprise code that wants SDP, exponential
cones and integer variables in C, with no dependencies and no licence server, and
a migration path from a MOSEK-style API (the 48 MOSEK examples are reproduced
with identical optimal values). Outside that niche — large LP/MIP at industrial
scale — it is not (yet) a substitute.
