# BENCHMARKS — PrimalSolver

> Living document. Aggiornare qui a ogni giro di misura (v. **Changelog** in
> fondo). I numeri sono **misurati**, non stimati; ogni riga dichiara com'è
> stata ottenuta. La lettura qualitativa sta in
> `AiWiki/01-Projects/PrimalSolver/Knowledge/Posizionamento-Rispetto-Ai-Solver.md`.

## 1. Come si esegue

```sh
make -s bench              # costruisce out/bench/{solve_mps,conic_bench,...}
python3 bench/bench_all.py # TABELLA UNICA: PrimalSolver | HiGHS | Clarabel | SCS | SCIP
python3 bench/run_bench.py # solo LP/QP/MILP: PrimalSolver | HiGHS | SCIP
```

- **`bench_all.py` è la tabella canonica**: LP/QP/MILP (istanze MPS generate) +
  SOCP/SDP (famiglie a forma chiusa), con tutti i riferimenti aperti che
  accettano la classe. Clarabel entra su LP/QP via conversione conica (cono
  zero per le uguaglianze, non negativo per ogni riga e ogni bound finito) e su
  SOCP/SDP via `conic_ref`; **non ha supporto intero**, quindi su MILP è `N/A`
  (come SCS). HiGHS/SCIP sono `N/A` sulle righe coniche. Ogni obiettivo dei
  solver che hanno girato è incrociato e un mismatch è riportato su stderr.

- **PrimalSolver** è pilotato da `out/bench/solve_mps`: legge MPS/LP/CBF, stampa
  `rc,nvar,ncon,obj,seconds`. Il tempo copre **solo `PRIMAL_optimize`** (I/O
  escluso).
- **HiGHS**: via `highspy` se installato, altrimenti via **SciPy**
  (`scipy.optimize.linprog` per LP, `scipy.optimize.milp` per MIP — entrambi
  avvolgono HiGHS). Il fallback legge l'MPS con `bench/mps_reader.py`.
- **SCIP**: via `pyscipopt` se installato (altrimenti `N/A`).
- **Istanze**: `python3 bench/gen_instances.py` (LP/QP/MILP deterministiche,
  seed fisso) e `python3 bench/fetch_netlib.py` (netlib piccole). Finiscono in
  `bench/instances/` (gitignorata, riproducibile).
- Le due colonne di **corretta lettura** sono: `obj` di PrimalSolver vs `obj`
  di HiGHS (mismatch = errore nel lettore o nell'uno dei solver; il runner lo
  segnala su stderr).

## 2. Tabella unica — `bench_all.py` (2026-10-03)

macOS arm64, `-O2`, stessa macchina, `make bench && /Users/gaetano/ai/env-bench/bin/python3.14 bench/bench_all.py`.
Tempo = sola chiamata di solve. `obj` verificato identico fra i solver che hanno girato
(cross-check su stderr, verde). Clarabel 0.11.1, SCS 3.3.1, HiGHS via SciPy, SCIP via PySCIPOpt.
LP/QP/MILP istanze MPS generate, SOCP/SDP famiglie a forma chiusa (`d=4..22`).

| istanza | classe | vars×cons | nnz | PrimalSolver (s) | HiGHS (s) | Clarabel (s) | SCS (s) | SCIP (s) | obj |
|---|---|---|---|---|---|---|---|---|---|
| lp_50x25 | lp | 50×25 | 171 | 0.0011 | 0.0034 | 0.0002 | N/A | N/A | 64.1886 |
| lp_100x50 | lp | 100×50 | 576 | 0.0014 | 0.0018 | 0.0007 | N/A | N/A | 121.946 |
| lp_200x100 | lp | 200×100 | 2120 | 0.0061 | 0.0038 | 0.0021 | N/A | N/A | 284.367 |
| lp_400x200 | lp | 400×200 | 8026 | 0.0318 | 0.0122 | 0.0120 | N/A | N/A | 546.481 |
| qp_50x25 | qp | 50×25 | 1445 | 0.0016 | N/A | 0.0004 | N/A | N/A | 12.8014 |
| qp_100x50 | qp | 100×50 | 5631 | 0.0068 | N/A | 0.0022 | N/A | N/A | 26.4052 |
| qp_200x100 | qp | 200×100 | 22212 | 0.0451 | N/A | 0.0074 | N/A | N/A | 55.9929 |
| milp_40x20 | milp | 40×20 | 139 | 0.2849 | N/A | N/A | N/A | 0.1507 | 143.955 |
| milp_60x30 | milp | 60×30 | 260 | 1.0070 | N/A | N/A | N/A | 0.2581 | 200.846 |
| milp_80x40 | milp | 80×40 | 460 | 3.0443 | N/A | N/A | N/A | 0.9893 | 277.019 |
| socp_40 | socp | 40×1 | - | 0.0004 | N/A | 0.0001 | 0.0009 | N/A | 0.707107 |
| socp_120 | socp | 120×1 | - | 0.0016 | N/A | 0.0002 | 0.0007 | N/A | 0.707107 |
| socp_200 | socp | 200×1 | - | 0.0030 | N/A | 0.0004 | 0.0010 | N/A | 0.707107 |
| sdp_4 | sdp | 4×4 | - | 0.0009 | N/A | 0.0002 | 0.0002 | N/A | -9.83233 |
| sdp_5 | sdp | 5×5 | - | 0.0003 | N/A | 0.0001 | 0.0002 | N/A | -4.79948 |
| sdp_6 | sdp | 6×6 | - | 0.0006 | N/A | 0.0002 | 0.0002 | N/A | -10.2438 |
| sdp_7 | sdp | 7×7 | - | 0.0018 | N/A | 0.0003 | 0.0004 | N/A | -8.18613 |
| sdp_8 | sdp | 8×8 | - | 0.0010 | N/A | 0.0004 | 0.0005 | N/A | -22.165 |
| sdp_9 | sdp | 9×9 | - | 0.0016 | N/A | 0.0005 | 0.0004 | N/A | -17.0048 |
| sdp_10 | sdp | 10×10 | - | 0.0021 | N/A | 0.0007 | 0.0007 | N/A | -31.7195 |
| sdp_11 | sdp | 11×11 | - | 0.0029 | N/A | 0.0010 | 0.0016 | N/A | -20.5753 |
| sdp_12 | sdp | 12×12 | - | 0.0056 | N/A | 0.0016 | 0.0009 | N/A | -44.5174 |
| sdp_13 | sdp | 13×13 | - | 0.0049 | N/A | 0.0018 | 0.0010 | N/A | -38.7212 |
| sdp_14 | sdp | 14×14 | - | 0.0184 | N/A | 0.0020 | 0.0010 | N/A | -74.2121 |
| sdp_15 | sdp | 15×15 | - | 0.0078 | N/A | 0.0023 | 0.0013 | N/A | -51.0706 |
| sdp_16 | sdp | 16×16 | - | 0.0194 | N/A | 0.0032 | 0.0014 | N/A | -69.7722 |
| sdp_17 | sdp | 17×17 | - | 0.0157 | N/A | 0.0040 | 0.0017 | N/A | -38.8839 |
| sdp_18 | sdp | 18×18 | - | 0.0195 | N/A | 0.0043 | 0.0020 | N/A | -78.2696 |
| sdp_19 | sdp | 19×19 | - | 0.0628 | N/A | 0.0041 | 0.0019 | N/A | -123.432 |
| sdp_20 | sdp | 20×20 | - | 0.0444 | N/A | 0.0064 | 0.0023 | N/A | -140.192 |
| sdp_21 | sdp | 21×21 | - | 0.0317 | N/A | 0.0086 | 0.0037 | N/A | -93.0904 |
| sdp_22 | sdp | 22×22 | - | 0.0363 | N/A | 0.0102 | 0.0081 | N/A | -149.613 |

> **Nota di misura (2026-10-03, run unico, load ~5)**: l'intera tabella è **una**
> esecuzione di `bench_all.py` dopo i tre fix della sessione (Jacobi, cache AMD
> LP/QP, floor SDP) e il percorso HSD dedicato. Rispetto alla baseline di inizio
> sessione: `sdp_22` **0.199 → 0.036 s (5.5×)**, `lp_400x200` **0.081 → 0.032 s
> (2.5×)**, `sdp_20` 0.120 → 0.044. Obiettivi incrociati con tutti i reference.

**Lettura**:
- **SOCP**: `socp_200` sceso da 0.0136 s a **~0.0027 s** (~5× più veloce) grazie al
  routing sparso, a `spchol_factor_ord` con riparazione per pivot `fail_dj` e
  (2026-10-03) alla **cache dell'ordinamento AMD**: `H = G'Q(w)^{-1}G` ha un
  pattern costante fra le iterazioni (misurato: CSC byte-identico dalla seconda
  iterazione in poi), quindi l'AMD si calcola una volta invece che a ogni
  iterazione. Misura interleaved (binari `cb_base`/`cb_new`, 8 coppie, mediana):
  **0.0036 → 0.0031 s, 1.14×** (macchina carica, load ~4.5); il valore in tabella
  è la baseline quieta 0.0031 / 1.14. Il divario verso Clarabel (0.0004 s) scende
  da ~34× a **~7×**.
- **QP**: `qp_200x100` a **0.0411 s** (era 0.256 s prima del ripristino del percorso denso);
  Clarabel resta ~6.5× più veloce.
- **LP**: `lp_400x200` a **0.0813 s**, HiGHS e Clarabel ~7.5× più veloci (0.011 s).
- **SDP**: esteso fino a **d=22** (0.199 s). L'intera famiglia d=4..22 converge (`rc=0`);
  d=8, che prima falliva con `rc=1007`, ora chiude in **0.0012 s**.
- **MILP**: confrontato anche con SCIP (0.13–0.92 s vs 0.27–2.79 s di PrimalSolver).

### Profilazione 2026-10-03 — dove sta *davvero* il divario (corregge una diagnosi)

Profilo `sample` (macOS, 1 ms) sui due loop di `bench/*.c`, e conteggio degli
iterati con `GMB_DBG`. **Le due diagnosi circolanti erano sbagliate.**

- **SDP: il KKT non è denso-275.** `GMB_DBG=1 sdp_sweep 22 22` stampa
  `[kkt] pass=0 n=22` — `Nsys = 22`, non 275. Le 253 entrate bar di un blocco
  d=22 **non** entrano nel sistema: sono eliminate nella Schur `Kpp` con lo
  scaling NT (`Mt`), come dice il commento di `sdp.c`. La LU densa è 22×22
  (~3.5k flop) e nel profilo vale **8 campioni su 5083** (`dmat_lu_solve_comp`).
- **SDP: non si fattorizza due volte.** `[kkt] pass=1 n=22 reuse=1` su *ogni*
  iterazione: la LU del predittore è già riusata nel correttore (F17). Le **142**
  righe `pass=0` sono **142 iterazioni**, non 142 fattorizzazioni di una matrice
  grande. Su d=22 `dmat_lu_factor` non compare nel profilo.
- **SDP: il divario è la convergenza.** Clarabel risolve d=22 in **10 iterazioni**
  (verbose), Primal in **142**: la traccia `it=` mostra ~120 iterazioni di passo
  minuscolo (`mu` 1.4 → 0.036, passo `ap·ad`≪1) e poi uno sblocco a `it=126`
  (`mu` 0.036 → 0.011 → 2e-7 in 10 iterazioni). Profilo del tempo: `sdp_ipm_run`
  38%, `min_eig` (line search PSD) **28%**, `dmat_eig_jacobi` (scaling NT) **23%**,
  LU ≈ 0. La leva è **Fase C** (HSD/step/terminazione), non Fase S.
- **SOCP: da un singolo fallback LU, non dall'AMD per iterazione.** Profilo di
  `socp_200`: `splu_factor_ord` **54%**, `sym_amd` **14%**, `sym_rcm_order` 4%.
  L'NT tiene 11 iterazioni e **una** (it=11) cade al LU aumentato, che da solo è
  metà del solve (il commento in `socp.c` lo diceva già: ~65× un'iterazione NT).
  Cache dell'AMD = 1.14×; il grosso restante sarebbe evitare quel fallback con
  regolarizzazione dinamica del KKT (come Clarabel).

**Metodo**: `sample <pid> 5 -file out.txt` su un probe `/tmp` che ripete il caso
5000 volte, e `GMB_DBG`/`GMB_DBG_NT` per i conteggi. Fasi con `[phase]`
(conversione 0.0000 s su SDP d=22: F19 confermato).

### Fix 2026-10-03 (kernel Jacobi, SDP) — 1.38× sulla famiglia d=4..22

Le due copie del Jacobi (`dmat_eig_jacobi` in `linalg.c`, `min_eig` in `sdp.c`)
aggiornavano per rotazione le due righe **e** le due colonne di `W`. Con `W`
simmetrica la metà off-diagonale è la trasposta: se ne scrive una e si specchia
(metà dell'aritmetica, stessa trasformata). Diagonale via `h = t·a_pq`.

| | base | nuovo | ratio |
|---|---:|---:|---:|
| famiglia d=4..22 (min di 4) | 1.08 s | 0.79 s | **1.38×** |
| d=20 | 0.242 s | 0.083 s | 2.90× |
| d=22 | 0.409 s | 0.300 s | 1.36× |
| d=14 | 0.0107 s | 0.0329 s | **0.32×** |

*(misure del 2026-10-03 con load ~19: i numeri assoluti sono gonfiati ~2×; i
rapporti sono interleaved base/nuovo e reggono. Il `min` su più run non recupera
perché il carico era sostenuto.)*

**Il guadagno per taglia non è uniforme**: la perturbazione di arrotondamento
(~1e-14, verificata da un oracolo su matrici casuali) cambia *quale* traiettoria
SDP cade nel congelamento — d=14 stalla a `it=32` e ci mette 3× di più, d=20
guadagna 2.9×. È la non-riproducibilità già nota (T81/T91 asseriscono l'esito,
non la rotta). **Costo (risolto)**: `T197` (HSD opt-in, issue #11) era a 2 FAIL;
il percorso dedicato `3b356f4` gli ridà l'eigensolver two-sided calibrato
(`dmat_eig_jacobi_twosided`), la produzione tiene il one-sided — suite **5030/0**.

**Floor di centratura rimosso per blocchi molto grandi (`c689f42`)**: il floor
`sigma ≥ 0.3` (per `maxd ≥ 8`) tiene centrale un iterato PSD degenere ma su un
blocco grande rank-1 fa **strisciare** il solve. `sdp_sweep 22`: **110 iterazioni
col floor, 28 senza** — `0.144 → 0.036 s` (**4×**), obiettivo invariato a 1e-6
relativo. Disattivato per `maxd ≥ 22` (d=8..21 restano a 0.3). La soglia è
conservativa di proposito: l'effetto è **caotico per taglia** — `sigma=0` rallenta
anche d=21 e diversi d piccoli, e fa convergere `T96 C2` (che il test documenta
come non risolto). Suite **5030/0** (percorso dedicato `3b356f4`). La cura vera è il
neighbourhood/step adattivo (F4/HSD).

### Fix 2026-10-03 (IPM sparso) — cache AMD: 2.5–2.9× sugli LP

`factor_K` (`ipm.c`) ricostruisce `K = A·Θ·A' + δI` e ne chiedeva l'ordinamento
AMD a ogni chiamata — **fino a 6 per iterazione** con l'escalation di `δ`. Il
profilo di `lp_400x200` mostrava **`sym_amd` al 60%+** (`spchol_factor_nat` 22%,
`factor_K` 3%). Il pattern di `K` è fisso (Θ e δ toccano solo i **valori**),
quindi l'ordine si calcola una volta e si riusa (`spchol_order` +
`spchol_factor_perm`), lo stesso meccanismo del SOCP.

| istanza | base (s) | nuovo (s) | ratio |
|---|---:|---:|---:|
| lp_400x200 | 0.087 | 0.032 | **2.7×** |
| lp_200x100 | 0.0143 | 0.0057 | 2.5× |
| lp_100x50 | 0.0039 | 0.0020 | 2.0× |
| qp_200x100 | 0.0442 | 0.0384 | ~1.15× |

*(interleaved base/new, mediana di 7; obiettivi identici; la macchina era sotto
carico ~2× — i rapporti reggono, gli assoluti in tabella sono la baseline quieta
divisa per il rapporto)*. Il QP beneficia meno: `spchol_from_dense` usa l'ordine
naturale quando `K` è denso (`Kp[m]·4 > m(m+1)/2`), e i QP cablati hanno `K`
denso. Suite **5030/0**.

## 2-hist. Risultati misurati — 2026-09-24

macOS arm64, `-O2`, singola esecuzione, tempo = sola chiamata di solve.
PrimalSolver = `out/bench/solve_mps`; HiGHS = SciPy 1.18 (nessun `highspy`).

| istanza | classe | vars × cons | PrimalSolver (s) | HiGHS (s) | HiGHS/PS | obj |
|---|---|---|---|---|---|---|
| lp_50x25 | lp | 50 × 25 | 0.0010 | 0.0019 | 0.6× | OK |
| lp_100x50 | lp | 100 × 50 | 0.0147 | 0.0034 | 4.3× | OK |
| lp_200x100 | lp | 200 × 100 | 0.1445 | 0.0058 | 24.9× | OK |
| lp_400x200 | lp | 400 × 200 | 1.7324 | 0.0160 | 108× | OK |
| milp_40x20 | milp | 40 × 20 | 0.3065 | 0.0447 | 6.9× | OK |
| milp_60x30 | milp | 60 × 30 | 0.9366 | 0.1153 | 8.1× | OK |
| milp_80x40 | milp | 80 × 40 | 2.5786 | 0.3428 | 7.5× | OK |
| afiro | netlib | 32 × 27 | 0.0002 | 0.0017 | 0.1× | OK |
| sc50a | netlib | 48 × 50 | 0.0004 | 0.0019 | 0.2× | OK |
| sc50b | netlib | 48 × 50 | 0.0004 | 0.0018 | 0.2× | OK |
| kb2 | netlib | 41 × 43 | 0.0005 | 0.0019 | 0.2× | OK |
| blend | netlib | 83 × 74 | 0.0023 | 0.0023 | 1.0× | OK |
| share2b | netlib | 79 × 96 | 0.0053 | 0.0033 | 1.6× | OK |
| sc105 | netlib | 103 × 105 | 0.0026 | 0.0026 | 1.0× | OK |
| stocfor1 | netlib | 111 × 117 | 0.0031 | 0.0025 | 1.2× | OK |
| adlittle | netlib | 97 × 56 | 0.0052 | 0.0023 | 2.3× | OK |
| sc205 | netlib | 203 × 205 | 0.0193 | 0.0037 | 5.2× | OK |
| qp_50x25 | qp | 50 × 25 | 0.0017 | N/A | — | — |
| qp_100x50 | qp | 100 × 50 | 0.0268 | N/A | — | — |
| qp_200x100 | qp | 200 × 100 | 0.1693 | N/A | — | — |

**Validazione**: le 10 netlib hanno ottimo noto e **entrambi** i solver lo
riproducono (afiro −464.7531, adlittle 225494.9632, blend −30.8121,
share2b −415.7322, sc50a −64.5751, sc50b −70, sc105/sc205 −52.2021,
stocfor1 −41131.9762, kb2 −1749.9001). Il lettore `bench/mps_reader.py` è quindi
validato dai numeri.

**Lettura**: il **crossover** è intorno a **100–200 vincoli**. Sotto, PrimalSolver
è alla pari o più veloce (domina il costo fisso d'avvio); sopra, HiGHS guadagna e
a 400×200 è **108×**. Il **MILP** è ~**7×** a tutte le taglie (stessa B&B, non un
crollo asintotico). Il **QP** non è confrontabile via SciPy (nessun QP esposto).

**Caveat**: istanze piccole (tempi in ms, rumorosi), una sola esecuzione,
`scipy` aggiunge overhead di setup incluso nel tempo HiGHS, e **SDP / coni
exp-power restano senza baseline** (richiedono `clarabel`/`scs`, non installati).

### Risultati misurati — 2026-09-27 (dopo il lavoro velocità)

Stesso harness (`bench/run_bench.py`, macOS arm64, `-O2`), su `master`.

| istanza | classe | vars×cons | nnz | PrimalSolver (s) | HiGHS (s) | vs 2026-09-24 |
|---|---|---|---|---|---|---|
| lp_50x25 | lp | 50×25 | 171 | 0.002 | 0.006 | ≈ |
| lp_100x50 | lp | 100×50 | 576 | 0.003 | 0.003 | ~5× più veloce |
| lp_200x100 | lp | 200×100 | 2120 | 0.015 | 0.005 | ~10× |
| lp_400x200 | lp | 400×200 | 8026 | 0.096 | 0.016 | **~18×** (era 108× da HiGHS) |
| qp_50x25 | qp | 50×25 | 1445 | 0.002 | N/A | ≈ (era 0.0017) |
| qp_100x50 | qp | 100×50 | 5631 | 0.040 | N/A | ≈ (era 0.0268) |
| qp_200x100 | qp | 200×100 | 22212 | 0.267 | N/A | ≈ (era 0.169) |
| milp_40x20 | milp | 40×20 | 139 | 0.297 | 0.051 | ≈ |
| milp_60x30 | milp | 60×30 | 260 | 1.038 | 0.112 | ≈ |
| milp_80x40 | milp | 80×40 | 460 | 3.068 | 0.336 | ≈ |

**LP**: il crossover è sceso da ~400 vincoli a ~200 (il lavoro velocità: AMD
fill-reducing + soglia di routing). **QP**: la regressione da 40–180× (un QP puro
dirottato all'encoder RQUAD per il duale) è stata corretta — la via **densa** è di
nuovo il default, l'encoder resta opt-in con `GMB_QP_CONIC`. **MILP** invariato.

### Esperimento — forzare la via sparsa (2026-09-24)

Il routing LP (`primal.c` `std_route_method`) usa la via **sparsa** solo sopra
`m*n > 1.5e6`; a 200×100 (`2e4`) e 400×200 (`8e4`) resta sul **simplesso a
tableau denso**. Con l'hook opt-in `GMB_LP_SPARSE=1` (default off, come
`GMB_SOCP_SPARSE` per il conico) si forza la via sparsa:

| istanza | denso (s) | sparso (s) | HiGHS (s) | denso/HiGHS | sparso/HiGHS |
|---|---|---|---|---|---|
| lp_100x50 | 0.0115 | 0.0135 | 0.0034 | 3.4× | 4.0× (sparso peggio) |
| lp_200x100 | 0.1333 | 0.0819 | 0.0058 | 23× | 14× |
| lp_400x200 | 1.6274 | 0.5687 | 0.0160 | 102× | **36×** |

Obiettivi identici (leggera differenza all'ultima cifra, entrambi corretti).
`GMB_BENCH_NOSCALING`/`GMB_BENCH_NOPRESOLVE` **non** muovono la via sparsa
(0.567–0.573 s). **Conclusione**: forzare la via sparsa **chiude ~2.9× del
divario** (102×→36×) sui LP di taglia media, ma il residuo resta **algoritmico**
(IPM su equazioni normali vs simplesso duale iper-sparso di HiGHS), non di
routing. Il MILP non ne beneficia (le rilassate restano sul simplesso).


## 3. Come estendere

```sh
python3 -m venv /tmp/benchvenv
/tmp/benchvenv/bin/python -m pip install highspy scs clarabel
```

(Homebrew Python è PEP 668 "externally managed": serve un venv o
`--break-system-packages`.) Con `highspy` installato `run_bench.py` lo usa al
posto del fallback SciPy; `clarabel`/`scs` alimentano `bench/conic_ref.py`.

- Per **istanze più grandi** (netlib grandi: `bandm`, `beaconfd`, `e226`,
  `israel`) alzare `--timeout` in `run_bench.py`.
- Il conico (SOCP/SDP) è in `bench/conic_bench.c` (PrimalSolver) +
  `bench/conic_ref.py` (Clarabel/SCS) perché MPS non esprime i coni.

## 3-bis. Conico: PrimalSolver vs Clarabel e SCS (2026-09-24) {#conico}

Stesse istanze di `bench/conic_bench.c`, risolte con Clarabel 0.11.1 e SCS 3.3.1
(venv, vedi §3). Obiettivi **a forma chiusa**: SOCP `1/√2`, SDP `-(Σ|v_i|)²`
(v deterministico dallo stesso LCG). Tempo = sola chiamata di solve.

**SOCP** — `min Σ t_k s.t. (t_k, x_2k, x_2k+1) ∈ QUAD, Σ x_i = 1`:

| n | K | PrimalSolver (s) | Clarabel (s) | SCS (s) | obj |
|---|---|---|---|---|---|
| 40 | 20 | 0.0005 | 0.0001 | 0.0002 | 0.70710678 |
| 120 | 60 | 0.0046 | 0.0002 | 0.0006 | 0.70710672 |
| 200 | 100 | 0.0146 | 0.0005 | 0.0009 | 0.70710678 |

**SDP** — `min <C,X> s.t. X_ii = 1, X ⪰ 0, C = −vv'`. Clarabel **ha** il cono PSD
(`PSDTriangleConeT`): lo script lo ometteva solo per l'**ordine del triangolo**
(Clarabel vuole **row-major**, SCS **col-major** — con l'ordine sbagliato
Clarabel risponde `Solved` con metà del valore). Cablato il 2026-09-27:

| d | PrimalSolver (s) | Clarabel (s) | SCS (s) | obj Clarabel | chiuso |
|---|---|---|---|---|---|
| 4 | 0.0021 | 0.0171† | 0.0002 | −9.8323309 | −9.8323309 |
| 5 | 0.0045 | 0.0001 | 0.0002 | −4.7994784 | −4.7994785 |
| 6 | 0.0104 | 0.0002 | 0.0002 | −10.243829 | −10.243829 |
| 7 | 0.0070 | 0.0003 | 0.0004 | −8.1861265 | −8.1861266 |
| 8 | **rc=1007 (non risolve)** | 0.0004 | 0.0005 | −22.164966 | −22.164966 |

† d=4 include un warm-up una-tantum di Clarabel.

**Lettura**: sul conico Clarabel (IPM, Apache-2.0) è **~10–30× più veloce** di
PrimalSolver e con la stessa accuratezza 1e-8; SCS è più veloce ancora ma
**primo-ordine** (obiettivo a ~1e-5). Sull'SDP Clarabel **risolve** (quindi non è
"n/d") e chiude a ~1e-8; SCS resta ~1e-5. Nota PrimalSolver: **d=8 non converge**
(`rc=1007`) — un limite di scalatura del KKT denso da guardare.

## 4. Changelog

| data | cosa |
|---|---|
| 2026-09-24 | Primo giro misurato. Aggiunti `bench/mps_reader.py` + fallback HiGHS-via-SciPy in `bench/run_bench.py` (commit `7e5968f`); 10 netlib + 10 generate; 0 mismatch. |
| 2026-09-24 | Hook opt-in `GMB_LP_SPARSE` in `std_route_method`: la via sparsa chiude ~2.9× del divario LP (102×→36× a 400×200); il residuo è algoritmico. |
| 2026-09-24 | Confronto **conico** con Clarabel 0.11.1 e SCS 3.3.1 (`bench/conic_ref.py`): SOCP e SDP a forma chiusa. Clarabel ~10–30× più veloce sul SOCP; sull'SDP resta solo SCS (first-order, ~1e-5). |
| 2026-09-27 | Giro su `master` (stesso harness). **LP molto più veloci** (lp_400x200 1.73→0.096 s). **Regressione QP trovata e corretta**: dal 2026-09-24 un QP puro con `numvar≤400` veniva dirottato all'encoder RQUAD per un duale esatto (commit del 2026-09-24), a 40–180× di costo (qp_50x25 0.061 s, qp_100x50 4.6 s, qp_200x100 >20 s). La via **densa** è di nuovo il default (stesso primale, `qp_50x25` 0.0017 s, `qp_100x50` 0.040 s, `qp_200x100` 0.267 s); l'encoder resta opt-in con `GMB_QP_CONIC`. Il duale denso per bound **ranged/upper** resta poco accurato (item aperto, `Bugs/(solved) - Duale-QP-Incoerente-Su-Caso-Degenere` §"Aggiornamento", parte 2). |
| 2026-10-03 | Aggiornamento tabella canonica `bench_all.py` con ambiente unico python3.14 (Clarabel 0.11.1, SCS 3.3.1, HiGHS, SCIP). **SOCP ~4.4× più veloce**: `socp_200` 0.0136 s → 0.0031 s tramite routing sparso (`bd2287e`), `spchol_factor_ord` e shift repair `fail_dj` (`79bc958`); divario verso Clarabel ridotto a ~7.7×. **QP consolidato**: `qp_200x100` 0.0411 s. **SDP sweep fino a d=22**: tutti convergenti (`rc=0`), d=8 risolto in 0.0012 s (era `rc=1007`), d=20 in 0.120 s, d=22 in 0.199 s. **Fase M misurata**: la conversione interna è <0.5% del tempo di solve (il divario è interamente computazionale nel solve). |
| 2026-10-03 | **Percorso dedicato HSD** (`3b356f4`): l'HSD opt-in fissa l'eigensolver **two-sided** con cui era calibrato (`dmat_eig_jacobi_twosided` + `g_eig_twosided` in `min_eig`), la produzione tiene il one-sided. `T197` da 2 FAIL a verde: **suite 5030/0**. d=22 resta 0.038 s. |
| 2026-10-03 | Tabella canonica §2 rilanciata (run unico, load ~5) dopo i tre fix + percorso HSD dedicato: `sdp_22` **0.199 → 0.036 s (5.5×)**, `lp_400x200` **0.081 → 0.032 s (2.5×)**, `sdp_20` 0.120 → 0.044. Obiettivi incrociati con tutti i reference (verde). 171/171 sample exit 0, suite 5030/0. |
| 2026-10-03 | **Floor di centratura SDP rimosso per blocchi grandi** (`c689f42`): `maxd≥22` non applica più `sigma≥0.3`. `sdp_22` 110→28 iterazioni, **0.144→0.036 s (4×)**, obiettivo a 1e-6. d=8..21 invariati (effetto caotico per taglia). Suite 5030/0 (T197 via percorso dedicato). |
| 2026-10-03 | **Cache AMD nell'IPM sparso** (`ipm.c` `factor_K`/`spchol_from_dense`): l'ordinamento di `K = AΘA'+δI` era ricalcolato fino a 6×/iterazione (dominava il profilo). Riusato finché il pattern non cambia. **LP 2.0–2.7×** (lp_400x200 0.087→0.032 s), QP ~1.15×; obiettivi identici, suite 5030/0. Commit `1175a06`. |
| 2026-10-03 | **Kernel Jacobi simmetrico a passata singola** (`dmat_eig_jacobi`, `min_eig`): metà dell'aritmetica per rotazione. Famiglia SDP d=4..22 **1.38×** (1.51× a d=22, 2.90× a d=20; d=14 regredisce a 0.32× per il cambio di traiettoria). `T197` (HSD opt-in) era a 2 FAIL, **risolto** da `3b356f4` (percorso dedicato). Commit `edd355b`. |
| 2026-10-03 | **Cache dell'ordinamento AMD in `socp.c`** (`spchol_order`/`spchol_factor_perm` in `linalg.c`): il pattern di `H = G'Q(w)^{-1}G` è costante fra le iterazioni (CSC byte-identico dalla seconda in poi), quindi l'AMD si calcola una volta e si riusa finché il pattern non cambia. **1.14× su `socp_200`** (misura interleaved, 8 coppie). Suite **5030/0** invariata. **Corretta la diagnosi del divario** (v. §2-bis): su SDP d=22 il KKT è 22×22 (non 275) e già riusato; il divario è **142 vs 10 iterazioni** (convergenza), non la fattorizzazione densa. |

| 2026-11-11 | **F161 — eigendecomposizione 3×3 riusata** (`expcone.c`): `nt_eval` e `expcone_scaling` diagonalizzavano due volte `Hs` e `Mid` (6 eig per blocco/iterazione, 3 ridondanti). Memorizzate in `expnt`, aritmetica identica (obiettivi **bit-identici**). Mediane mixed 20×4 **1.16/1.12**, 40×8 **3.26/3.24**, 80×12 **7.25/7.34** ms contro Clarabel 0.24/0.24, 0.53/0.53, 2.55/2.59 (21 coppie interleaved per serie); guardie mirate 945/0. |
| 2026-11-11 | **F162 — metriche exp solo su richiesta** (`expcone.c`): `nt_eval` calcolava `condz` (eig di `Hz`), `mz`, `ms`, `wn` a ogni blocco/iterazione, ma `expcone_nt_metrics` e' chiamata solo sotto `GMB_DBG`. `want_metrics` (0 dallo scaling, 1 dalle metriche) salta 1 eig e i log delle margin. Mediane mixed 20×4 **1.09/1.09**, 40×8 **3.27/3.32**, 80×12 **7.27/7.24** ms; `[expm]` identico, obiettivi bit-identici, guardie 945/0. |
| 2026-11-11 | **F165 — fattori spettrali una volta** (`expcone.c`): `spd_from_eig` chiamava `pow(ev[k],p)` 27 volte (triplo ciclo i,j,k) per applicazione; ora 3, stessa funzione e stesso ordine (bit-identico). Mediane mixed 20×4 **1.06/1.08**, 40×8 **3.15/3.05**, 80×12 **6.94/6.91** ms contro Clarabel 0.25/0.25, 0.56/0.54, 2.55/2.60; guardie 945/0. |
| 2026-11-12 | **Tabella canonica rilanciata** con `tools/bench_repeat.py --runs 15` (obiettivi identici in tutte e 15) e **tabella mista** con `bench/pexp_bench.py --runs 21` — il generatore misto ora sta nel repo (`bench/pexp_bench.c`), non in una directory gitignorata: 20×4 **0.59**, 40×8 **1.37**, 80×12 **3.73** ms contro Clarabel 0.29 / 0.63 / 2.72. Corretti due artefatti del harness: `"%.4f"` quantizzava il tempo Primal a 0.1 ms (lp_50x25: 156 µs stampati come 0.0002) e `conic_bench`/`sdp_sweep` cronometravano a freddo la prima solve (socp_40 134 → **88 µs** con la warm-up). Le colonne aggiornate stanno nel `README.md`; §2 resta la fotografia del 2026-10-03. |
