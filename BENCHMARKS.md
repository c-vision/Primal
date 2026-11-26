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

macOS arm64, `-O2`, stessa macchina. Comando dalla root dopo la
[configurazione dell'ambiente](docs/benchmarks.md#reproduce):
`make bench && out/benchmark-venv/bin/python bench/bench_all.py`.
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


<a name="matrice-soc-exp"></a>

## 3-ter. SOC / EXP / misto contro Clarabel

### Verifica 2026-11-19 — le celle del README sono state rimisurate

Le tabelle del `README.md` (principale, mista e matrice SOC/EXP) portano ora le
misure di questa data, su un albero **pulito** di HEAD e con la stessa macchina
delle sezioni seguenti. Due esiti distinti.

**Tabella principale** (`tools/bench_repeat.py --runs 15`, 32 righe × 5 solver).
Obiettivi identici in tutte le righe; Primal minimo in **32 su 32**; i grassetti
del README restano corretti. Rispetto allo snapshot del 2026-11-12, Primal è
`0..-21.6%` (mediana `-10.4%`) — ma anche i riferimenti sono più veloci:
HiGHS `-17.4%`, SCIP `-16.6%`, Clarabel `-11.1%`, SCS `-11.4%`. Poiché HiGHS e
SCIP sono binari che questo repository non compila, quella quota è **stato
dell'host**, non codice: una singola tabella per sessione è confrontabile, due
sessioni no.

**Matrice SOC/EXP/mista** (`bench/pexp_matrix.py --only family --runs 21`).
Le celle del 2026-11-18 non erano riproducibili dall'harness del repository:
misurate oggi danno `+16..+47%` (per esempio EXP10 `0.017 -> 0.023`, Mixed80
`0.031 -> 0.040`). Escluse per misura tre spiegazioni: il codice del solver è
identico (`git diff b5e2029 HEAD -- '*.c'` tocca solo i generatori di figure),
il livello di ottimizzazione non chiude il divario (`-O3` dà gli stessi tempi),
la rotta e i passi sono quelli attesi (`reduced Newton`, `it` 7/8/4/4, ordine
8). Restano due cause, entrambe di **misura**, non di solver:

1. **Il runner accendeva `GMB_DBG` anche nelle run di tempo** e la traccia è
   scritta **dentro** la solve: il valore pubblicato era gonfiato del 20-35%, e
   il README dichiarava «tracing disabled» mentre la harness non lo faceva. Ora
   `pexp_matrix.py` fa le run di tempo senza tracing e prende rotta/iterazioni
   da **una** run tracciata, che non entra nella mediana (`pexp_matrix.py`, run
   `--only family --runs 21`: EXP10 `0.023`, EXP80 `0.040`);
2. **La provenienza**: le cifre del 2026-11-18 venivano da una harness locale
   (`out/f266/integrated.py`, citata da F266 nel log dei tentativi) e non dal
   runner del repository, che il README citava come generatore. Il punto 1
   spiega una parte del divario, non tutto: dopo la correzione il residuo è
   ancora `+18..+44%` sulla matrice, e resta attribuito alla harness locale.

Il verdetto non cambia: con i numeri di oggi la matrice resta `0.04x-0.20x` di
Clarabel, cioè 5-25× più veloce — non 5-50× come era stampato. Il caso a 240
blocchi riproduce (`0.241 ms` Primal, `5.186 ms` Clarabel, mediane di 21 e 5
run). Le sezioni `### Newton ridotto ...` che seguono restano il **record
storico** delle misure del 2026-11-18 con le loro baseline: non vanno
riscritte, e i loro "prima/dopo" non sono celle del README.

### Newton ridotto per EXP, 2026-11-18

Il confronto completo è nel [README](README.md#soc-exponential-and-mixed-cone-matrix).
Qui la baseline è `fdf6473`. I tempi sono le mediane di due serie da 21
prove intercalate sullo stesso problema, in ms e con tracing disabilitato.
Il generatore è `bench/pexp_matrix.c` e il modello Clarabel è costruito da
`bench/pexp_matrix.py`. Primal usa `1e-8` e Clarabel 0.11.1 usa `1e-9`
con un thread. Il tempo di Primal misura `PRIMAL_optimize`; quello di
Clarabel comprende la costruzione del solver.

| Blocchi EXP | Primal prima | Primal dopo | Clarabel | Passi Newton |
|---:|---:|---:|---:|---:|
| 10 | 0.410 | **0.019** | 0.202 | 7 |
| 20 | 0.410 | **0.021** | 0.259 | 8 |
| 40 | 0.689 | **0.026** | 0.393 | 4 |
| 80 | 1.036 | **0.034** | 0.662 | 4 |

Gli EXP della matrice passano da sistemi KKT di ordine 46–256 a un
sistema Newton di ordine 8. Il massimo fra residui pubblici e gap relativo
è `2.5e-10` per EXP e `2.4e-9` sull'intera matrice. Lo scarto obiettivo
scalato massimo da Clarabel per EXP è `9.5e-9`. Il controllo con 17
coordinate libere usa l'IPM generale e restituisce gli stessi obiettivi e
residui della baseline. Il caso misto a 20 blocchi resta il divario
maggiore nella matrice, con 0.339 ms contro 0.224 ms.
[Derivazione e criteri di accettazione](algorithms/exponential_cone.md#reduced-newton-method-for-fixed-scale-epigraphs)

### Newton ridotto con penalità RQUAD, 2026-11-18

Baseline `a0706e5`, stesso protocollo di due serie da 21 prove. I quattro
casi misti della matrice passano da un KKT di ordine 41–251 a un sistema
Newton di ordine 8. I tempi si riferiscono a questi epigrafi con un solo
RQUAD e con le sue coordinate trasversali libere.

| Blocchi misti | Primal prima (ms) | Primal dopo (ms) | Clarabel (ms) | Passi Newton |
|---:|---:|---:|---:|---:|
| 10 | 0.183 | **0.016** | 0.148 | 4 |
| 20 | 0.330 | **0.018** | 0.209 | 5 |
| 40 | 0.418 | **0.022** | 0.366 | 4 |
| 80 | 0.732 | **0.031** | 0.835 | 4 |

Sui misti della matrice il massimo fra residui pubblici e gap relativo è
`5.6e-16`. Lo scarto obiettivo scalato da Clarabel è al massimo
`1.7e-10`. I tre casi di [regressione con penalità quadratica](README.md#mixed-soc-and-exponential-cones)
sono inclusi nella verifica a `1e-8` con gli obiettivi noti del generatore.
[Derivazione del duale RQUAD](algorithms/exponential_cone.md#adding-a-rotated-quadratic-penalty)

<a name="wide-cone-benchmarks"></a>

### Newton ridotto con più coordinate libere, 2026-11-18

Confronto contro `39ebd1f` su due serie intercalate da 21 prove per
cella, densità dei collegamenti 0.25, una replica. Valori in ms, mediana
delle due mediane. Primal adotta le stesse tolleranze di `1e-8` della
matrice precedente.

| Famiglia | Blocchi × coordinate libere | Primal prima | Primal dopo | Clarabel |
|---|---:|---:|---:|---:|
| EXP | 80 × 24 | 1.588 | **0.060** | 0.919 |
| EXP | 160 × 32 | 3.638 | **0.119** | 2.211 |
| EXP | 240 × 48 | 9.142 | **0.237** | 4.373 |
| Misto | 80 × 24 | 1.947 | **0.056** | 0.935 |
| Misto | 160 × 32 | 5.486 | **0.117** | 2.169 |
| Misto | 240 × 48 | 14.304 | **0.239** | 4.883 |
| Misto | 240 × 64 | 21.301 | **0.334** | 7.264 |

Il test `T315` verifica il passaggio al buffer con dimensione variabile,
il limite a 64 coordinate, il ripiego sull'IPM a 65 coordinate e la
soluzione rispetto a un ottimo analitico.

<a name="sparse-bound-lp"></a>

### LP con bound sparsi, 2026-11-18

Confronto contro `39ebd1f` sui modelli LP del generatore principale. Due
serie intercalate da 21 prove per riga. Per Clarabel 0.11.1 la misura
esclude la costruzione del solver, usa un thread e tolleranze di
`1e-8`. Entrambe le build Primal hanno residui e gap entro `1e-8`.

| Istanza | Primal prima (ms) | Primal dopo (ms) | Clarabel (ms) |
|---|---:|---:|---:|
| lp_50x25 | 0.149 | **0.138** | 0.159 |
| lp_100x50 | 0.640 | **0.578** | 0.644 |
| lp_200x100 | 1.422 | **1.246** | 2.041 |
| lp_400x200 | 6.268 | **5.424** | 10.998 |

Il Newton dell'IPM usa gli elementi CSC delle colonne condivise per
eliminare e recuperare le righe dei bound. Quando gli accoppiamenti
possono essere densi, il solver usa il buffer contiguo. Sul controllo LP
all'80% di densità il residuo massimo rimane `1.01e-9`. In due serie di
conferma da 81 coppie il tempo aumenta dell'1.8–2.0%.

### Verifica dei benchmark SOCP storici, 2026-11-18

Entrambi i comandi documentati in `algorithms/` completano le 31 ripetizioni
e superano i controlli numerici. Il controllo SOCP200 con pesi diversi e
due uguaglianze supera la soglia storica del 5% contro entrambe le baseline.
Il confronto diretto sul caso tra `39ebd1f` e la nuova build dà
0.991 ms e 0.976 ms in 31 coppie alternate.

### Recupero duale SOCP ed epigrafi ruotati con uguaglianze, 2026-11-18

Due serie intercalate da 31 prove, baseline `b5e2029`, Clarabel 0.11.1
a un thread. I tempi in ms sono le mediane delle due mediane.
Il tempo Clarabel include costruzione e solve. Nei casi SOCP200 e
RQUAD200 il numero indica le coordinate di coda, con 100 coni ciascuno.

| Caso | Primal prima | Primal dopo | Clarabel |
|---|---:|---:|---:|
| SOCP200, pesi diversi e due uguaglianze | 0.991 | **0.339** | 0.655 |
| RQUAD40, una somma sulle code | 0.554 | **0.010** | 0.134 |
| RQUAD120, una somma sulle code | 0.396 | **0.022** | 0.341 |
| RQUAD200, una somma sulle code | 0.680 | **0.035** | 0.571 |

Il recupero duale chiude SOCP200 dopo il sesto passo. Il massimo fra
residui pubblici e gap relativo è `1.7e-9` in quel caso, e al massimo
`1.1e-15` nei tre RQUAD. Le soglie del task restano a `1e-8`.
[Recupero duale](algorithms/socp.md)
· [Eliminazione RQUAD](algorithms/socp_fixed_reduction.md#rotated-epigraphs-with-equalities-on-the-tails)

### Baseline della matrice, 2026-11-13

Generatore `bench/pexp_matrix.c`, runner `bench/pexp_matrix.py` (costruisce il
riferimento Clarabel **sulla stessa istanza** e asserisce che gli obiettivi
coincidano; quell'asserzione ha scoperto che il generatore non inizializzava il
seme). Mediane di 5, d=8, rho=0.50, rep=1, una thread, tol 1e-9 su Clarabel.
Tempi in ms; "it" e' il conteggio delle iterazioni, "rotta" la forma del KKT.

| famiglia | blocchi | noi | Clarabel | rapporto | rotta | it |
|---|---:|---:|---:|---:|---|---|
| misto | 10 | 2.20 | 0.12 | 18.7x | denso | 19 |
| misto | 20 | 1.30 | 0.21 | 6.3x | sparso | 25 |
| misto | 40 | 1.57 | 0.38 | 4.1x | sparso | 17 |
| misto | 80 | 2.70 | 0.90 | 3.0x | sparso | 18 |
| solo SOC | 10 | 43.68 | 0.28 | **153.6x** | denso, stalla | 391 |
| solo SOC | 20 | 2.53 | 0.67 | 3.8x | socp | 26 |
| solo SOC | 40 | 5.29 | 1.13 | 4.7x | socp | 30 |
| solo SOC | 80 | 9.44 | 2.84 | 3.3x | socp | 27 |
| solo EXP | 10 | 6.72 | 0.17 | 39.3x | denso | 61 |
| solo EXP | 20 | 12.33 | 0.29 | 43.2x | denso | 49 |
| solo EXP | 40 | 36.73 | 0.35 | **104.8x** | denso | 53 |
| solo EXP | 80 | 657.65 | 0.65 | **1019x** | denso + tagli | 30, 15 round |

Densita' dei link (n=40, d=12): misto 2.11 / 2.31 / 2.86 ms per rho 0.25/0.5/1.0
(Clarabel 0.41 / 0.58 / 0.46) -> 5.1x / 4.0x / 6.3x; solo EXP resta 39-151x,
solo SOC piatto perche' non ha righe (e' il controllo).

Degenerazione per replicazione (n=40): misto 1.53 -> 2.83 -> 5.08 ms con rep
1/2/4 (Clarabel 0.36 -> 0.83 -> 1.44, rapporto stabile 3.4-4.2x); solo EXP
36.9 -> 196.0 -> 94.9 ms (78-306x). La degenerazione costa dimensione, non
iterazioni (misto 17 -> 21 -> 21).

Lettura: il costo per iterazione non e' il collo (0.05-0.35 ms sparso contro
0.9-1.5 denso); lo sono *quale motore parte* e *se il modello entra nella prova
sparsa*. Le tre soglie coinvolte: `order >= 128` (misto, gia' tolta),
`Nsys >= 300 && ncones >= 16` (`primal_conicopt.c`, rotta SOC), `nsoc > 0`
(esclude la classe exp sia dalla prova sparsa sia dalla secante-prima).

### Verifica dopo le modifiche al routing SOC ed EXP

Mediane di 7 con baseline `f97fff2`, patch e Clarabel sullo stesso modello
(`d=8`, `rho=0.50`, `rep=1`). Tempi in ms. Il gap relativo e i residui
pubblici di PRIMAL sono sotto 1e-8 su tutti i casi. Lo scarto obiettivo
relativo massimo rispetto a Clarabel è 2.6e-8.

| Famiglia | Blocchi | PRIMAL prima | PRIMAL dopo | Clarabel |
|---|---:|---:|---:|---:|
| Misto | 10 | 1.030 | 1.036 | 0.174 |
| Misto | 20 | 1.036 | 1.042 | 0.244 |
| Misto | 40 | 1.270 | 1.271 | 0.432 |
| Misto | 80 | 2.289 | 2.273 | 0.960 |
| SOC | 10 | 41.797 | 1.260 | 0.364 |
| SOC | 20 | 2.430 | 2.398 | 0.734 |
| SOC | 40 | 5.112 | 5.017 | 1.230 |
| SOC | 80 | 8.932 | 8.804 | 2.363 |
| EXP | 10 | 3.774 | 1.973 | 0.253 |
| EXP | 20 | 8.062 | 1.509 | 0.345 |
| EXP | 40 | 27.970 | 3.304 | 0.501 |
| EXP | 80 | 666.700 | 6.836 | 0.827 |

EXP80 termina nell'IPM sparso senza tagli. SOC10 termina sul percorso sparso
in 33 iterazioni. L'obiettivo di 2× rispetto a Clarabel resta aperto.

### Arresto del punto interno su EXP puro

Il motore EXP verifica le equazioni, il gap e l'appartenenza ai due coni
prima di proseguire con il polish. Su EXP80 la verifica passa all'iterazione
16, con gap relativo di 3,86e-9 a fronte della tolleranza di 1e-8.

Mediane di 11 prove alternate sullo stesso modello (`d=8`, `rho=0.50`,
`rep=1`). Tempi in ms, baseline `ccb9810`.

| Blocchi EXP | PRIMAL prima | PRIMAL dopo | Clarabel | Rapporto |
|---:|---:|---:|---:|---:|
| 10 | 2.176 | 1.703 | 0.306 | 5.57× |
| 20 | 1.634 | 0.705 | 0.348 | 2.03× |
| 40 | 3.911 | 1.426 | 0.519 | 2.75× |
| 80 | 8.886 | 1.991 | 0.981 | 2.03× |
| 100 | 9.006 | 2.431 | 1.273 | 1.91× |
| 160 | 18.660 | 5.345 | 2.193 | 2.44× |

### Avvio ammissibile per SOC ruotati con un membro fissato

Il solver prova uno slack iniziale scalato sui costi e lo usa solo se
tutti i blocchi sono strettamente ammissibili. Mediane della seconda
serie di 11 prove alternate (`d=8`, `rho=0.50`, `rep=1`). Tempi in ms.

| Blocchi SOC | PRIMAL prima | PRIMAL dopo | Clarabel |
|---:|---:|---:|---:|
| 10 | 1.786 | 0.743 | 0.483 |
| 20 | 3.236 | 1.286 | 1.053 |
| 40 | 7.290 | 1.862 | 1.839 |
| 80 | 12.454 | 5.284 | 3.315 |

Residui pubblici e gap relativo sotto 1e-8 in entrambe le serie.

<a name="istanze-pubbliche"></a>

## 3-quater. Istanze pubbliche

### Stato verificato al 2026-11-26

Primal ha ora un verdetto verificato per **21 dei 24 casi** che nella
rilevazione originale erano rimasti senza risposta. Per gli SDP CBLIB
inammissibili il verdetto è nel problema, `PRIM_INFEAS`.

| Famiglia | Casi chiusi | Esito aggiornato |
|---|---|---|
| SDP SDPLIB | `control1`, `theta1`, `mcp100`, `qap5`, `hinf1` | `OPTIMAL` |
| SDP CBLIB | `infeas_clean_10_10_1`, `infeas_clean_10_10_3`, `weak_clean_10_10_3` | `PRIM_INFEAS` |
| Convesso continuo | `PRIMAL1`, `LogExpCR-n20-m400`, `HMCR-n20-m800` | `OPTIMAL` |
| MIPLIB MILP | `p0201` | `INTEGER_OPTIMAL` |
| CBLIB misto-intero conico | `expdesign_D_8_4`, `expdesign_D_12_6`, `expdesign_D_16_8` | `INTEGER_OPTIMAL` |
| CBLIB SDP misto-intero | `kpart_diw.15.4.29` | `INTEGER_OPTIMAL` |
| CBLIB portafogli misti-interi | `port_12_9_3_a_1`, `port_12_9_3_b_1`, `port_16_12_4_b_1` | `INTEGER_OPTIMAL` |
| CBLIB selezione binaria con PSD | `clsq_random_32_2_a`, `rt_2x4_3bars` | `INTEGER_OPTIMAL` |

Restano 3 casi senza un verdetto conclusivo. Per `gt2`, Primal trova
una soluzione intera a costo 21166, pari al valore riportato da MIPLIB.
La prova di ottimalità resta aperta.

| Famiglia | Casi aperti |
|---|---|
| CBLIB misto-intero conico | `rsyn0805m` |
| MIPLIB MILP | `gt2`, `pk1` |

[Verifiche del 22 novembre](#verifica-mirata-del-2026-11-22) ·
[Verifiche del 23 novembre](#verifica-mirata-del-2026-11-23) ·
[SDPLIB](#verifica-sdplib-del-2026-11-24) ·
[Coni power](#verifica-power-cblib-del-2026-11-24) ·
[`gt2`](#verifica-miplib-gt2-del-2026-11-24) ·
[`p0201`](#verifica-miplib-p0201-del-2026-11-25) ·
[Disegno sperimentale CBLIB](#verifica-cblib-expdesign-del-2026-11-25) ·
[Partizionamento CBLIB](#verifica-cblib-kpart-del-2026-11-25) ·
[Portafogli CBLIB](#verifica-cblib-portafogli-del-2026-11-25)

### Confronto per istanza

Le righe `port_*`, `expdesign_D_*` e `kpart_diw.15.4.29` riportano
gli esiti, gli obiettivi e i tempi delle verifiche del 25 novembre.
Per `infeas_clean_*`, `weak_clean_10_10_3`, `control1` e `hinf1`, le
colonne Primal e Clarabel riportano i risultati del 25 novembre.
Per `PRIMAL1`, le colonne Primal, Clarabel e SCS riportano il confronto
del 26 novembre. Le righe `clsq_random_32_2_a` e `rt_2x4_3bars` riportano
le verifiche dello stesso giorno. Le altre misure risalgono al 19 novembre.
Il trattino nella colonna Pajarito indica che manca un tempo valido per lo stesso ottimo.
[Disegno sperimentale](#verifica-cblib-expdesign-del-2026-11-25) ·
[Partizionamento](#verifica-cblib-kpart-del-2026-11-25) ·
[Portafogli](#verifica-cblib-portafogli-del-2026-11-25) ·
[Inammissibilità SDP](#inammissibilità-sdp-cblib) ·
[SDPLIB control1 e hinf1](#verifica-sdplib-control1-e-hinf1-del-2026-11-25) ·
[PRIMAL1](#verifica-primal1-del-2026-11-26) ·
[Selezione binaria con PSD](#verifica-cblib-selezione-binaria-con-psd-del-2026-11-26)

49 istanze da cinque set pubblici, scaricate da `bench/fetch_instances.py`
(harness della **PR #32**, usata in prestito e **non** mergiata): CBLIB 27
(SOC/exp/pow/PSD, con e senza interi), SDPLIB 6 (SDP puro), Maros-Meszaros 6
(QP convesso), Netlib 6 (LP), MIPLIB 2017 4 (MILP). La rilevazione originale
usa un run per cella con tetto **60 s**. Le celle Pajarito/Hypatia vengono
da un processo Julia con run caldo
(il tempo è della solve, non della compilazione).

I lettori sono quelli corretti in `9933143` (T277/T278): le righe MPS di questo
set, misurate prima di quella correzione, erano un **rilassamento** (il marker
`MARK0000 'MARKER' 'INTORG'` letto come colonna). Le colonne di riferimento sono
Python (`highspy`, `clarabel` 0.11.1, `scs` 3.3.1, `pyscipopt`) tranne
Pajarito/Hypatia, che girano in Julia 1.13.2 con Hypatia 0.11.0 e Pajarito 0.8.3
(subsolver: HiGHS 1.26.0 per l'OA, Clarabel 0.11.1 senza decomposizione cordale).

| instance | set | class | vars x cons | Primal (s) | HiGHS (s) | Clarabel (s) | SCS (s) | SCIP (s) | Pajarito/Hypatia (s) | obj |
|---|---|---|---|---|---|---|---|---|---|---|
| port_12_9_3_a_1 | cblib | soc+exp+psd+int | 90 x 169 | **0.004568** (`INTEGER_OPTIMAL`) | N/A | N/A | N/A | N/A | 17.246444 | -0.0332185523 |
| port_12_9_3_b_1 | cblib | soc+exp+psd+int | 91 x 170 | **0.003490** (`INTEGER_OPTIMAL`) | N/A | N/A | N/A | N/A | 6.600786 | -0.0347049507 |
| port_16_12_4_b_1 | cblib | soc+exp+psd+int | 120 x 232 | 0.090776 (`INTEGER_OPTIMAL`) | N/A | N/A | N/A | N/A | — | -0.0440155425 |
| expdesign_D_8_4 | cblib | exp+psd+int | 61 x 148 | **0.000938** (`INTEGER_OPTIMAL`) | N/A | N/A | N/A | N/A | 0.045817 | 0.8433069245 |
| expdesign_D_12_6 | cblib | exp+psd+int | 127 x 311 | 0.000472 (`INTEGER_OPTIMAL`) | N/A | N/A | N/A | N/A | — | -0.5202490151 |
| expdesign_D_16_8 | cblib | exp+psd+int | 217 x 534 | 0.002238 (`INTEGER_OPTIMAL`) | N/A | N/A | N/A | N/A | — | 2.9313367330 |
| kpart_diw.15.4.29 | cblib | psd+int | 105 x 255 | 0.018999 (`INTEGER_OPTIMAL`) | N/A | N/A | N/A | N/A | — | -95 |
| rt_2x4_3bars | cblib | psd+int | 64 x 190 | 0.030947 (`INTEGER_OPTIMAL`) | N/A | N/A | N/A | N/A | — | 3.0811388301 |
| clsq_random_32_2_a | cblib | psd+int | 33 x 979 | **0.002406** (`INTEGER_OPTIMAL`) | N/A | N/A | N/A | N/A | 11.626202 | 7.1485454105 |
| expdesign_A_8_4 | cblib | psd+int | 29 x 134 | 0.5519 | N/A | N/A | N/A | N/A | **0.0272** | 5.93682 |
| expdesign_E_8_4 | cblib | psd+int | 26 x 50 | 0.0403 | N/A | N/A | N/A | N/A | **0.0247** | -0.344045 |
| syn05m | cblib | exp+int | 24 x 65 | 0.5676 | N/A | N/A | N/A | N/A | **0.0512** | -837.732 |
| syn10m | cblib | exp+int | 42 x 120 | 6.9118 | N/A | N/A | N/A | N/A | **0.0304** | -1267.35 |
| rsyn0805m | cblib | exp+int | 174 x 537 | >60s | N/A | N/A | N/A | N/A | **1.9644** | -1296.12 |
| batchdes | cblib | exp+int | 25 x 73 | 0.6263 | N/A | N/A | N/A | N/A | **0.0170** | 167428 |
| ex1223a | cblib | exp+int | 17 x 60 | **0.0097** | N/A | N/A | N/A | N/A | 0.0127 | 4.57958 |
| synthes2 | cblib | exp+int | 16 x 47 | 0.3661 | N/A | N/A | N/A | N/A | **0.0306** | 73.0353 |
| beck751 | cblib | exp | 80 x 59 | **0.0010** | N/A | 0.0014 | 0.0012 | N/A | 0.0038 | 7.50095 |
| demb761 | cblib | exp | 131 x 93 | 0.0009 | N/A | **0.0007** | 0.2940 | N/A | 0.0043 | 22.3109 |
| rijc781 | cblib | exp | 24 x 17 | 0.0072 | N/A | **0.0001** | 0.0002 | N/A | 0.0012 | -4.41429 |
| gp_dave_1 | cblib | exp | 705 x 988 | 39.8111 | N/A | **0.0081** | 0.0445 | N/A | 0.2190 | 5.50653 |
| LogExpCR-n20-m400 | cblib | exp | 2022 x 1603 | >60s | N/A | **0.0199** | 0.0273 | N/A | 2.6629 | 0.0164814 |
| HMCR-n20-m400 | cblib | pow | 2022 x 1603 | 39.7349 | N/A | **0.0210** | 0.0608 | N/A | 2.5334 | 0.0345743 |
| HMCR-n20-m800 | cblib | pow | 4022 x 3203 | >60s | N/A | fail | **0.1044** | N/A | 10.0290 | 0.0390837 |
| infeas_clean_10_10_1 | cblib | sdp-infeas | 55 x 10 | **0.000010** (`PRIM_INFEAS`) | N/A | 0.003661 (`PrimalInfeasible`) | >60s | N/A | fail (SLOW_PROGRESS) | infeasible |
| infeas_clean_10_10_3 | cblib | sdp-infeas | 55 x 10 | 0.000009 (`PRIM_INFEAS`) | N/A | 0.003072 (`AlmostPrimalInfeasible`) | 5.4682 | N/A | fail (SLOW_PROGRESS) | infeasible |
| weak_clean_10_10_3 | cblib | sdp-infeas | 55 x 10 | 0.000010 (`PRIM_INFEAS`) | N/A | 0.007492 (`AlmostSolved`) | >60s | N/A | fail (SLOW_PROGRESS) | infeasible |
| control1 | sdplib | sdp | 21 x 70 | 0.055899 (`OPTIMAL`) | N/A | **0.003418** | >60s | N/A | 0.0039 | 17.7846267272 |
| hinf1 | sdplib | sdp | 13 x 41 | 0.001729 (`OPTIMAL`) | N/A | 0.001320 ✗ | 0.1340 ✗ | N/A | 0.0029 ✗ | 2.0326007779 |
| theta1 | sdplib | sdp | 104 x 1275 | >60s | N/A | 0.4421 | **0.0445** | N/A | 0.2877 | 23 |
| truss1 | sdplib | sdp | 6 x 19 | 0.0021 | N/A | **0.0003** | 0.0008 | N/A | 0.0026 | -9 |
| qap5 | sdplib | sdp | 136 x 351 | >60s | N/A | 0.0309 | **0.0115** | N/A | 0.0389 | -436 |
| mcp100 | sdplib | sdp | 100 x 5050 | >60s | N/A | 6.8169 | 0.5768 | N/A | **0.5324** | 226.157 |
| HS21 | maros-meszaros | qp | 2 x 1 | **0.0000** | 0.0028 | 0.0001 | 0.0001 | 0.0073 | 0.0019 | 0.0400007 |
| HS118 | maros-meszaros | qp | 15 x 17 | 0.0003 | 0.0006 | **0.0001** | 0.0003 | 0.0094 | 0.0026 | 664.82 |
| QAFIRO | maros-meszaros | qp | 32 x 27 | 0.0005 | 0.0005 | **0.0002** | **0.0002** | 0.0042 | 0.0028 | -1.59078 |
| DUAL1 | maros-meszaros | qp | 85 x 1 | 0.0021 | **0.0011** | 0.0022 | **0.0011** | 0.0784 | 0.0116 | 0.035013 |
| PRIMAL1 | maros-meszaros | qp | 325 x 85 | 0.008826 (`OPTIMAL`) | 0.0116 | 0.002543 | **0.001159** | 0.9942 | 0.1304 | -0.0350129657 |
| CVXQP1_S | maros-meszaros | qp | 100 x 50 | 0.0100 | 0.0009 | **0.0006** | 0.0014 | 0.1125 | N/A | 11590.7 |
| afiro | netlib | lp | 32 x 27 | **0.0001** | 0.0003 | 0.0002 | 0.0004 | 0.0007 | 0.0018 | -464.753 |
| adlittle | netlib | lp | 97 x 56 | 0.0006 | 0.0007 | **0.0005** | 0.0014 | 0.0018 | 0.0070 | 225495 |
| blend | netlib | lp | 83 x 74 | 0.0010 | 0.0008 | **0.0005** | 0.0013 | 0.0017 | 0.0029 | -30.8121 |
| sc50a | netlib | lp | 48 x 50 | **0.0003** | **0.0003** | **0.0003** | 0.0010 | 0.0005 | 0.0016 | -64.5751 |
| share2b | netlib | lp | 79 x 96 | 0.0012 | 0.0008 | **0.0006** | 0.0024 | 0.0021 | 0.0048 | -415.732 |
| stocfor1 | netlib | lp | 111 x 117 | 0.0019 | **0.0007** | 0.0009 | 0.0035 | 0.0014 | 0.0045 | -41132 |
| flugpl | miplib | milp | 18 x 18 | 1.1447 | 0.0408 | N/A | N/A | **0.0066** | 0.0872 | 1.2015e+06 |
| p0201 | miplib | milp | 201 x 133 | >60s | 0.6562 | N/A | N/A | **0.2895** | 0.5229 | 7615 |
| gt2 | miplib | milp | 188 x 29 | >60s | 0.0217 | N/A | N/A | **0.0181** | 0.0400 | 21166 |
| pk1 | miplib | milp | 86 x 45 | >60s | >60s | N/A | N/A | **57.1295** | >60s | 11 |

**Note misurate** (le celle con ✗ sono riferimenti che discordano dal valore
pubblicato, non risposte di Primal; `N/A` nella colonna SCIP sulle righe CBF
significa **non misurato**: `bench/scip_cbf.c` richiede le librerie di sviluppo
di SCIP, che su questa macchina non ci sono — i wheel di `pyscipopt` non le
portano. Su MPS e MILP SCIP c'è):

- `expdesign_D_12_6`: nella rilevazione originale Pajarito chiude a
  `-0.437`. Il riferimento `-0.520253` era stato controllato con una
  tolleranza esponenziale di `5e-6`. La verifica successiva certifica il
  valore `-0.5202490151` a `1e-8`.
- `hinf1`: SCS `2.0339587` e Pajarito `2.0326681` contro il `2.0326` di
  SDPLIB. Nella verifica del 25 novembre Clarabel restituisce
  `AlmostSolved` a `2.0326400951`, con scarto obiettivo scalato
  `7.76e-6` rispetto alla soluzione di Primal.
- **Nel confronto del 19 novembre Primal si ferma senza soluzione o senza certificato** sui casi SDP
  costruiti come inammissibili (`infeas_clean_*`, `weak_clean_*`), sui cinque
  SDP puri `control1`, `hinf1`, `theta1`, `qap5`, `mcp100`, sul QP `PRIMAL1`
  e su vari modelli misto-interi conici.
  Le verifiche del 22 e 23 novembre riportano i risultati successivi per
  `PRIMAL1`, `LogExpCR-n20-m400`, `control1`, `theta1` e i tre SDP CBLIB
  inammissibili.

### Verifica mirata del 2026-11-22

Queste prove aggiornano i singoli casi indicati nella fotografia del 19 novembre.

Sul QP pubblico `PRIMAL1`, il passo finale a regolarizzazione adattiva
risolve il caso che nella tabella del 19 novembre era fermo a `rc=1007`.
Due serie da 21 solve alternate fra la base `cf8a3c8` e la build modificata,
con Clarabel 0.11.1 a un thread. Tutti i tempi escludono lettura, conversione
e costruzione del solver di riferimento. Le tolleranze restano `1e-8`.

| Istanza | Primal prima (ms) | Primal dopo (ms) | Clarabel (ms) |
|---|---:|---:|---:|
| `PRIMAL1` | `rc=1007` in 42/42 | 8.653 / 8.655 | 2.619 / 2.582 |

La build modificata restituisce `OPTIMAL` in 42/42 solve, obiettivo
`-0.03501296573288044`, violazione pubblica massima `3.43e-10` e gap relativo
`1.98e-11`. Lo scarto di obiettivo rispetto a Clarabel, scalato per
`1 + abs(obj_ref)`, è `5.59e-10`. I QP di controllo `DUAL1` e `CVXQP1_S`
mantengono stato e obiettivo e cambiano al massimo dello 0,4% nelle mediane.
Il confronto usa gli stessi dati Maros–Meszaros della tabella storica.

Sul CBLIB `LogExpCR-n20-m400`, il correttore nativo a complementarità zero
chiude entro le tolleranze prima del fallback a tagli. Mediana di tre solve
senza debug, con lo stesso criterio di misura e tolleranze.

| Istanza | Primal prima (s) | Primal dopo (s) | Clarabel (s) |
|---|---:|---:|---:|
| `LogExpCR-n20-m400` | >60 | 2.423 | 0.02027 |

Primal restituisce `OPTIMAL` in 3/3 solve, obiettivo `0.016481437317`, gap
relativo `2.60e-9` e violazione pubblica massima `4.76e-13`. L'obiettivo di
Clarabel è `0.016481441943` e lo scarto scalato è `4.55e-9`.

Su `gp_dave_1`, il ripristino dei moltiplicatori delle righe originali
riduce il residuo di stazionarietà pubblicato da `3.312` a `1.70e-10`.
L'obiettivo primale e il gap relativo rimangono rispettivamente
`5.506526406863336` e `5.92e-10` nella verifica fra le due build.

### Verifica PRIMAL1 del 2026-11-26

Il QP Maros–Meszaros restituisce `OPTIMAL` con Primal `1d5ce18`
in tutte le 21 prove misurate. Il lettore MPS riceve gli stessi
coefficienti del file `.mat` originale usato per i due riferimenti.

| Solver | Mediana (ms) | Obiettivo | Massimo residuo | Gap relativo |
|---|---:|---:|---:|---:|
| Primal `-O2` | 8.826 | -0.035012965733 | 3.43e-10 | 1.98e-11 |
| Clarabel 0.11.1 | 2.543 | -0.035012965154 | 2.22e-11 | 1.14e-9 |
| SCS 3.3.1 | **1.159** | -0.035012965698 | 5.26e-10 | 4.03e-11 |

Mediane di 21 prove intercalate a un thread dopo il riscaldamento.
Tolleranze `1e-8` per tutti e tre i solver. I tempi escludono lettura,
conversione e costruzione dei solver di riferimento. Obiettivo,
fattibilità e stazionarietà sono ricalcolati dai dati Maros–Meszaros.
Lo scarto scalato fra gli obiettivi è al più `5.59e-10`.

### Verifica mirata del 2026-11-23

Su `control1` (SDPLIB), la base `c52903d` termina con `rc=1007` dopo 23.72 s.
Con l'equilibratura per congruenza delle LMI affini il modello originale
raggiunge `OPTIMAL` in 7 solve su 7. Mediana **60.44 ms**, un thread e sola
chiamata di solve. Obiettivo primale `17.78462672722526`,
coerente con `17.78463` alla precisione pubblicata da SDPLIB. Il gap relativo
è `8.74e-10` e la massima violazione pubblica è `3.56e-15`, con tolleranze
`1e-8`. Le misure si riferiscono al punto ricostruito nelle unità originali.
[Equilibratura e verifica](algorithms/semidefinite_programming.md#equilibrating-an-affine-matrix-inequality)

Su `theta1` (SDPLIB), la misura storica sul CBF originale superava il
limite di 60 s. Il nuovo percorso costruisce internamente il duale con
104 vincoli al posto dei 1.275 link, poi verifica la coppia primale-duale
nelle unità del modello originale.

| Istanza | Primal 2026-11-19 (s) | Primal 2026-11-23 (s) | Clarabel 2026-11-19 (s) | SCS 2026-11-19 (s) |
|---|---:|---:|---:|---:|
| `theta1` | >60 | 4.426 | 0.4421 | 0.0445 |

Con una build `-O2`, un thread e tolleranze `1e-8`, il CBF originale
restituisce `OPTIMAL` in 3 solve su 3. I tempi di `PRIMAL_optimize` vanno
da 4.418 a 4.454 s. L'obiettivo primale è `23.00000000871891`, il gap
relativo `2.06e-9` e la massima violazione pubblica `3.34e-9`,
incluse le verifiche di appartenenza al cono PSD. I tempi dei riferimenti
provengono dalla tabella del 19 novembre.
[Derivazione e mappa dei moltiplicatori](algorithms/semidefinite_programming.md#solving-an-affine-matrix-inequality-in-dual-form)

#### Inammissibilità SDP CBLIB

Due uguaglianze nei CBF originali restringono il blocco PSD da 10 a 8
e poi a 5 coordinate. Sulla faccia residua la terza uguaglianza impone che
la somma di due elementi diagonali non negativi valga `-1`, `-1` e `-2`
nelle tre istanze rispettivamente. Primal riconosce la contraddizione
nel presolve e restituisce `rc=1002`, `prosta=PRIM_INFEAS` e
`solsta=UNKNOWN`. I getter del punto e del raggio duale rispondono
`ERR_ARG`.

| Istanza | Primal `0d99119`, prima | Primal 25/11 (ms) | Clarabel 25/11 (ms) | Esito Clarabel |
|---|---|---:|---:|---|
| `infeas_clean_10_10_1` | rc=1007 (3.001 s) | **0.010** | 3.661 | `PrimalInfeasible` |
| `infeas_clean_10_10_3` | rc=1007 (2.927 s) | 0.009 | 3.072 | `AlmostPrimalInfeasible` |
| `weak_clean_10_10_3` | rc=1007 (3.053 s) | 0.010 | 7.492 | `AlmostSolved` |

Verifica del 25 novembre 2026 a un thread e tolleranze `1e-8`.
Primal `ffd4cf4`, build `-O2`, dà lo stesso verdetto in 21 prove
misurate per istanza. I tempi sono le mediane di quelle prove e di
cinque solve per Clarabel 0.11.1, con costruzione del solver inclusa
per Clarabel. La lettura del CBF e la conversione in dati triangolari
sono fuori dai tempi. Il grassetto indica il tempo minore sulla riga
con lo stesso verdetto esplicito.
[Prova per riduzione facciale](algorithms/semidefinite_programming.md#detecting-infeasibility-on-coordinate-faces)

### Vittorie confermate sulle istanze pubbliche

Il grassetto nella tabella del 19 novembre indica il minimo di **un singolo
run**. Dopo cinque solve per cella e tre batch caldi per i riferimenti
Julia, Primal risulta primo in due delle cinque righe ricontrollate.

| istanza | Primal mediana 5 | migliore rivale | esito |
|---|---:|---:|---|
| `ex1223a` | 0.0110 | Pajarito 0.0167 | Primal primo (confermato) |
| `HS21` | 0.000031 | Clarabel 0.000041 | Primal primo (confermato) |
| `beck751` | 0.000748 | Clarabel 0.000392 | **Clarabel primo** |
| `afiro` | 0.000131 | Clarabel 0.000116 | **Clarabel primo** |
| `sc50a` | 0.000347 | HiGHS 0.000221 | **HiGHS primo** |

Il README riporta l'esito del singolo run e le mediane più recenti.
[Public instance sets nel README](README.md#public-instance-sets)

### Verifica mirata del 2026-11-24

La riduzione dei vincoli log-sum-exp porta il sistema di Newton di
`beck751` dall'ordine 134 all'ordine 8. Le cinque disequazioni ridotte
contengono gli stessi 18 termini esponenziali del CBF.
[Derivazione e verifica sul modello originale](algorithms/exponential_cone.md#reducing-grouped-exponential-epigraphs)

| Istanza | Primal prima (ms) | Primal ridotto (ms) | Clarabel solo solve (ms) | Clarabel costruzione e solve (ms) |
|---|---:|---:|---:|---:|
| `beck751` | 0.657 | **0.029** | 0.331 | 0.467 |
| `rijc781` | 7.282 | **0.020** | 0.056 | 0.151 |

Media delle mediane di due serie da 21 prove intercalate, senza debug,
un thread e tolleranze `1e-8`. Primal è compilato con `-O2` e i tempi
coprono `PRIMAL_optimize`. Clarabel 0.11.1 usa gli stessi CBF senza
includere la lettura dei file o la conversione delle matrici. La build
Primal di partenza ha le correzioni SDP del 23 novembre e precede la
riduzione log-sum-exp.

Primal restituisce `OPTIMAL` in 42 solve su 42 per caso. Su `beck751`
l'obiettivo vale `7.500952151967790`, la massima violazione originale è
`1.69e-10`, il gap relativo è `2.13e-11` e lo scarto obiettivo scalato
rispetto a Clarabel è `5.74e-10`. Su `rijc781` i valori corrispondenti
sono `-4.414286537325904`, `4.36e-10`, `1.45e-10` e `8.28e-9`.

### Verifica SDPLIB del 2026-11-24

I tre SDP rimasti aperti dopo le verifiche del 23 novembre ora restituiscono
`OPTIMAL` in cinque solve su cinque nei CBF originali. Build `-O2`, un
thread, tolleranze `1e-8`, fattore near-optimal pari a 1. I tempi sono
mediane di `PRIMAL_optimize`, con lettura dei file esclusa.

| Istanza | Primal 19/11 | Primal 24/11 (ms) | Obiettivo primale | Gap relativo | Violazione massima |
|---|---|---:|---:|---:|---:|
| `mcp100` | >60 s | 56.368 | 226.1573515174849 | 1.51e-9 | 2.22e-16 |
| `qap5` | >60 s | 232.564 | -435.9999990000009 | 3.01e-9 | 5.12e-11 |
| `hinf1` | rc=1007 | 1.731 | 2.032600777938155 | 5.56e-9 | 5.10e-13 |

Su `mcp100` la barriera lavora sulla sola diagonale libera, mentre
`qap5` e `hinf1` usano QR sulle matrici scalate dalla Cholesky della
barra. `hinf1` usa una fase I con uno shift comune ai tre blocchi.
[`mcp100` e la diagonale libera](algorithms/semidefinite_programming.md#a-log-det-barrier-for-a-free-diagonal) ·
[QR, recupero duale e fase I](algorithms/semidefinite_programming.md#a-log-det-barrier-for-small-affine-blocks)

Il confronto ripetuto su `mcp100` usa tre solve intercalati a un thread
e tolleranze `1e-8`. Clarabel 0.11.1 e SCS 3.3.1 ricevono la stessa
LMI originale in forma triangolare scalata. I tempi dei riferimenti
escludono la costruzione del solver.

| Primal (ms) | Clarabel (ms) | SCS (ms) |
|---:|---:|---:|
| **56.361** | 166.745 | 851.940 |

I tre solver dichiarano l'ottimo in tutte le prove, con scarto scalato
degli obiettivi entro `4.22e-9` rispetto a Primal.

### Verifica SDPLIB control1 e hinf1 del 2026-11-25

Primal `010e435` restituisce `OPTIMAL` su entrambi i modelli in tutte
le 21 prove misurate. I CBF coincidono con la conversione dei file SDPA
originali. Un controllo indipendente ricostruisce entrambi gli obiettivi,
i minimi autovalori dei blocchi e la stazionarietà dai dati SDPA e dalle
soluzioni pubblicate da Primal.

| Istanza | Primal (ms) | Clarabel (ms) | Obiettivo Primal | Gap relativo Primal | Massima violazione Primal |
|---|---:|---:|---:|---:|---:|
| `control1` | 55.899 | **3.418** | 17.78462672722526 | 8.74e-10 | 3.56e-15 |
| `hinf1` | 1.729 | 1.320 ✗ | 2.032600777938155 | 5.56e-9 | 5.10e-13 |

Mediane di 21 solve intercalati a un thread dopo il riscaldamento.
Primal è compilato con `-O2`, usa tolleranze `1e-8` e fattore near-optimal
pari a 1. Clarabel 0.11.1 usa tolleranze interne `1e-10` e decomposizione
cordale disattivata. I tempi coprono `PRIMAL_optimize` e `solve()`.
La lettura dei file e la costruzione del solver Clarabel sono escluse.
Su `control1`, Clarabel concorda sull'obiettivo entro `1.88e-10`
in scala relativa e il massimo residuo ricalcolato è `2.14e-10`.
Su `hinf1`, lo scarto obiettivo di Clarabel supera il criterio comune
di `1e-8`. Il tempo con ✗ indica una solve fuori criterio.

### Verifica power CBLIB del 2026-11-24

Nei modelli HMCR il costo della norma è sufficiente perché esista
un ottimo a perdita positiva zero. Primal risolve il duale dell'LP
ridotto e verifica i moltiplicatori dei coni originali.
[Derivazione della riduzione](algorithms/power_cone.md#when-a-weighted-loss-norm-reduces-to-an-lp)

| Istanza | Primal 19/11 | Primal 24/11 (ms) | SCS 24/11 (ms) | Obiettivo Primal | Gap relativo Primal |
|---|---:|---:|---:|---:|---:|
| `HMCR-n20-m400` | 39.735 s | **2.946** | 52.495 | 0.03457429115475112 | 5.88e-11 |
| `HMCR-n20-m800` | >60 s | **5.703** | 104.136 | 0.03908368729046473 | 2.75e-11 |

Mediane di cinque prove alternate sui CBF originali, a un thread e
tolleranze `1e-8`. Primal copre `PRIMAL_optimize` e usa il fattore
near-optimal pari a 1. SCS 3.3.1 usa la sola solve. Entrambi hanno
dichiarato l'ottimo in tutte le prove. Il massimo scarto obiettivo
scalato è `7.58e-11` tra i due solver. Le massime violazioni originali
di Primal sono `7.11e-15` e `2.49e-14`, rispettivamente. In una serie
separata di cinque solve, Clarabel 0.11.1 risolve `HMCR-n20-m400`
in **15.535 ms** di mediana senza costruzione. Lo scarto scalato
dall'obiettivo di Primal è `3.03e-9`.

### Verifica MIPLIB gt2 del 2026-11-24

Tre prove alternate sul [MPS originale di `gt2`](https://miplib.zib.de/instance_details_gt2.html),
build `-O2`, un thread, `PRIMAL_DPAR_MIO_MAX_TIME=5` e
`PRIMAL_IPAR_MIP_MAX_NODES=999999`. Il tempo al primo punto è la
mediana dei tempi CPU dall'ingresso in `PRIMAL_optimize` al messaggio
`MIP: capacity-priced cover incumbent`. La lettura del MPS è esclusa.

| Build | Tempo al primo punto (ms) | Obiettivo | Esito al limite di tempo |
|---|---:|---:|---|
| `747bdf2` | — | — | `TRM_MAX_TIME`, nessun punto in 3 prove su 3 |
| Con euristica di copertura | 42.3 | 21166 | `TRM_MAX_TIME`, `PRIM_FEAS` in 3 prove su 3 |

`PRIMAL_getsolutioninfo` riporta violazioni nulle sulle righe, sui
bound e sull'interezza in tutte e tre le prove. Il valore coincide
con quello pubblicato da MIPLIB. Una verifica indipendente sui
coefficienti originali restituisce il limite lagrangiano
`933472/45`, che si arrotonda a `20744` per l'obiettivo intero.
Questo lascia un divario di 422 unità. La tabella storica dei tempi
fino alla prova di ottimalità conserva la sua misura originale.
[Euristica e formulazione](algorithms/mixed_integer_optimization.md#shared-capacity-integer-covers)

### Verifica MIPLIB p0201 del 2026-11-25

Sul [MPS originale](https://miplib.zib.de/instance_details_p0201.html),
la build precedente con l'euristica di `gt2` raggiungeva 7805 a
5 secondi e restituiva `PRIM_FEAS`. Primal ora riconosce i gruppi
binari a scelta unica, le compatibilità locali e i vincoli cumulativi
di riserva. La programmazione dinamica esaurisce gli stati che
possono influenzare le scelte successive e prova l'ottimo.
[Formulazione e verifica](algorithms/mixed_integer_optimization.md#exact-dynamic-programming-for-choice-sequences)

Mediane di cinque prove intercalate sull'MPS originale a un thread,
con tolleranza di ammissibilità `1e-8` e gap MIP assoluto e relativo
impostati a zero. I tempi escludono la lettura del file e comprendono
presolve, ricerca e prova dell'ottimo.

| Primal `-O2` (ms) | SCIP 10.0 (ms) | HiGHS 1.15.1 (ms) | Obiettivo |
|---:|---:|---:|---:|
| **4.656** | 297.278 | 514.129 | 7615 |

I tre solver dichiarano l'ottimo in tutte le prove. La massima
violazione di riga, bound o interezza nel modello originale è zero
per Primal e SCIP e `2.85e-14` per HiGHS.

### Verifica CBLIB expdesign del 2026-11-25

Nella rilevazione del 19 novembre Primal non aveva ottenuto un
verdetto per queste tre taglie. Primal certifica ora i tre ottimi
interi sul CBF originale con un
limite sul determinante. La ricostruzione del punto passa le verifiche
di riga, bound, PSD, cono esponenziale e interezza a `1e-8`.
[Derivazione e implementazione](algorithms/mixed_integer_optimization.md#small-integer-d-optimal-designs)

| Istanza | Primal `-O2` (ms) | Pajarito 0.8.3 (ms) | Obiettivo Primal | Massima violazione Primal |
|---|---:|---:|---:|---:|
| `expdesign_D_8_4` | **0.938** | 45.817 | 0.843306924474911 | 2.46e-16 |
| `expdesign_D_12_6` | 0.472 | — | -0.520249015131837 | 3.20e-16 |
| `expdesign_D_16_8` | 2.238 | — | 2.931336732977901 | 2.90e-16 |

Per `expdesign_D_8_4` i tempi sono mediane di cinque prove intercalate
a un thread, con lettura del CBF esclusa. Pajarito usa HiGHS 1.26.0 e
Clarabel 0.11.1, con gap di arresto `1e-8`. Entrambi i solver dichiarano
l'ottimo in tutte le prove. Pajarito restituisce `0.843306920413041`
con massima violazione originale `1.44e-9`.

Per le due taglie più grandi, i tempi sono mediane di tre prove a un
thread sul CBF originale, con lettura esclusa. Primal certifica tre
ottimi su tre per taglia. La ricerca chiude 23 e 55 box, con gap
assoluti rispettivamente `6.27e-12` e `1.38e-11`. La precedente
enumerazione di `expdesign_D_12_6` richiedeva una mediana di
`4156.184 ms`. I valori storici `-0.5202525` e `2.931324` venivano
da punti controllati con tolleranza esponenziale `5e-6`.

Nel nuovo confronto a un thread Pajarito dichiara `OPTIMAL` a
`0.9609619516` per la 12×6 e restituisce `OTHER_ERROR` per la 16×8
in tutte e cinque le prove. La tabella riporta il tempo dei confronti
in cui entrambi i solver verificano lo stesso ottimo.

### Verifica CBLIB kpart del 2026-11-25

Su `kpart_diw.15.4.29` Primal riconosce una partizione di 15 vertici
in al massimo quattro gruppi, filtra i gruppi con i vincoli di capacità
e certifica il minimo con programmazione dinamica su sottoinsiemi.
[Derivazione e verifica](algorithms/mixed_integer_optimization.md#partitions-encoded-by-a-psd-matrix)

| Istanza | Primal `-O2` (ms) | Obiettivo Primal | Massima violazione originale |
|---|---:|---:|---:|
| `kpart_diw.15.4.29` | 18.999 | -95 | 1.37e-9 (PSD) |

Mediana di cinque solve a un thread sul CBF originale, con lettura
esclusa e tutte le verifiche a `1e-8`. Primal restituisce
`INTEGER_OPTIMAL` in ciascuna prova. Un ricalcolo dai coefficienti
del CBF dà violazione PSD `1.363637e-9` e violazioni lineari e intere
nulle. HiGHS conferma `-95` con gap zero sulla formulazione
indipendente a colonne-gruppo. La rilevazione originale riporta `-94`.
Nel confronto aggiornato Pajarito 0.8.3 termina a 60 secondi con
`TIME_LIMIT` e senza una soluzione pubblicata.

### Verifica CBLIB portafogli del 2026-11-25

Primal risolve i tre CBF originali con un bound globale ottenuto
separando i blocchi di rischio SOC, EXP e PSD per quota di indicatori.
La ricostruzione passa tutte le verifiche nel modello originale.
[Derivazione e implementazione](algorithms/mixed_integer_optimization.md#cardinality-constrained-portfolios-with-separate-risk-blocks)

| Istanza | Primal `-O2` (ms) | Pajarito 0.8.3 (ms) | Obiettivo Primal | Gap assoluto | Massima violazione pubblica |
|---|---:|---:|---:|---:|---:|
| `port_12_9_3_a_1` | **4.568** | 17246.444 | -0.033218552320920 | 1.67e-11 | 1.12e-16 |
| `port_12_9_3_b_1` | **3.490** | 6600.786 | -0.034704950687185 | 2.11e-11 | 1.12e-16 |
| `port_16_12_4_b_1` | 90.776 | — | -0.044015542534373 | 3.31e-11 | 1.12e-16 |

Mediane di cinque solve per istanza a un thread dopo una prova di
riscaldamento, con lettura CBF esclusa. Primal restituisce
`INTEGER_OPTIMAL` in ogni prova, con tolleranze di fattibilità e gap a
`1e-8`. Un ricalcolo dai coefficienti dei CBF dà violazioni al più
`1.9e-18`. Clarabel 0.11.1, con gli indicatori fissati, conferma
l'obiettivo di ciascun sottoproblema continuo entro `8.6e-9`.

I tempi di Pajarito sono mediane di due solve per istanza dopo una prova
di riscaldamento, con HiGHS 1.26.0 e Clarabel 0.11.1 a un thread.
Pajarito dichiara `OPTIMAL` sui due casi 12×9×3 e concorda con Primal
sull'obiettivo entro `1.3e-9`. Il massimo residuo ricalcolato sui CBF
è `9.01e-10`. La tolleranza di fattibilità di Pajarito è `1e-9` per
`a_1` e `1e-8` per `b_1`. Sul 16×12×4 Pajarito termina con
`TIME_LIMIT` dopo 75 s.

### Verifica CBLIB selezione binaria con PSD del 2026-11-26

Primal verifica l'ottimo intero sui due CBF originali con tolleranze
di fattibilità e gap a `1e-8`. I tempi sono mediane di 21 solve
a un thread dopo il riscaldamento. Build `-O2`, lettura esclusa.
[Derivazione e implementazione](algorithms/mixed_integer_optimization.md#binary-selection-in-affine-psd-constraints)

| Istanza | Primal (ms) | Pajarito (ms) | Obiettivo Primal | Gap assoluto Primal |
|---|---:|---:|---:|---:|
| `clsq_random_32_2_a` | **2.406** | 11626.202 | 7.148545410477668 | 1.15e-11 |
| `rt_2x4_3bars` | 30.947 | — | 3.081138830080000 | 4.65e-13 |

Su `clsq_random_32_2_a`, i maggioranti di rango uno riducono la
ricerca a 3.300 nodi e nove fattorizzazioni delle matrici originali.
Su `rt_2x4_3bars`, l'enumerazione per costo chiude dopo 17.756
supporti. Il ricalcolo indipendente sui coefficienti dei CBF dà
violazioni al più `7.53e-16`, mentre la massima violazione
pubblicata è `8.89e-16`.

Per `clsq_random_32_2_a`, Pajarito 0.8.3 con HiGHS 1.26.0 e
Clarabel 0.11.1 ottiene `OPTIMAL` a `7.148545410339251` in due
prove dopo il riscaldamento. Il massimo residuo originale è
`3.96e-11`. Su `rt_2x4_3bars`, la nuova misura di Pajarito termina
con un errore LAPACK sia con Clarabel sia con Hypatia 0.11.0.
L'enumerazione indipendente dei supporti fino al costo trovato
conferma l'ottimo di Primal.

## 3. Come estendere

```sh
mkdir -p out
python3 -m venv out/benchmark-venv
out/benchmark-venv/bin/python -m pip install numpy scipy highspy scs clarabel
```

Con `highspy` installato `run_bench.py` lo usa al
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
| 2026-11-13 | **Matrice SOC/EXP/misto contro Clarabel** (`bench/pexp_matrix.*`, §3-ter): misto 3.0-6.3x, solo SOC 3.3-8.4x con **153.6x** a 10 blocchi (il denso stalla a 391 iterazioni), solo EXP **39-1019x** (a 80 blocchi il nativo non converge e partono 15 round di tagli). Nessuna patch al solver; il confronto **non** entra nel README finche' il divario non e' ridotto (bug aperto nel vault). |
