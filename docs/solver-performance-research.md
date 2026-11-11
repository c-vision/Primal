# Confronto architetturale e piano prestazioni

Analisi del 2026-11-04, checkout `06f767b`. Ricerca documentale e ispezione
del codice; nessun nuovo benchmark eseguito, nessuna accelerazione dimostrata
da questa analisi. Le pagine upstream `stable`, `dev`, `main` e `latest` sono
mobili: prima di implementare, fissare versione e commit della fonte scelta.
Il seguito implementativo F110 e le nuove misure sono documentati in
[higher-order PEXP correction](../algorithms/pexp-higher-correction.md); la dichiarazione
iniziale qui sopra riguarda la sola ricerca del 4 novembre.

## Conclusione operativa

Primal deve distinguere costo per iterazione, numero di iterazioni e costo dei
percorsi di recupero. Un kernel più rapido non risolve un percorso che perde
centralità e passa ai tagli. Un algoritmo di primo ordine può ridurre molto il
costo per iterazione, ma non garantisce lo stesso costo per arrivare a 1e-8.

Il confronto utile è per classe e qualità raggiunta: LP/MIP con HiGHS e COPT;
SOCP/exp/power/SDP con Clarabel, SCS, COSMO e COPT dove supportati. La GPU è una
configurazione da misurare separatamente, con hardware dichiarato.

## Cosa fanno gli altri

| Solver | Meccanismo verificato | Conseguenza per Primal |
|---|---|---|
| Clarabel | Embedding omogeneo con obiettivo quadratico diretto; KKT simmetrico quasi-definito; backend QDLDL con analisi simbolica e aggiornamenti dei valori | Studiare formulazione, scaling e direzione insieme; conservare Q senza espansioni coniche quando possibile; separare analisi e fattorizzazione numerica |
| SCS | Operator splitting; sistema lineare riusabile finché la scalatura resta fissa; LDL diretto o CG matrix-free; accelerazione di Anderson | Possibile motore distinto per problemi grandi a precisione moderata, con riuso dei fattori; Anderson non si applica automaticamente al Newton attuale |
| COSMO | ADMM con proiezioni coniche; Anderson con salvaguardia; decomposizione cordale e fusione delle clique PSD | Sfruttare prima la struttura dei blocchi SDP; valutare un percorso ADMM separato solo con un carico che lo giustifica |
| HiGHS | Simplex rivisto; IPX con precondizionamento di base; HiPO con fattorizzazione diretta e scelta fra sistema aumentato/equazioni normali; PDLP; branch-and-cut | Il vantaggio LP/MIP richiede basi sparse e riottimizzazione, presolve e selezione del metodo, oltre al kernel IPM |
| COPT | Portfolio pubblico di simplex, barrier, PDLP; SDP primal-dual/duale/ADMM; GPU per metodi e classi dichiarati | Prendere come riferimento funzionale e sperimentale; la documentazione pubblica non permette di attribuire il vantaggio a un particolare kernel proprietario |

Fonti: [Clarabel e formulazione](https://github.com/oxfordcontrol/Clarabel.rs),
[backend QDLDL](https://github.com/oxfordcontrol/Clarabel.rs/blob/main/src/solver/core/kktsolvers/direct/quasidef/ldlsolvers/qdldl.rs),
[SCS algebra lineare](https://www.cvxgrp.org/scs/linear_solver/index.html),
[COSMO ciclo ADMM](https://github.com/oxfordcontrol/COSMO.jl/blob/master/src/solver.jl),
[COSMO accelerazione](https://oxfordcontrol.github.io/COSMO.jl/stable/acceleration/),
[COSMO decomposizione](https://oxfordcontrol.github.io/COSMO.jl/stable/decomposition/),
[HiGHS metodi](https://ergo-code.github.io/HiGHS/dev/solvers/),
[COPT metodi](https://guide.coap.online/copt/en-doc/modeling.html).

## Differenze concrete nel checkout

### Exp/power: evitare una diagnosi troppo generale

`expcone.c::nt_eval` forma Hz, decompone matrici 3x3 e inverte Hw per costruire
la metrica. Esiste già un percorso secante opt-in `expcone_scaling_da`:
proporre semplicemente «aggiungere Dahl-Andersen» ripeterebbe lavoro esistente.

F90–F95 di `attempts.md` documentano, su revisioni precedenti, degradi vicino
al bordo e fallimenti di modifiche locali della metrica. F95 mostra che alcune
Hz calcolate perdono perfino l'identità di omogeneità. È evidenza su quei punti
e su quella rappresentazione; non prova che ogni algoritmo basato su una
matrice 3x3 densa debba fallire.

Nel sorgente Clarabel `expcone.rs`, `update_scaling` valuta **gradiente e
Hessiano della barriera duale**, poi aggiorna Hs. `combined_ds_shift` include
`higher_correction`; questa risolve H*u=ds con Cholesky 3x3 e annulla la
correzione se la fattorizzazione fallisce. Anche Clarabel conserva matrici
dense 3x3: non è corretto attribuire il suo vantaggio all'assenza di Hessiani
densi. Restano da confrontare coordinate primali/duali, normalizzazione,
metrica, correzione e ricerca del passo come sistema coerente.

Nelle righe exp/power di `sdp.c` il RHS viene da `rt + rde` e
`expcone_scaling` costruisce `rt=(-s-smu*g)/gap`; in questi punti non compare
la correzione di ordine superiore osservata in Clarabel. È una differenza
concreta da verificare su un singolo passo, non una prova che basti aggiungere
un termine al RHS per risolvere lo stallo.

Fonte primaria:
[implementazione del cono esponenziale Clarabel](https://docs.rs/clarabel/latest/src/clarabel/solver/core/cones/expcone.rs.html).
Questa ispezione non dimostra quale singola differenza causi il divario.

### SOCP e KKT: parte dell'architettura esiste già

`socp.c` ha NT con equazioni normali/Cholesky, fallback LU e un KKT
quasi-definito opt-in (`GMB_SOCP_LDL`). `linalg.h` espone già
`spqd_analyze`, `spqd_factor`, `spqd_solve`. Non manca semplicemente «LDL».
F56–F57 e F63–F80 conservano prove, regressioni e misure inconcludenti:
un nuovo backend deve prima spiegare il difetto che supera.

Il backend QDLDL di Clarabel inizializza il simbolico e gli spazi di lavoro,
aggiorna i coefficienti e rifattorizza. In Primal il percorso LU continua a
costruire/liberare fattori; un ordinamento riusato non equivale a riusare tutto
il simbolico. Per la LU con pivoting il pattern dei fattori può cambiare:
non trasferire la garanzia quasi-definita a una matrice generale.

`sdp.c` riusa già i fattori predictor/corrector quando la matrice coincide.
Non riproporre quel riuso come nuovo guadagno: F17 ne documenta anche il
risultato prestazionale inconcludente.

### SDP: struttura prima del parallelismo indiscriminato

Scomporre una grande PSD sparsa in clique riduce la dimensione delle operazioni
spettrali, ma richiede vincoli di consistenza e ricostruzione duale corretti.
COSMO espone anche strategie di fusione: molte clique minuscole non sono
necessariamente migliori. Una PSD con pattern aggregato denso non trae lo
stesso beneficio. La famiglia corrente C=-vv' del benchmark ha costo denso:
non usarla come unica prova della decomposizione cordale.

### LP/MIP: il confronto con HiGHS è una linea separata

`simplex.c` comprende tableau, dual simplex e revised simplex, ma i percorsi
ispezionati mantengono esplicitamente B^-1 e ricostruiscono basi con algebra
densa. HiGHS documenta hot start dalla base ottima dopo modifiche LP.
Prima di aggiungere thread al branch-and-bound, misurare tempo per rilassata,
riuso della base, pivot e refactor. Non basta contare i nodi.

## GPU: cosa è trasferibile

SCS documenta sia GPU indirect sia direct con cuDSS; la documentazione Python
attuale raccomanda cuDSS rispetto al vecchio indirect. COPT documenta GPU
barrier anche conica e PDLP per LP. HiGHS documenta PDLP GPU e differenze fra
build e dispositivi. Sono percorsi specifici: non implicano accelerazione
automatica dei loop C99 di Primal.

Su questo Mac non si può assumere disponibile CUDA. SCS documenta anche un
backend sparse LDL di Apple Accelerate: è un candidato CPU locale per un
confronto del kernel, non una GPU e non un miglioramento già misurato. Ogni
backend esterno proposto va mantenuto opzionale per preservare la build libm.

Fonti: [SCS installazione/backend](https://www.cvxgrp.org/scs/install/python.html),
[COPT GPU](https://guide.coap.online/copt/en-doc/parameter.html),
[HiGHS GPU](https://ergo-code.github.io/HiGHS/dev/guide/gpu/).

## Sequenza proposta e criteri scritti prima delle prove

Questi sono obiettivi di esperimento, **non previsioni né risultati**.

1. **Misure confrontabili.** Nel benchmark attuale Clarabel è costruito prima
   del timer (`bench/conic_ref.py:88`, `bench/bench_all.py:137`); SCS include
   setup nel timer. Separare preparazione+solve e solo solve, fissare versioni,
   backend e thread. Usare livelli 1e-4, 1e-6, 1e-8 verificati nelle unità del
   modello originale: stessi numeri nei settings non bastano. Registrare
   status, infeasibilità, gap, memoria, iterazioni e percorso finale. Riportare
   fallimenti separatamente dai tempi; scartare misure con confini diversi.
2. **Prima prova conica: direzione exp/power.** Riprodurre un punto di
   `logistic_large` e un bordo T91 in un test sul singolo passo. Confrontare
   equazioni, scaling duale e correzione Clarabel con un oracolo indipendente
   ad alta precisione. Criterio: backward error <=1e-10 nel test del passo,
   nessuna perdita dei gate pubblici e nessuna regressione T91. Solo dopo,
   obiettivo di solve: eliminare il retry sul caso scelto e ridurre il tempo
   mediano almeno del 30%, con conferma indipendente. Se il test del passo
   fallisce, fermare la patch; non compensare allargando le tolleranze.
3. **Seconda prova: costo KKT isolato.** Su un SOCP rappresentativo contare
   analisi simboliche, refactor, fill e allocazioni; confrontare kernel sulla
   stessa matrice/RHS. Obiettivo >=20% sul solve completo, nessuna regressione
   >5% sul controllo scelto, a qualità invariata. Rigettare se il kernel pesa
   troppo poco per raggiungere l'obiettivo, prima di implementare un backend.
4. **SDP sparso.** Su un caso con struttura cordale confrontare clique/fill e
   costo spettrale, con controllo denso. Obiettivo >=2x nel tempo totale sul
   caso strutturato; duale ricostruito e gap originali devono passare. Non
   promettere guadagni sui casi densi; fermare se l'overhead annulla il vantaggio.
5. **LP/MIP.** Prima una sequenza LP con soli bound modificati: obiettivo
   >=2x rispetto alla risoluzione corrente, base e certificati misurati. Poi
   propagare il riuso al MIP e misurare separatamente costo/nodo e nodi.
6. **Primo ordine/GPU.** Aprire una linea distinta solo se i casi grandi a
   precisione moderata restano prioritari dopo i profili. Richiede proiezioni,
   terminazione/certificati e backend dedicati; non è una patch al Newton.

Non avviare tutte le prove insieme. Prima aggiornare la baseline su una sola
famiglia; ogni nuova implementazione avrà una voce propria in `attempts.md`.
Nessun tuning di sigma, pivot floor o sostituzione di inverse già respinti
senza una nuova differenza verificabile. Nessuna tabella README aggiornata
con questa sola ricerca.

Nota sul registro locale: la ricerca iniziale ha rilevato una divergenza
preesistente (3525 righe originali contro 2239 nello specchio, mancanti
F60–F95). Il seguito F110 ha conservato lo storico e i nuovi tentativi,
riallineando lo specchio il 2026-11-05.
