# Attribuzione dei sei check sanitizer rossi — 2026-10-21

Il fast-forward di `sdptimeref` **non introduce i sei errori**: lo snapshot
completo di master precedente al merge (`ed7fc79`) li riproduce già. Il merge
porta master a `2a71577`; anche quel commit, e il successivo checkpoint KKT
`5f7c3c1`, mostrano quattro check T244 e due T197 rossi sotto ASan+UBSan.
Sono due solve numericamente falliti, non sei diagnostiche di memoria.

## Metodo e limiti

Ogni revisione storica è estratta integralmente con `git archive`, compilata
e linkata in una directory separata sotto `out/san_cause_20261021/`.
Apple clang 21.0.0 (`clang-2100.3.34.2`), arm64, flag identici:

```text
-std=c99 -Wall -Wextra -pedantic -O2 -g
-fno-omit-frame-pointer -fsanitize=address,undefined
ASAN_OPTIONS=detect_leaks=0
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
```

Il runner include l'intero `test_primal.c` della revisione, rinomina il suo
`main` e chiama T244, poi T197. Tutte le variabili `GMB_*` ereditate vengono
rimosse; T197 abilita da sé il percorso HSD previsto dal test. Nessun oggetto
è condiviso fra gli snapshot completi. I confronti di singoli file sotto
sono invece dichiaratamente ablazioni, con lo stesso harness e gli stessi
altri oggetti. I log `build.log`, `link.log`, `run.log`, `result.json` sono
conservati per revisione; non sono artefatti versionati.

Queste sono riproduzioni mirate sulla toolchain attuale, non attestazioni
di come girassero i binari storici. Non è stata ripetuta la suite completa
per ogni commit. Nessuna tolleranza di accettazione è stata modificata.

## Snapshot completi

| Revisione | T244: check falliti / 4 | T197: check falliti / 3 |
|---|---:|---:|
| `1acd29b`, primo commit | 4 | 0 |
| `7d0ef74` | 4 | 0 |
| `57dfe14`, figlio diretto di `7d0ef74` | 4 | 2 |
| `95f1c32`, 2 ottobre | 0 | 2 |
| `557d501`, padre di `edd355b` | 0 | 2 |
| `edd355b`, 3 ottobre | 4 | 2 |
| `3b356f4`, 3 ottobre | 4 | 2 |
| `ed7fc79`, master prima del merge | 4 | 2 |
| `2a71577`, master dopo il fast-forward | 4 | 2 |

La storia di T244 non è monotona: era rosso nella radice, è diventato verde,
poi è ricomparso con `edd355b`. T197 va attribuito separatamente.

Il confine di T197 è `7d0ef74` → `57dfe14` (28 settembre). L'unico file
modificato è `linalg.c`: il criterio Jacobi assoluto `off < 1e-30` diventa
relativo, `off <= 1e-28 * ||A||F²`. Con la stessa toolchain sanitizer il padre
termina con `best_viol=0.9653`, il figlio con `best_viol=3.532` e due check
rossi. Questo identifica il cambiamento che espone la fragilità numerica;
non giustifica un ritorno globale al criterio assoluto.

## T244: rotazioni Jacobi di `edd355b`

`edd355b` modifica soltanto `linalg.c` e `sdp.c`: sostituisce le rotazioni
Jacobi a due passate con l'aggiornamento simmetrico a una passata, sia in
`dmat_eig_jacobi` sia in `min_eig`. Padre verde, figlio rosso, stessi test.

Su tutti gli oggetti del padre `557d501`, mantenendo identico l'harness:

| Unica sostituzione | T244: check falliti / 4 |
|---|---:|
| nessuna | 0 |
| solo `linalg.o` da `edd355b` | 4 |
| solo `sdp.o` da `edd355b` | 1 |
| entrambi (snapshot completo `edd355b`) | 4 |

Il cambio dell'eigensolver è dunque sufficiente a riprodurre tutti e quattro
i check; anche il solo `min_eig` causa mancata convergenza. Non serve
un'interazione con le modifiche SOCP di ottobre inoltrato.

Con `GMB_DBG=1`, il padre recupera un punto SDP misurato e risponde con
`status=0` (`rel_pri=2.14e-7`, `rel_dual=1.16e-11`, `rel_gap=1.73e-6`,
fattore near-optimal esistente 1000). Il figlio termina quel percorso senza
un punto recuperato, poi il fallback non converge; l'API restituisce 1007.
Il cambio altera la traiettoria numerica e la sua accettazione. Il confronto
non dimostra che la formula Jacobi sia matematicamente errata, né che un
rollback globale sia una correzione adeguata per tutti i modelli.

## Stato della correzione

L'indagine sopra non cambia sorgenti di produzione o soglie; la correzione
è venuta dopo, nella sezione seguente.

## Causa radice e correzione (2026-10-21)

**Le prove non rilevano errori di memoria o di UB associati ai sei check.** Compilando la suite con
sanitizer singoli: ASan da solo passa tutto; T244 fallisce con quattro
sotto-verifiche UBSan indipendenti (`alignment`, `null`, `pointer-overflow`,
`signed-integer-overflow`), nessuna delle quali segnala nulla. Hanno in comune
solo l'effetto sull'ottimizzatore: impediscono la vettorizzazione. Infatti
T244 e T197 falliscono anche senza sanitizer a `-O0`, `-O1` e con
`-fno-vectorize`. La rotazione Jacobi di `edd355b` ha solo cambiato la
traiettoria che espone il difetto.

**T244.** A `-O0` la traiettoria coincide con `-O2` fino a it=5. A it=7 una
coppia di variabili libere spezzate è esplosa (`x = 4.8e6`, `s = 7e-16`,
`x/s = 6.6e21`). L'eliminazione R+ a equazioni normali mette quel fattore in
una riga primale (equilibratura `1.5e-22`): risolta a `3e-21` in unità
equilibrate, vale `0.026` in unità vere, e la direzione ha `|rp + A d| = 0.056`
contro `|rp| = 1e-12`. Il passo pieno porta `pfeas` da `1e-12` a `2.8e-2`.
Correzione in `sdp.c`: una direzione che manca le proprie righe primali di
oltre `10|rp| + 1e-10(1+|b|)` viene ricalcolata con il blocco R+ in forma
aumentata (come nel modo SECANT), che resta attiva per il resto della
soluzione. T244 passa a ogni livello, senza FMA e sotto UBSan.

**T286 senza FMA** (non fra i sei, emerso compilando con
`-ffp-contract=off`, cioè come gcc su x86-64 senza `-march`): il solver SOCP
legacy restituiva passi **negativi** dal ratio test R+/SOC una volta che una
coordinata scendeva sotto zero per arrotondamento, e divergeva fino a
`feas_p=3.5e+260`. Ora `step_len` restituisce 0. Nello stesso giro `T83 A`
(ottimo piatto) ha la tolleranza su `u` derivata dall'errore sull'obiettivo.

**T197 resta.** È il percorso HSD opt-in, che il codice stesso dichiara
inchiodato all'arrotondamento storico; anche scrivere `ds` in forma diretta lo
fa fallire ovunque. Fallisce a `-O0`, `-O1`, senza FMA e sotto UBSan. Renderlo
robusto è lavoro sull'HSD (issue #11), non una correzione di questo giro.

Stato misurato: 5168/0 a `-O2`, `-O3`, con e senza `GMB_SOCP_LDL`, e sotto
ASan; 5166/2 (solo T197) a `-O0`, `-O1`, senza FMA, sotto UBSan e ASan+UBSan;
nessuna diagnostica sanitizer; 171/171 esempi a exit 0.

## Revisione indipendente del commit `472dfbb`

Le misure del paragrafo precedente sono dell'altra sessione. Su uno spazio
di build separato sono stati riconfermati 5168/0 a O2, default e opt-in, e
5166/2 sotto ASan+UBSan opt-in, senza diagnostiche. I sorgenti del suo
snapshot di verifica coincidono con quelli committati per `sdp.c`, `socp.c`
e `test_primal.c`.

La revisione ha trovato un ulteriore errore nel ramo lineare del ratio test
SOC. Per `A*a²+B*a+C <= 0`, con `A=0`, il limite superiore esiste per
`B>0`, non per `B<0`. Dal punto `(1,0)`:

| Direzione | Risultato di `472dfbb` | Passo esatto |
|---|---:|---:|
| `(1,1)` | 0 | 1 |
| `(-1,1)` | 1 | 0.5 |

Il ramo è corretto e T287 verifica questi due casi geometrici, la permanenza
nel foglio positivo del cono, due attraversamenti quadratici e le guardie
R+/SOC fuori dal dominio. La funzione è esposta nel solo header interno
`socp.h`, non nell'API pubblica `primal.h`.

Per T197 sono state provate esclusivamente in `out/review_hsd/`:

| Variante isolata | Migliore violazione normalizzata ASan+UBSan | Esito |
|---|---:|---|
| baseline `472dfbb` | 2.098 | max_iter, 2 check rossi |
| denominatore del merito con tau del trial | 2.098 | invariato |
| scalari R+ espliciti nel KKT HSD | 1.499 | ricerca di linea esaurita |
| precedente + ripristino per copia dei trial rifiutati e tau corretto | 9.011 | peggiore |

Nessuna di queste varianti è promossa nei sorgenti. Una sonda della baseline
ha misurato residui delle equazioni Newton ancora intorno a `1e-6` dopo le
cinque rifiniture (1978 direzioni oltre `1e-13`). È un limite concreto della
direzione numerica HSD, non soltanto l'osservazione che cambia arrotondamento.
Non è stata dimostrata una correzione robusta di T197. Nessuna soglia del
solver è stata allentata e nessun test è stato saltato.

T287 isolato passa 8/0 a O0/O1/O2/O3, senza FMA e sotto ASan+UBSan,
con compilazione senza warning. Controllo negativo: lo stesso test con la
funzione di `472dfbb` produce 4/4, inclusi il passo nullo verso l'interno e
il passo che attraversa il bordo. Risultati locali in
`out/review_final/geometry/results.json` e `negative.log`.

La suite completa finale O2 passa **5176/0** con entrambi i backend. La
matrice completa O0/O1/O3/no-FMA e i 171 sample restano le misure dell'altra
sessione sul checkpoint `472dfbb`: non sono stati rieseguiti integralmente
dopo il successivo fix del ratio test. Le nuove prove sugli altri livelli
riguardano specificamente la geometria di T287.

Verifica finale ASan+UBSan: **5174 PASS / 2 FAIL** con entrambi i backend,
runner exit 1; i soli due errori sono T197 e non ci sono diagnostiche
ASan/UBSan. Build senza warning. Log in `out/review_final/`.

L'ispezione dei log sample dell'altra AI conferma **171/171 exit 0** e nove
output con differenze numeriche (un decimo differisce solo per il tempo).
Nei log conservati il gate `[augx]` si attiva in **24** esempi, non 25;
il README è allineato a questa evidenza. I sorgenti testati coincidono col
checkpoint, ma non si tratta di una nuova esecuzione dei sample dopo T287.

## Chiusura di T197 e difetto aperto senza FMA (2026-10-21)

**T197 è risolto** dalla patch descritta in `hsd-dependent-rows.md`: righe
dipendenti eliminate solo nel percorso HSD, KKT aumentato con direzioni
scalari e PSD esplicite, arresto misurato sul modello originale, costi ridotti
scalari pubblicati dal publisher SDP. Matrice completa su `5f02cb2` più i tre
file: **5196/0** a O0, O1, O2, O3, sotto ASan, UBSan e ASan+UBSan, con e senza
`GMB_SOCP_LDL`; nessuna diagnostica; 171/171 esempi con output invariati.

**Resta T286 senza FMA** (`-ffp-contract=off`, come gcc su x86-64 senza
`-march`): 7 check, sempre la rotta SOCP legacy sulla variante con 60 coni
accoppiati, `rc=1007`. Era nascosto dal troncamento a zero di `472dfbb`, che
nel ramo lineare usava il segno sbagliato di `B`; `5f02cb2` ha corretto la
geometria (giusto) e il caso è riapparso. Traccia: a it=5 la direzione NT
viene rifiutata con `mu=1e-12`, la direzione LU porta `mu` a `5e-17` in un
passo (frazione di passo 0.99995), e da it=7 il passo primale è 0 perché il
punto è numericamente sul bordo: con `mu~1e-17` la distanza `v0^2-|v|^2` è
sotto l'arrotondamento con cui la si calcola. L'infeasibilità primale resta
a `1e-8`, il gap a `2e-7`, fino al limite di iterazioni.

Tentativo scartato: frazione 0.99 sul solo ramo LU. Risolve la sonda isolata,
ma nella suite completa T286 resta rosso, T279 diventa rosso senza FMA (4
check) e `logcontrast_lasso` cambia beta alla quinta cifra anche con FMA. La
causa è una perdita di centralità (`mu` scende molto più in fretta dei
residui); la correzione di principio è una salvaguardia che non lasci scendere
il bersaglio `sigma*mu` sotto una frazione dei residui. Cambia però le
traiettorie di tutto il SOCP legacy e va progettata e verificata a parte.

## T286 senza FMA: recupero di bordo (2026-10-21)

Misura prima della scelta: la centralità per cono, `sqrt(det s * det lam)/mu`
(1 sul cammino centrale; per R+ `s_i*lam_i/mu`), con `det` calcolato come
`(v0-|v|)(v0+|v|)`. Con FMA resta sopra 0.02 fino alla convergenza; senza FMA
il primo passo LU (it=5) porta un blocco a `det <= 0` prima della convergenza.
Toccare il bordo di per sé è comune e innocuo: nella suite con FMA 10 delle
16 soluzioni SOCP sparse lo fanno almeno una volta (fino a 3 iterazioni) e
convergono tutte. Una salvaguardia sul passo avrebbe quindi cambiato 10 casi
sani; lo stallo vero ha invece una firma propria: passo nullo ripetuto.

Correzione in `socp_solve_sparse`: dopo **due passi bloccati consecutivi**
(lunghezza primale o duale `<= 1e-13`) senza convergenza, i blocchi schiacciati
(scarto dal bordo `<=` soglia) vengono riportati dentro di
`10*max|residuo| + 1e-9*(1+|v0|)` (primale da `rg`, duale da `rd`), poi
l'iterazione riparte; al massimo cinque volte per soluzione. Sulla sonda di
T286 senza FMA scatta una volta a it=8 (180 blocchi, spinta `1.3e-7`) e la
soluzione converge (`rc=0`, obiettivo 1.118033974 contro `sqrt(1.25)`).

Verifica, albero isolato `c2ab91f` + `socp.c`: **5196/0** a O0/O1/O2/O3,
**senza FMA**, sotto ASan, UBSan e ASan+UBSan, con e senza `GMB_SOCP_LDL`;
build senza warning, nessuna diagnostica. Il recupero non scatta mai nella
suite con FMA né nei 171 esempi (output identici a `c2ab91f`, salvo il tempo
di `regression_ls`); senza FMA scatta due volte, entrambe sulle varianti a 60
coni di T286 (la variante 1 prima convergeva dopo 7 iterazioni sul bordo).
