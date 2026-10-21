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
| `95f1c32`, 2 ottobre | 0 | 2 |
| `557d501`, padre di `edd355b` | 0 | 2 |
| `edd355b`, 3 ottobre | 4 | 2 |
| `3b356f4`, 3 ottobre | 4 | 2 |
| `ed7fc79`, master prima del merge | 4 | 2 |
| `2a71577`, master dopo il fast-forward | 4 | 2 |

La storia di T244 non è monotona: era rosso nella radice, è diventato verde,
poi è ricomparso con `edd355b`. T197 va attribuito separatamente.

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

**Nessuno dei sei è un errore di memoria o di UB.** Compilando la suite con
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
