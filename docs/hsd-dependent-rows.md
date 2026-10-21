# T197: righe dipendenti, KKT HSD e qualità pubblicata

La regressione riguarda T197; T287 era già verde. La correzione lascia HSD
opt-in (`GMB_SDP_HSD`) e mantiene tolleranze e limite di iterazioni.

## Cause isolate

T197 ha dieci uguaglianze ma rango otto, anche includendo il termine noto.
Il KKT Schur iniziale 12×12 ha due direzioni nulle: i due valori singolari
minori misurati sono circa `3.8e-16` e `8.3e-18`. Trattarlo come invertibile
lega la direzione agli arrotondamenti. Un solve in double-double non recupera
l'informazione già persa assemblando in double il complemento di Schur.

La sola riduzione delle righe non chiude il caso; nemmeno il solo blocco
scalare aumentato. La variante che trattiene anche le direzioni PSD resta
singolare finché non vengono rimosse le uguaglianze dipendenti. Dopo la
riduzione e il KKT completo, T197 converge nelle build misurate.

Due ulteriori controlli hanno impedito di promuovere una risposta incompleta:

- Un residuo dell'embedding include gli offset proporzionali a `theta`.
  Essere ammissibile nell'embedding non implica rispettare le tolleranze
  del modello originale. Invertendo le righe, il vecchio arresto accettava
  un residuo primale originale di `4.44e-8`. Ora il selettore del miglior
  punto e l'arresto misurano anche i residui senza gli offset artificiali.
- Il publisher SDP nativo copiava `y` e il duale PSD, ma lasciava a zero
  i costi ridotti scalari. Il getter pubblico misurava quindi una violazione
  di stazionarietà di circa 6, benché il duale interno fosse corretto. Ora
  vengono pubblicati `slc/suc` e, fuori dai coni, `slx/sux` da `-(c+A'y)`.

L'arresto HSD controlla anche la complementarietà scalare
`(x/tau)*(s/tau)/(1+s/tau)` alla tolleranza duale dichiarata, coerentemente
con la classificazione dei bound nel getter pubblico. Questo rende più
stretta l'accettazione; nessuna soglia è stata aumentata.

## Implementazione

Nel solo percorso HSD, Gram-Schmidt ripetuto con pivot sulle righe
normalizzate di `[A, Abar, b]` seleziona una base indipendente. Includere `b`
impedisce di scartare una contraddizione come `x=2`, `2x=5`. La tolleranza
di rango deriva da epsilon macchina e dimensione; non è una tolleranza di
accettazione del punto. Dopo il solve il duale viene riportato alle righe
originali, con zeri sulle righe rimosse. Il gate finale controlla tutte le
righe originali, dominio, stazionarietà e gap alle tolleranze richieste.

Il KKT conserva esplicitamente le direzioni scalari e i triangoli inferiori
delle direzioni PSD. Nelle righe di traccia gli elementi fuori diagonale
contano due volte; nelle equazioni di cono una sola. L'equivalenza con il
vecchio Schur è stata verificata numericamente sul primo passo, prima della
promozione della variante. Il solve conserva la fattorizzazione in
double-double e le cinque rifiniture già previste.

## Verifica

T197 misura ora i getter primali/duali e il gap, oltre al margine noto, anche
invertendo l'ordine delle uguaglianze. T288 chiama HSD direttamente su un
modello con duplicati scalati, controlla primalità, dualità, PSD e obiettivo
nelle righe originali, poi verifica il rifiuto del duplicato incoerente.

- Runner mirato: **23/0** a O0/O1/O2/O3, senza FMA e sotto ASan+UBSan.
- Controllo negativo con `sdp.c` di `5f02cb2`: **3/20** nello stesso runner
  sanitizer. Nessun test viene saltato nella variante corretta.
- Suite completa O2: **5196/0**, default e `GMB_SOCP_LDL=1`.
- Matrice completa, ripetuta in modo indipendente su uno snapshot di HEAD
  (`5f02cb2`) con i soli tre file della patch: **5196/0** a O0, O1, O2, O3,
  sotto ASan, UBSan e ASan+UBSan (con e senza `GMB_SOCP_LDL`), build senza
  warning e nessuna diagnostica sanitizer. È la prima suite sanitizzata
  completamente verde.
- Senza FMA (`-ffp-contract=off`): **5189/7** a questo checkpoint, i sette
  check di T286 sulla rotta SOCP legacy, già rossi in `5f02cb2` e non legati
  all'HSD. Risolti dopo dal recupero di bordo in `socp.c`; vedi
  `sanitizer-regression-analysis.md`.
- 171/171 esempi a exit 0, con output identici a `5f02cb2` (l'unica
  differenza è il tempo stampato da `regression_ls`).

Sul T197 originale, la build sanitizer misura dai getter pubblici:
`primalinfeas=2.87e-9`, `dualinfeas=2.31e-16`,
`pobj=-0.855065849564`, `dobj=-0.855065857582`.

Le verifiche complete usano uno snapshot di `5f02cb2` con i soli tre file
della patch, per non interferire con la riorganizzazione degli esempi in
un'altra sessione. Log e sonde sono in `out/hsd_f60/`.

Questo risultato chiude il caso misurato, non certifica la robustezza
generale dell'HSD: vicino al bordo restano possibili grandi residui Newton
transitori. Il backend resta sperimentale e spento di default.
