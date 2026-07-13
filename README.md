# Pi5-LLM-Energy-Benchmark

Misurazione di **efficienza energetica**, **qualità della risposta** e **latenza
di inferenza** di LLM quantizzati eseguiti localmente con **llama.cpp** su
**Raspberry Pi 5 (8 GB)**. Il consumo è misurato **direttamente dal PMIC del
Raspberry Pi 5** (nessuno strumento esterno): il PC host orchestra l'intero
esperimento via **SSH**.

## Idea generale

Il PC host coordina tutto:

1. si collega via **SSH** (Paramiko) al Raspberry Pi 5, alimentato dal suo
   **alimentatore ufficiale** (USB-C, 5,1 V / 3 A, 15 W) e sempre acceso;
2. campiona il **PMIC** del Pi (`vcgencmd pmic_read_adc`) a ~10 Hz: la potenza è
   la somma dei prodotti corrente×tensione sui 12 rami della scheda;
3. registra il consumo **idle** del Pi e lo usa come **bias** da sottrarre;
4. per ogni **modello × numero di thread (1, 2, 4) × ripetizione (×3)** avvia
   `llama-cli` (il modello resta caricato in memoria per l'intera sessione),
   **azzera la cronologia (`/clear`) prima di ogni prompt**, invia i prompt dei
   **5 benchmark** e misura energia, latenza, token/s e qualità;
5. salva tutto in **CSV** e genera i **grafici** di confronto con pandas,
   seaborn e scikit-learn.

Non è richiesto alcun hardware di misura esterno né alimentazione via GPIO
(che esporrebbe il PMIC a rischio di danneggiamento).

## Misura del consumo tramite il PMIC

Il Raspberry Pi 5 integra un **PMIC** (Power Management IC) con un ADC che espone
tensione e corrente di **12 rami** di alimentazione. Il comando
`vcgencmd pmic_read_adc` restituisce queste grandezze; la potenza istantanea è:

```
P_pmic = Σ (I_k · V_k)   con k = 1..12
```

Il metodo riprende il progetto open source **jfikar/RPi5-power**
(vedi `scripts/rpi5_power.sh`, incluso come riferimento).

- **Nessuna correzione per impostazione predefinita** (`corr_slope = 1.0`,
  `corr_offset = 0.0`): si registra la somma grezza del PMIC, misura coerente e
  ripetibile del consumo interno della scheda, adeguata al **confronto** tra
  modelli e quantizzazioni.
- La correzione lineare proposta da jfikar (`1.1451·P + 0.5879`) per stimare il
  consumo reale alla presa è **specifica di una singola scheda+alimentatore** e
  non è trasferibile: va usata solo se ricalibrata sul proprio setup
  (parametri `corr_slope`/`corr_offset` in `config.json`).
- Il campionamento avviene in un **thread in background** su una connessione SSH
  **dedicata**, con timestamp sul clock dell'host (lo stesso usato per delimitare
  le finestre di inferenza): non serve sincronizzare gli orologi di host e Pi.

## Modelli (6)

Tre famiglie, ciascuna in due livelli di quantizzazione (Q4_K_M e Q8_0):

| Modello | Quant. | Dimensione su disco |
|---|---|---|
| qwen2.5-1.5b-instruct | Q4_K_M | 1,12 GB |
| qwen2.5-1.5b-instruct | Q8_0   | 1,89 GB |
| gemma-2-2b-it         | Q4_K_M | 1,71 GB |
| gemma-2-2b-it         | Q8_0   | 2,78 GB |
| Llama-3.2-3B-Instruct | Q4_K_M | 2,02 GB |
| Llama-3.2-3B-Instruct | Q8_0   | 3,42 GB |

Comando di lancio (per configurazione):

```
~/llama.cpp/build/bin/llama-cli -m <modello>.gguf -t N -c 512 -n 256
```

dove `N` è il numero di thread (1, 2 o 4). `llama-cli` si avvia in modalità
interattiva di default; eventuali argomenti aggiuntivi si impostano in
`config.json` (`llama.extra_args`).

## Benchmark (5)

CommonSenseQA, BIG-Bench Hard, TruthfulQA, GSM8K, HumanEval — campioni curati in
`datasets/*.jsonl` (rispettivamente 15, 15, 15, 15 e 13 item, **73 prompt** in
totale). Con `max_samples_per_benchmark` in `config.json` si può valutare solo un
**sottoinsieme fisso** di prompt per benchmark — identico per tutte le
configurazioni, così il confronto resta equo — per contenere la durata sul
dispositivo (`0` o assente = tutti i campioni). Il loader conta automaticamente le
righe dei JSONL.

L'esperimento è organizzato in **due campagne complementari**, con due config
pronti (output separati, si lanciano con `--config`):

- `config_run1.json` — **confronto tra modelli**: dataset intero (73 prompt), a
  2 thread (configurazione più efficiente), 3 ripetizioni → 6 × 3 = 18 registrazioni, 1314 inferenze;
- `config_run2.json` — **scaling sul parallelismo**: 3 campioni/benchmark (15
  prompt), thread 1/2/4, 3 ripetizioni → 6 × 3 × 3 = 54 registrazioni, 810 inferenze.

```bash
python run_benchmark.py --config config_test.json   # prova rapida (~1 min): verifica la misura del PMIC
python run_benchmark.py --config config_run1.json   # campagna A -> results_run1.csv
python run_benchmark.py --config config_run2.json   # campagna B -> results_run2.csv
```

## Struttura

```
config.json                 parametri (Pi, PMIC, llama, thermal, modelli, benchmark)
config_run1.json            campagna A: dataset intero, 2 thread, 3 ripetizioni
config_run2.json            campagna B: 3 campioni/benchmark, thread 1/2/4, 3 ripetizioni
config_test.json            mini-run di prova (~1 min) per verificare la misura del PMIC
requirements.txt            dipendenze del venv host
datasets/*.jsonl            i 5 benchmark
run_benchmark.py            entry point eseguito dal PC host
inspect_models.py           ispezione rapida: prompt / risposta / punteggio per modello
scripts/
  pmic.py                   backend PMIC: lettura vcgencmd, somma I·V, energia su finestra
  integrity.py              verifica sha256 dei modelli (anti-corruzione)
  rpi5_power.sh             script di riferimento di jfikar/RPi5-power
  pi_ssh.py                 SSH Paramiko + shell interattiva llama-cli + vcgencmd
  llama_parser.py           parsing riga timing e testo generato
  datasets_loader.py        caricamento JSONL
  scoring.py                scoring per-benchmark (con flag di revisione)
  thermal.py                monitoraggio termico + decodifica throttling
  benchmark_runner.py       orchestrazione del flusso completo + CSV
  analysis.py               aggregazione e grafici (pandas/seaborn/scikit-learn)
  simulation.py             backend simulati per il dry-run (--simulate)
recordings/                 CSV risultati, dati aggregati, grafici
tesi/                       tesi LaTeX e schemi hardware/software
docs/                       checklist esperimento + diagramma di sequenza del codice
```

## Metriche

- **Efficienza energetica**: **energia netta per inferenza (J)**, dove
  energia_netta = energia_misurata − energia_idle sulla finestra di inferenza.
  L'energia è l'integrale (regola dei trapezi) della potenza campionata dal PMIC.
  *Non* si normalizza per token: il formato di output di llama.cpp non riporta il
  conteggio esatto dei token e una stima introdurrebbe un'approssimazione inutile.
- **Latenza**: tempo wall-clock dell'inferenza misurato dall'host (end-to-end:
  include il modesto overhead di comunicazione SSH/Ethernet), più le statistiche
  `prompt processing` e `generation` in token/s.
- **Qualità**: punteggio per-benchmark (match della lettera per la scelta
  multipla, corrispondenza esatta, estrazione del valore numerico, esecuzione
  dei test per HumanEval). Anche **TruthfulQA** è usato in forma a **scelta
  multipla** (variante MC1): 4 opzioni per domanda (l'affermazione veritiera + 3
  errate), valutate per corrispondenza dell'opzione scelta — automatico,
  deterministico e confrontabile, senza LLM-giudice (baseline casuale 25%). Le
  voci del codice restano marcate `needs_review` per un'eventuale rifinitura
  manuale.

## Esecuzione delle inferenze e parsing

- Il modello è avviato una sola volta in **modalità interattiva** e resta
  caricato per l'intera sessione (nessun ricaricamento a ogni prompt).
- Prima di ogni prompt si invia `/clear`: ogni inferenza è così **indipendente**
  dalle precedenti e il contesto non si accumula fino a saturare la finestra
  `-c` (con `ctx_size = 512`, valore contenuto per limitare la memoria sul Pi, i
  singoli prompt + risposta stanno comunque nel contesto).
- Ogni prompt è inviato come **una sola riga** (gli a-capo interni diventano
  spazi): in modalità interattiva il ritorno a capo conferma l'invio dell'input,
  ed è l'approccio affidabile su tutte le versioni di llama-cli.
- La **fine di ogni generazione** è rilevata dalla riga di statistiche che
  llama.cpp stampa a fine risposta:

  ```
  [ Prompt: 16,4 t/s | Generation: 15,8 t/s ]
  ```

  Questo delimitatore è univoco e **non risente dei caratteri `>`** presenti nel
  prompt o nella risposta (es. le annotazioni `->` o i doctest `>>>`), che
  altrimenti potrebbero essere scambiati per il prompt di fine risposta. Il
  simbolo `>` viene usato solo per rilevare che il modello è pronto **dopo il
  caricamento**. Il parser gestisce sia il punto sia la virgola come separatore
  decimale.

## Gestione termica e throttling

Il Pi 5 si scalda durante l'esecuzione: oltre ~80-85 °C entra in **throttling
termico** e cala le prestazioni, falsando latenza e t/s. Comportamento
configurabile nella sezione `thermal` di `config.json`:

- **Monitoraggio per inferenza** (`enabled`, `log_per_inference`): dopo ogni
  risposta vengono letti `vcgencmd measure_temp` e `vcgencmd get_throttled`;
  temperatura e stato di throttling finiscono nel CSV (`temp_c`, `throttled_hex`,
  `throttle_active`). `get_throttled` è decodificato bit a bit.
- **Rilevazione e ripetizione**: se durante una ripetizione si rileva throttling
  e `abort_on_throttle` è `true`, la registrazione viene **scartata e ripetuta**
  (fino a `max_retries`); le righe conservano `throttle_event` per tracciabilità.

Il case monta una **ventola di raffreddamento attiva** che tiene la temperatura
ampiamente sotto le soglie di throttling: il modulo si limita quindi a
**monitorare** (nessun cool-down tra le configurazioni, mai risultato necessario).

## Ispezione rapida delle risposte

Per vedere cosa risponde davvero un modello (utile a diagnosticare punteggi
bassi) senza avviare l'intera campagna:

```bash
python inspect_models.py                                   # 1 campione/benchmark
python inspect_models.py --samples 2 --threads 4
python inspect_models.py --models qwen2.5-1.5b-instruct-q4_k_m.gguf
python inspect_models.py --benchmarks gsm8k humaneval --out ispezione.txt
```

Per ogni modello e benchmark stampa il prompt, la risposta del modello, il
punteggio e i token/s, usando la stessa shell interattiva del benchmark.

## Uso

Ambiente virtuale già presente: `envEnergyBenchmark`. Per ricrearlo, installare
le dipendenze con `pip install -r requirements.txt` (principali: `paramiko` per
SSH; `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` per l'analisi).

```bash
# attiva il venv (Windows)
envEnergyBenchmark\Scripts\activate

# prova della pipeline SENZA hardware (genera CSV + grafici simulati)
python run_benchmark.py --simulate

# esecuzione reale: Pi acceso e raggiungibile via Ethernet all'IP di config.json,
# SSH abilitato e `vcgencmd pmic_read_adc` disponibile sul Pi
python run_benchmark.py

# opzioni utili
python run_benchmark.py --no-analysis    # salta la generazione dei grafici
python run_benchmark.py --verbose        # log dettagliato

# rigenera SOLO i grafici/tabelle da un CSV esistente (senza rifare il benchmark);
# 2o argomento opzionale = config con i percorsi di output desiderati
python -m scripts.analysis recordings/results.csv config.json
```

Output principali:

- `recordings/results.csv` — una riga per ogni (modello, thread, ripetizione,
  benchmark, campione) con energia netta, energia totale/idle, potenza media,
  latenza, prompt/gen t/s, score, temperatura e stato di throttling;
- `recordings/raw/aggregated_by_benchmark.csv`, `recordings/raw/energy_per_config.csv`, `recordings/raw/ranking_composite.csv`, `recordings/raw/pareto_energia.csv`, `recordings/raw/pareto_latenza.csv`;
- `recordings/plots/*.png` — efficienza (energia netta per inferenza), **energia
  totale consumata per configurazione**, latenza, potenza, throughput (prompt
  processing e generation), **distribuzioni a boxplot di energia e latenza**,
  qualità (heatmap), trade-off efficienza/qualità, **frontiera di Pareto**
  (energia--qualità e latenza--qualità), classifica composita e termico.

## Note operative

- Il costo energetico del **caricamento** del modello è registrato a parte
  (riga `__load__`), così l'energia di inferenza non lo include.
- Il **bias idle** va rimisurato se cambiano le condizioni (temperatura,
  periferiche collegate).
- **Integrità dei modelli** (`config.json → verify_models`): all'avvio calcola lo
  sha256 di ogni GGUF e lo confronta con una baseline salvata (`models.sha256.json`,
  creata alla prima esecuzione); se un file è cambiato/corrotto il benchmark si
  ferma. Se aggiorni volontariamente un modello, cancella quella voce dalla baseline.
- Tutti i parametri (Pi, PMIC, `llama`, `thermal`, modelli, thread, ripetizioni,
  benchmark, verifica integrità e percorsi dei dataset) sono raccolti in
  `config.json` (per le due campagne si usano `config_run1.json` e `config_run2.json`).
- Il campionatore PMIC è **resiliente**: chiude ogni sessione SSH dopo la lettura
  (così non esaurisce i canali del server su campagne lunghe) e, se le letture
  iniziano a fallire, si **riconnette** e riprende automaticamente; se una
  registrazione resta senza campioni lo segnala con un WARNING. In analisi le
  eventuali righe senza energia valida vengono escluse dalle metriche energetiche.
