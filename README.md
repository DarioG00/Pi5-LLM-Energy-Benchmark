# Pi5-LLM-Energy-Benchmark

Misurazione di **efficienza energetica**, **qualità della risposta** e **latenza
di inferenza** di LLM quantizzati eseguiti localmente con **llama.cpp** su
**Raspberry Pi 5 (8 GB)**, usando un **Otii Arc** (Qoitech) come strumento di
misura del consumo, pilotato dal PC host via API TCP.

## Idea generale

Il PC host orchestra tutto:

1. configura e accende l'**Otii Arc** via API TCP (5 V, limite di corrente a 5 A
   per gestire i picchi del Pi 5, generatore nominale 2.4 A);
2. alimenta il Raspberry Pi 5 e attende il boot;
3. apre una connessione **SSH** (Paramiko) verso il Pi (192.168.10.2, utente
   `dario`);
4. registra il consumo **idle** del Pi e lo usa come **bias** da sottrarre;
5. per ogni **modello × numero di thread (1, 2, 4) × ripetizione (×3)** avvia
   `llama-cli` (il modello resta caricato in memoria per l'intera sessione),
   invia i prompt dei **5 benchmark** attendendo ogni volta il prompt `>`, e
   misura energia, latenza, token/s e qualità;
6. salva tutto in **CSV** e genera i **grafici** di confronto con pandas,
   seaborn e scikit-learn.

## Modelli (6)

gemma-2-2b-it, Llama-3.2-3B-Instruct, qwen2.5-1.5b-instruct, ciascuno in Q4_K_M
e Q8_0.

Comando di lancio (per configurazione):

```
~/llama.cpp/build/bin/llama-cli -m <modello>.gguf -t N -c 512 -n 128
```

dove `N` è il numero di thread (1, 2 o 4). `llama-cli` si avvia in modalità
interattiva di default, quindi non serve il flag `-i`; eventuali argomenti
aggiuntivi si possono comunque impostare in `config.json` (`llama.extra_args`).

## Benchmark (5)

CommonSenseQA, BIG-Bench Hard, TruthfulQA, GSM8K, HumanEval — campioni curati in
`datasets/*.jsonl` (rispettivamente 15, 15, 15, 15 e 13 item, **73 prompt** in
totale). Per aumentarne il numero basta aggiungere righe ai JSONL: il loader li
conta automaticamente. Considerando 6 modelli × 3 configurazioni di thread × 3
ripetizioni, la campagna completa esegue $18 \times 73 \times 3 = 3942$ inferenze.

## Struttura

```
config.json                 parametri (Pi, Otii, llama, thermal, modelli, benchmark)
requirements.txt            dipendenze del venv host (Python 3.13)
datasets/*.jsonl            i 5 benchmark
run_benchmark.py            entry point eseguito dal PC host
scripts/
  otii_controller.py        controllo Otii Arc (API TCP): alimentazione, canali, energia
  pi_ssh.py                 SSH Paramiko + shell interattiva llama-cli + vcgencmd
  llama_parser.py           parsing timing (prompt/gen t/s) e testo generato
  datasets_loader.py        caricamento JSONL
  scoring.py                scoring per-benchmark (semplice, con flag revisione)
  thermal.py                monitoraggio termico + decode throttling + cooldown
  benchmark_runner.py       orchestrazione del flusso completo + CSV
  analysis.py               aggregazione e grafici (pandas/seaborn/scikit-learn)
  simulation.py             backend simulati per il dry-run (--simulate)
recordings/                 progetti Otii, CSV risultati, raw, plots
```

## Metriche

- **Efficienza energetica**: energia_netta (J) / token_generati → **J/token**,
  dove energia_netta = energia_misurata − energia_idle sulla finestra di
  inferenza (canale Main Power `mp` dell'Otii). Il numero di token generati non
  è riportato dal formato di output compatto di llama.cpp, quindi viene
  **approssimato con `n_predict`** (il massimo impostato da `-n`, cioè 128).
- **Latenza**: tempo wall-clock dell'inferenza (misurato dall'host) più le
  statistiche `prompt processing` e `generation` (token/s). Il parser riconosce
  il formato compatto realmente emesso dal build in uso
  (`Prompt: <x> t/s | Generation: <y> t/s`, con virgola o punto decimale) e, in
  subordine, il formato classico di llama.cpp.
- **Qualità**: punteggio per-benchmark (match della lettera, corrispondenza
  esatta, estrazione del valore numerico, esecuzione dei test per HumanEval).
  Le voci a bassa confidenza (TruthfulQA, codice) sono marcate `needs_review`
  per la rifinitura manuale successiva.

## Gestione termica e throttling

Il Pi 5 si scalda tra un'inferenza e l'altra: oltre ~80-85 °C entra in
**throttling termico** e cala le prestazioni, falsando latenza e t/s. Il
comportamento è configurabile nella sezione `thermal` di `config.json`:

- **Monitoraggio per inferenza** (`enabled`, `log_per_inference`): dopo ogni
  risposta vengono letti `vcgencmd measure_temp` e `vcgencmd get_throttled`;
  temperatura e stato di throttling finiscono nel CSV (`temp_c`, `throttled_hex`,
  `throttle_active`). `get_throttled` è decodificato bit a bit (under-voltage,
  ARM frequency capped, throttling, soft temp limit, sia "attivo ora" sia
  "avvenuto").
- **Rilevazione e ripetizione**: se durante una ripetizione si rileva throttling
  e `abort_on_throttle` è `true`, la registrazione viene **scartata e ripetuta**
  (fino a `max_retries`); le righe conservano comunque `throttle_event` per
  tracciabilità.
- **Cooldown** (`cooldown_enabled`): opzionalmente si attende che la SoC scenda
  sotto `cooldown_target_c` prima di ogni configurazione, con pausa minima e
  tetto massimo di attesa. L'attesa è **fuori dalla finestra di misura**, quindi
  non inquina l'energia.

Poiché il case del Raspberry Pi 5 monta una **ventola di raffreddamento attiva**,
nella configurazione di default il cooldown è **disattivato**
(`cooldown_enabled: false`) per massimizzare la velocità, mentre il monitoraggio
resta attivo come rete di sicurezza e per documentare l'assenza di throttling.
Il grafico `recordings/plots/thermal.png` mostra la distribuzione delle
temperature per modello/thread e l'analisi stampa un riepilogo termico con il
numero di misure eventualmente acquisite sotto throttling.

## Uso

Ambiente virtuale già presente: `envEnergyBenchmark` (Python 3.13). Se serve
ricrearlo, installare le dipendenze con `pip install -r requirements.txt`.

```bash
# attiva il venv (Windows)
envEnergyBenchmark\Scripts\activate

# prova della pipeline SENZA hardware (genera CSV + grafici simulati)
python run_benchmark.py --simulate

# esecuzione reale (Otii software con TCP server attivo su :1905,
# Otii Arc collegato, Pi cablato all'uscita e raggiungibile via Ethernet)
python run_benchmark.py

# opzioni utili
python run_benchmark.py --no-analysis   # salta la generazione dei grafici
python run_benchmark.py --verbose        # log dettagliato
```

Output principali:

- `recordings/results.csv` — una riga per ogni (modello, thread, ripetizione,
  benchmark, campione) con energia netta, latenza, token, J/token, prompt/gen
  t/s, score, temperatura e stato di throttling;
- `recordings/raw/aggregated_by_benchmark.csv`, `recordings/raw/ranking_composite.csv`;
- `recordings/plots/*.png` — efficienza (J/token), latenza, potenza, qualità
  (heatmap), trade-off efficienza/qualità, classifica composita e termico.

## Note operative

- Il costo energetico del **caricamento** del modello è registrato a parte
  (riga `__load__`), così l'energia di inferenza non lo include.
- Il **bias idle** va rimisurato se cambiano le condizioni (temperatura,
  periferiche collegate).
- Tutti i parametri (Pi, Otii, llama, thermal, modelli, thread, ripetizioni,
  benchmark) sono in `config.json`, modificabili senza toccare il codice.
- `config.json` contiene le credenziali SSH: trattalo come file riservato.
