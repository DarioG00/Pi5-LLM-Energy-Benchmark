# Checklist esperimento — Lunedì

**Obiettivo:** eseguire le due campagne di misura e portare a casa i CSV + grafici.
**Finestra:** arrivo ore 9:00, uscita ore 18:00. Fine prevista dei due run ~15:15 (ampio margine).
**Regola d'oro:** si lanciano SOLO due comandi (uno per campagna). Non modificare `config.json`.

---

## 0) Pre-flight all'arrivo (~10 min)

- [ ] **Raspberry Pi acceso**, cavo Ethernet collegato, ventola in funzione.
- [ ] Aprire **PowerShell** nella cartella del progetto:
      `cd C:\Users\guidi\codes\Pi5-LLM-Energy-Benchmark`
- [ ] Attivare l'ambiente virtuale:
      `envEnergyBenchmark\Scripts\activate`
      (il prompt deve mostrare `(envEnergyBenchmark)`)
- [ ] **Verificare che il Pi risponda** (sostituisci con la password se richiesta):
      `ssh dario@192.168.10.2 "vcgencmd pmic_read_adc | head -3 && ls ~/LLMs"`
      → devi vedere valori `current(...)`/`volt(...)` **e** i 6 file `.gguf`.
- [ ] **Azzerare la baseline di integrità** (evita blocchi da hash non allineati; si ricrea da sola):
      `del models.sha256.json`
- [ ] **Mini-test del PMIC (~1-2 min) — fallo SEMPRE prima di partire.** Esegue un solo modello, 1 prompt per benchmark, e produce un CSV completo che puoi aprire:
```powershell
python run_benchmark.py --config config_test.json
```
      Poi apri `recordings\results_test.csv` e guarda la colonna `energy_net_j`:
      - valori **positivi** (es. 1–50 J) e `avg_power_w` ~3–9 W → **il PMIC misura correttamente**, puoi procedere;
      - valori **0 o negativi** o `energy_total_j = 0` → il PMIC non sta misurando: verifica la connessione al Pi prima di lanciare i run veri.
      **Chiudi il CSV** (soprattutto se aperto in Excel) prima di continuare.
- [ ] *(Facoltativo)* Prova che tutti i modelli rispondano: `python inspect_models.py --benchmarks gsm8k --samples 1`

---

## 1) RUN 1 — Campagna A (ore 9:00)

Confronto tra i 6 modelli, dataset intero (73 prompt), **2 thread**, 3 ripetizioni → 18 registrazioni.

- [ ] Lanciare (salva anche il log su file):
```powershell
python run_benchmark.py --config config_run1.json *>&1 | Tee-Object -FilePath run1.log
```
- [ ] **Durata attesa:** ~3h30 → fine verso le **12:30**.
- [ ] **Nota:** il CSV `results_run1.csv` viene scritto **solo a fine run** (non durante), quindi non c'è nulla da aprire mentre gira. La correttezza del PMIC l'hai già verificata col mini-test al passo 0.
- [ ] **Non aprire `results_run1.csv` in Excel** mentre lo script è in esecuzione: il lock di Excel può far fallire il salvataggio finale.
- [ ] Controllo dal vivo: nel log deve scorrere `punteggio ... [OK/NO]` per ogni inferenza e `=== <modello> | t=2 | rip k/3 ===` a ogni configurazione (vedi tabella dei tempi in fondo).
- [ ] Output attesi a fine run: `recordings\results_run1.csv`, cartella `recordings\plots_run1\` con i grafici.

---

## 2) Tra i due run (~12:30)

- [ ] Verificare che Run 1 sia finito **senza errori** (ultima riga del log: "Analisi completata").
- [ ] **Backup immediato** dei risultati di Run 1 (copiali fuori dal progetto, es. su USB/cloud):
      `copy recordings\results_run1.csv <cartella_backup>`
      (e la cartella `recordings\plots_run1\`)
- [ ] Non serve modificare nulla: Run 2 usa un config e output separati.

---

## 2bis) RUN 2 — Campagna B (~12:35)

Scaling sul parallelismo, 3 campioni/benchmark (15 prompt), **thread 1/2/4**, 3 ripetizioni → 54 registrazioni.

- [ ] Lanciare:
```powershell
python run_benchmark.py --config config_run2.json *>&1 | Tee-Object -FilePath run2.log
```
- [ ] **Durata attesa:** ~2h40 → fine verso le **15:15**.
- [ ] Output attesi: `recordings\results_run2.csv`, cartella `recordings\plots_run2\`.

---

## 3) Dopo i due run (prima di uscire)

- [ ] Verificare la presenza di **tutti** i file:
      - `recordings\results_run1.csv` e `recordings\results_run2.csv`
      - `recordings\plots_run1\` e `recordings\plots_run2\` (con i .png)
      - `run1.log` e `run2.log`
- [ ] **Backup finale completo**: copia l'intera cartella `recordings\` (più i due `.log`) in un posto sicuro (USB + cloud). *Questa è la cosa più importante: non lasciare il laboratorio senza una copia dei dati.*

---

## Cosa significano gli avvisi nel log

- `Campionamento PMIC: letture non valide, tento la riconnessione...` seguito da `ripristinato` → si è **auto-corretto**, va bene.
- `Registrazione PMIC senza campioni` → quella singola registrazione ha l'energia non valida; annota quale configurazione e, se hai tempo, rilancia SOLO quel run alla fine.
- `Throttling rilevato ... scarto e ripeto` → normale gestione termica, il run viene ripetuto in automatico.
- `HASH CAMBIATO ... Verifica integrità fallita` → NON dovrebbe capitare (hai cancellato la baseline al passo 0). Se capita: `del models.sha256.json` e rilancia lo stesso comando.

---

## Contingenze (se qualcosa va storto)

- **Un run si interrompe / crasha:** i risultati **parziali vengono comunque salvati** nel CSV. Puoi rilanciare lo stesso comando (ripartirà da capo, sovrascrivendo quel CSV).
- **Il Pi non risponde all'avvio:** controlla cavo Ethernet e alimentazione; ripeti il comando `ssh ...` del passo 0.
- **Poco tempo:** Run 1 (Campagna A) è il più importante per il confronto tra modelli — assicurati almeno che quello finisca e sia salvato.

---

## Da NON fare

- ❌ Non lanciare `python run_benchmark.py` **senza** `--config` (userebbe `config.json`, parametri sbagliati).
- ❌ Non modificare `config_run1.json` / `config_run2.json` / `config.json`.
- ❌ Non chiudere PowerShell né mettere il PC in sospensione durante un run.
- ❌ Non tenere aperti in **Excel** i CSV di output mentre lo script gira (il lock può bloccare il salvataggio finale).
- ❌ Non uscire dal laboratorio senza aver copiato la cartella `recordings\`.

---

### Riepilogo comandi (in ordine)
```powershell
cd C:\Users\guidi\codes\Pi5-LLM-Energy-Benchmark
envEnergyBenchmark\Scripts\activate
del models.sha256.json
python run_benchmark.py --config config_test.json                # test PMIC ~1 min -> apri results_test.csv
python run_benchmark.py --config config_run1.json *>&1 | Tee-Object -FilePath run1.log   # ore 9:00
python run_benchmark.py --config config_run2.json *>&1 | Tee-Object -FilePath run2.log   # ~12:35
```

---

## Tempi attesi per modello (controllo dal vivo)

Stime basate sulle latenze reali del run del 10 luglio. Nel log vedi passare i modelli
uno alla volta (`=== <modello> | t=... | rip k/3 ===`): confronta con questa tabella per
capire se sei in linea. `*` = qwen Q8 non era nel run del 10/7, valore stimato.

**RUN 1 — Campagna A (2 thread), in ordine di esecuzione.** Ogni modello gira **3 volte** (ripetizioni):

| # | Modello | min per una run | totale (×3 rip) |
|---|---|---|---|
| 1 | gemma-2-2b-it Q4_K_M      |  9,8 | ~29 min |
| 2 | Llama-3.2-3B-Instruct Q4_K_M | 13,7 | ~41 min |
| 3 | qwen2.5-1.5b-instruct Q4_K_M |  5,8 | ~17 min |
| 4 | gemma-2-2b-it Q8_0        | 12,7 | ~38 min |
| 5 | Llama-3.2-3B-Instruct Q8_0   | 18,7 | ~56 min |
| 6 | qwen2.5-1.5b-instruct Q8_0   | ~7,8* | ~23 min |
| | **Totale Run 1** | | **~3h25** |

**RUN 2 — Campagna B (thread 1/2/4).** Ogni modello gira **9 volte** (3 thread × 3 rip):

| # | Modello | totale (9 run) |
|---|---|---|
| 1 | gemma-2-2b-it Q4_K_M      | ~24 min |
| 2 | Llama-3.2-3B-Instruct Q4_K_M | ~33 min |
| 3 | qwen2.5-1.5b-instruct Q4_K_M | ~16 min |
| 4 | gemma-2-2b-it Q8_0        | ~27 min |
| 5 | Llama-3.2-3B-Instruct Q8_0   | ~38 min |
| 6 | qwen2.5-1.5b-instruct Q8_0   | ~22 min |
| | **Totale Run 2** | **~2h40** |

> Se un modello impiega **molto più** del tempo indicato, controlla il log: probabile
> throttling termico (ripetizioni scartate) o rallentamento della connessione.
> Le stime assumono nessun throttling; il margine fino alle 18:00 (~2h30) assorbe scostamenti.
