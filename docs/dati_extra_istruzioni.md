# Dati aggiuntivi (richiesta del professore) — istruzioni

Obiettivo: portare i dataset da 15 a **30 prompt/benchmark** e rieseguire le due
campagne con più dati, **senza toccare** i dati e la tesi attuali (che restano
validi finché non si rigenerano).

Tutto è già predisposto. Servono tre passi.

## 1) Ampliare i dataset (sul PC, serve internet) — pochi minuti

Lo script usa la libreria ufficiale **`datasets`** (canale affidabile dell'Hub HF,
non il datasets-server che dava 503). Da installare una volta sola:

```powershell
envEnergyBenchmark\Scripts\activate
pip install datasets                   # solo la prima volta
python espandi_dataset.py --check      # (facoltativo) mostra alcuni item SENZA scrivere
python espandi_dataset.py              # scrive i dataset a 30 item
```

La prima esecuzione scarica e mette in cache i dataset (qualche decina di MB),
quindi può metterci un minuto o due; le volte successive è immediato.

- Fa automaticamente un **backup** dei dataset attuali in `datasets_bak/`
  (per tornare indietro basta ricopiarli).
- Mantiene i 15 prompt esistenti e ne aggiunge 15 nuovi dai benchmark ufficiali
  (Hugging Face), con lo **stesso formato e criterio di scoring**.
- Controlla qualche item, in particolare quelli di **BBH** (è il benchmark con la
  formattazione più varia). Se un benchmark desse errore di rete, lo script lo
  lascia invariato e lo segnala: si può rilanciare.

## 2) Rieseguire le campagne (in laboratorio, sul Pi)

Stessa procedura del run precedente (checklist `docs/checklist_lunedi.md`), ma con
i due config nuovi (output separati, non sovrascrivono i risultati attuali):

```powershell
del models.sha256.json                                   # baseline pulita (come sempre)
python run_benchmark.py --config config_run1_ext.json    # Campagna A: 30/bench, 2 thread, 3 rip  (~6,2 h)
python run_benchmark.py --config config_run2_ext.json    # Campagna B: 30/bench, thread 1/2/4, 1 rip (~8,0 h)
```

Totale ~**14 ore** (distribuibili su più sessioni; ognuno dei due può girare
anche di notte). Output in `recordings/results_run1_ext.csv` e
`recordings/results_run2_ext.csv` (+ cartelle `plots_run*_ext`).

## 3) Aggiornare la tesi

Mi porti i due CSV (`results_run1_ext.csv`, `results_run2_ext.csv`) e io rigenero
grafici, numeri e testo dei Risultati/Conclusioni con i nuovi dati.

---

### Note
- I config attuali (`config_run1.json`, `config_run2.json`) e i risultati attuali
  restano intatti: la tesi corrente è sempre riproducibile.
- Campagna A a **3 ripetizioni** (campagna primaria, stabilizza i modelli piccoli);
  Campagna B a **1 ripetizione** ma **30 prompt distinti** — statisticamente più
  solida di pochi prompt ripetuti (la variabilità tra prompt domina quella tra
  ripetizioni). Questa scelta va spiegata in tesi e la aggiungerò.
- Se il professore volesse un numero diverso da 30: `python espandi_dataset.py --n 50`.
