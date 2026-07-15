#!/usr/bin/env python3
"""Entry point eseguito dal PC host per il benchmark energetico LLM su Pi 5.

Orchestra dal PC host, via SSH, l'esecuzione dei modelli su llama.cpp sul Pi 5,
misurando per ogni inferenza energia (J), latenza e qualita' della risposta.
Il consumo e' misurato direttamente dal PMIC del Raspberry Pi 5
(`vcgencmd pmic_read_adc`, metodo jfikar/RPi5-power): nessun hardware di misura
esterno, il Pi e' alimentato dal suo alimentatore ufficiale.

--------------------------------------------------------------------------------
USO
--------------------------------------------------------------------------------
    python run_benchmark.py [--config FILE] [--simulate] [--no-analysis] [-v]

Argomenti:
    --config FILE   File di configurazione JSON da usare (default: config.json).
                    Determina modelli, thread, ripetizioni, benchmark, PMIC e
                    cartelle di output. Vedi sotto i config disponibili.
    --simulate      Dry-run senza hardware: usa backend finti (FakePmic/FakeSSH),
                    non serve il Pi. Disattiva automaticamente la verifica sha256
                    dei modelli. Utile per provare la pipeline e i grafici.
    --no-analysis   Esegue solo la raccolta dati e salva il CSV, senza generare
                    i grafici finali (l'analisi si puo' rilanciare a parte, vedi).
    -v, --verbose   Log di livello DEBUG invece di INFO.

--------------------------------------------------------------------------------
CONFIG DISPONIBILI
--------------------------------------------------------------------------------
    config.json            Base: 6 modelli, thread [1,2,4], 3 rip, 5 prompt/bench.
    config_test.json       Mini-test PMIC: 1 modello, 1 thread, 1 rip, 1 prompt.
                           Da lanciare prima di una campagna per verificare che
                           l'energia netta misurata sia positiva.
    config_run1.json       Campagna A (originale): dataset intero, thread [2],
                           3 ripetizioni. Confronto tra i sei modelli.
    config_run2.json       Campagna B (originale): thread [1,2,4], 3 rip,
                           3 prompt/bench. Effetto del parallelismo.
    config_run1_ext.json   Campagna A ESTESA: 30 prompt/bench, thread [2], 3 rip
                           (nessun max_samples -> dataset intero).  ~6,2 h.
    config_run2_ext.json   Campagna B ESTESA: 30 prompt/bench, thread [1,2,4],
                           1 ripetizione.  ~8,0 h.

    Nota: se un config NON contiene "max_samples_per_benchmark", vengono usati
    TUTTI i prompt del dataset (il parametro e' opzionale, default = nessun limite).

--------------------------------------------------------------------------------
ESEMPI
--------------------------------------------------------------------------------
    # Prova della pipeline senza hardware (nessun Pi richiesto):
    python run_benchmark.py --simulate

    # Test rapido del PMIC prima di una campagna lunga:
    python run_benchmark.py --config config_test.json

    # Le due campagne estese (piu' dati):
    python run_benchmark.py --config config_run1_ext.json
    python run_benchmark.py --config config_run2_ext.json

    # Salvare l'intero output della console in un file di log (PowerShell):
    python run_benchmark.py --config config_run1_ext.json *> run1_ext.log

    # Solo raccolta dati, grafici da rigenerare dopo:
    python run_benchmark.py --config config_run1_ext.json --no-analysis
    python -m scripts.analysis recordings/results_run1_ext.csv config_run1_ext.json

--------------------------------------------------------------------------------
OUTPUT
--------------------------------------------------------------------------------
    - CSV dei risultati e cartelle grafici/dati grezzi nei percorsi indicati dal
      config (es. recordings/results_run1_ext.csv, plots_run1_ext/, raw_run1_ext/).
    - Interruzione con Ctrl+C: i risultati parziali gia' raccolti vengono salvati.
    - Codice di ritorno 0 se e' stato prodotto un CSV, 1 altrimenti.

--------------------------------------------------------------------------------
PREREQUISITI (esecuzione reale)
--------------------------------------------------------------------------------
    - Raspberry Pi 5 acceso e raggiungibile via Ethernet all'IP di config.json;
    - accesso SSH abilitato; `vcgencmd pmic_read_adc` disponibile sul Pi;
    - modelli GGUF presenti sul Pi nei percorsi indicati dal config.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from scripts.benchmark_runner import BenchmarkRunner


def load_config(path: str) -> dict:
    """Legge il file di configurazione JSON e lo restituisce come dizionario."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def make_real_backends(cfg: dict):
    """Crea i backend reali (PmicMonitor e PiSSH) per l'esecuzione sull'hardware."""
    from scripts.pi_ssh import PiSSH, PiConfig
    from scripts.pmic import PmicMonitor, PmicConfig
    pi_cfg = PiConfig.from_dict(cfg["pi"])
    ssh = PiSSH(pi_cfg, ready_prompt=cfg["llama"].get("ready_prompt", "> "))
    monitor = PmicMonitor(pi_cfg, PmicConfig.from_dict(cfg.get("pmic", {})))
    return monitor, ssh


def make_sim_backends(cfg: dict):
    """Crea i backend simulati (FakePmic e FakeSSH) per il dry-run senza hardware."""
    from scripts.simulation import FakePmic, FakeSSH
    ssh = FakeSSH(cfg["pi"], cfg["llama"].get("ready_prompt", "> "))
    return FakePmic(cfg.get("pmic", {})), ssh


def main() -> int:
    """Punto di ingresso: carica la configurazione, esegue la campagna di benchmark e avvia l'analisi finale."""
    ap = argparse.ArgumentParser(description="Benchmark energetico LLM su Raspberry Pi 5")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--simulate", action="store_true",
                    help="usa backend simulati (nessun hardware richiesto)")
    ap.add_argument("--no-analysis", action="store_true",
                    help="non generare i grafici al termine")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("main")

    base_dir = os.path.dirname(os.path.abspath(args.config)) or "."
    cfg = load_config(args.config)

    if args.simulate:
        # In simulazione l'SSH e' finto e restituisce hash fittizi: la verifica
        # sha256 dei modelli reali non ha senso (non coincide con la baseline).
        cfg.setdefault("verify_models", {})["enabled"] = False
        log.info("Modalita' simulazione: verifica integrita' dei modelli disattivata.")

    power, ssh = make_sim_backends(cfg) if args.simulate else make_real_backends(cfg)
    runner = BenchmarkRunner(cfg, base_dir=base_dir, power=power, ssh=ssh)

    csv_path = None
    try:
        runner.load_samples()
        runner.setup_hardware()
        runner.run()
    except KeyboardInterrupt:
        log.warning("Interrotto dall'utente: salvo i risultati parziali...")
    except Exception as exc:  # noqa: BLE001
        log.exception("Errore durante il benchmark: %s", exc)
    finally:
        if runner.rows:
            csv_path = runner.save_csv()
        runner.teardown()

    if csv_path and not args.no_analysis:
        try:
            from scripts.analysis import run_analysis
            run_analysis(csv_path, cfg, base_dir)
        except Exception as exc:  # noqa: BLE001
            log.warning("Analisi non completata: %s", exc)

    return 0 if csv_path else 1


if __name__ == "__main__":
    sys.exit(main())
