#!/usr/bin/env python3
"""Entry point eseguito dal PC host per il benchmark energetico LLM su Pi 5.

Il consumo è misurato direttamente dal PMIC del Raspberry Pi 5
(`vcgencmd pmic_read_adc`, metodo jfikar/RPi5-power): nessun hardware di misura
esterno, il Pi è alimentato dal suo alimentatore ufficiale.

Esempi:
    python run_benchmark.py                 # esecuzione reale
    python run_benchmark.py --simulate      # prova della pipeline senza hardware
    python run_benchmark.py --no-analysis   # salta i grafici finali

Prerequisiti reali:
    - Raspberry Pi 5 acceso e raggiungibile via Ethernet all'IP di config.json;
    - accesso SSH abilitato; `vcgencmd pmic_read_adc` disponibile sul Pi.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from scripts.benchmark_runner import BenchmarkRunner


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def make_real_backends(cfg: dict):
    from scripts.pi_ssh import PiSSH, PiConfig
    from scripts.pmic import PmicMonitor, PmicConfig
    pi_cfg = PiConfig.from_dict(cfg["pi"])
    ssh = PiSSH(pi_cfg, ready_prompt=cfg["llama"].get("ready_prompt", "> "))
    monitor = PmicMonitor(pi_cfg, PmicConfig.from_dict(cfg.get("pmic", {})))
    return monitor, ssh


def make_sim_backends(cfg: dict):
    from scripts.simulation import FakePmic, FakeSSH
    ssh = FakeSSH(cfg["pi"], cfg["llama"].get("ready_prompt", "> "))
    return FakePmic(cfg.get("pmic", {})), ssh


def main() -> int:
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
