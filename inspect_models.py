#!/usr/bin/env python3
"""Ispezione rapida delle risposte dei modelli.

Per ogni modello e per ogni benchmark stampa: il prompt inviato, la risposta
effettiva del modello e il punteggio assegnato dallo scoring. Serve a capire
cosa risponde davvero il modello (utile per diagnosticare punteggi bassi).

Usa la stessa modalita' interattiva del benchmark: avvia il modello una volta,
invia i prompt uno per uno attendendo la riga dei timing e ne raccoglie la
risposta, azzerando la cronologia tra un prompt e l'altro (/clear). Stampa log
di avanzamento in tempo reale.

Uso:
    python inspect_models.py                         # 1 campione/benchmark, 1 thread
    python inspect_models.py --samples 2 --threads 4
    python inspect_models.py --models qwen2.5-1.5b-instruct-q4_k_m.gguf
    python inspect_models.py --benchmarks gsm8k humaneval --out ispezione.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from scripts.pi_ssh import PiSSH, PiConfig
from scripts.datasets_loader import load_all
from scripts.benchmark_runner import build_command
from scripts.llama_parser import parse_timings, extract_response
from scripts import scoring


def main() -> int:
    ap = argparse.ArgumentParser(description="Ispezione risposte dei modelli")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--samples", type=int, default=1,
                    help="numero di campioni per benchmark (default 1)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--models", nargs="*", default=None,
                    help="filtra per nome file dei modelli")
    ap.add_argument("--benchmarks", nargs="*", default=None,
                    help="filtra per nome dei benchmark")
    ap.add_argument("--max-chars", type=int, default=1500,
                    help="lunghezza massima della risposta stampata")
    ap.add_argument("--out", default=None, help="salva anche su file di testo")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("inspect")

    base = os.path.dirname(os.path.abspath(args.config)) or "."
    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    llama = cfg["llama"]

    samples = load_all(cfg["benchmarks"], base)
    by_bench: dict = {}
    for s in samples:
        if args.benchmarks and s.benchmark not in args.benchmarks:
            continue
        by_bench.setdefault(s.benchmark, [])
        if len(by_bench[s.benchmark]) < args.samples:
            by_bench[s.benchmark].append(s)

    models = cfg["models"]
    if args.models:
        models = [m for m in models if m["file"] in args.models]

    n_prompts = sum(len(v) for v in by_bench.values())

    out = open(args.out, "w", encoding="utf-8") if args.out else None

    def emit(txt: str = "") -> None:
        print(txt, flush=True)
        if out:
            out.write(txt + "\n")

    load_to = float(llama.get("load_timeout", 180))
    infer_to = float(llama.get("infer_timeout", 240))

    ssh = PiSSH(PiConfig.from_dict(cfg["pi"]),
                ready_prompt=llama.get("ready_prompt", "> "),
                prompt_mode=llama.get("prompt_mode", "inline"))
    ssh.wait_for_boot_and_connect()
    ssh.open_shell()
    try:
        for mi, m in enumerate(models, 1):
            log.info("[modello %d/%d] carico %s a %d thread "
                     "(il caricamento richiede alcune decine di secondi)...",
                     mi, len(models), m["file"], args.threads)
            emit("=" * 90)
            emit(f"MODELLO: {m['file']}  (family={m.get('family')}, "
                 f"quant={m.get('quant')})  thread={args.threads}")
            emit("=" * 90)
            cmd = build_command(llama, m["file"], args.threads)
            emit(f"$ {cmd}")
            load_out = ssh.launch_model(cmd, load_to)
            log.info("modello pronto, eseguo %d prompt...", n_prompts)
            k = 0
            if not load_out.strip():
                emit("!! nessun output all'avvio del modello: verificare il comando "
                     "o il percorso del binario/modello.")
            for bname, slist in by_bench.items():
                for s in slist:
                    k += 1
                    log.info("  [%d/%d] %s id=%s: invio prompt...", k, n_prompts, bname, s.id)
                    ssh.clear_history()
                    raw = ssh.send_prompt(s.prompt, infer_to)
                    resp = extract_response(raw, s.prompt)
                    tim = parse_timings(raw)
                    meta = dict(s.meta)
                    meta.setdefault("prompt", s.prompt)
                    sc = scoring.score(s.btype, resp, s.expected, meta)
                    shown = resp if len(resp) <= args.max_chars else resp[:args.max_chars] + " […]"
                    emit("-" * 90)
                    emit(f"[{bname}] id={s.id}  tipo={s.btype}")
                    emit("PROMPT:")
                    emit(s.prompt)
                    emit("ATTESO: " + (s.expected[:200] if s.expected else "(verifica tramite test)"))
                    emit("RISPOSTA DEL MODELLO:")
                    emit(shown if shown else "(vuota / nessun output)")
                    emit(f"PUNTEGGIO: {sc.get('score')}  (parsed={sc.get('parsed')}, "
                         f"confidence={sc.get('confidence')})  "
                         f"t/s: prompt={tim.get('prompt_tps')} gen={tim.get('gen_tps')}")
            ssh.stop_model()
            emit("")
    finally:
        try:
            ssh.stop_model()
        except Exception:
            pass
        ssh.close()
        if out:
            out.close()
    if args.out:
        print(f"\nReport salvato in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
