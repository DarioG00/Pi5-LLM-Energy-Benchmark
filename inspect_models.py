#!/usr/bin/env python3
"""Ispezione rapida delle risposte dei modelli.

Controllo visivo veloce che (a) tutti i modelli rispondano e (b) lo scoring di
`scripts/scoring.py` assegni i punteggi giusti. Per ogni modello e benchmark
stampa: prompt, risposta del modello, risposta attesa, punteggio con esito
[OK]/[NO], e a fine modello un riepilogo dei corretti per benchmark.

Usa la stessa modalita' interattiva del benchmark (llama-cli, /clear tra i
prompt) e stampa log di avanzamento in tempo reale.

Uso:
    python inspect_models.py                         # 1 campione/benchmark, 1 thread
    python inspect_models.py --samples 2 --threads 4
    python inspect_models.py --models qwen2.5-1.5b-instruct-q4_k_m.gguf
    python inspect_models.py --benchmarks truthfulqa gsm8k --out ispezione.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

from scripts.pi_ssh import PiSSH, PiConfig
from scripts.datasets_loader import load_all
from scripts.benchmark_runner import build_command
from scripts.llama_parser import parse_timings, extract_response
from scripts import scoring


def main() -> int:
    """Punto di ingresso dell'ispezione: per ogni modello e benchmark stampa prompt, risposta e punteggio, con i riepiloghi."""
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
        """Stampa una riga a schermo e, se richiesto, la scrive anche sul file di report."""
        print(txt, flush=True)
        if out:
            out.write(txt + "\n")

    load_to = float(llama.get("load_timeout", 180))
    infer_to = float(llama.get("infer_timeout", 240))

    # riepilogo globale: {benchmark: [corretti, totale]}
    globale = defaultdict(lambda: [0, 0])

    ssh = PiSSH(PiConfig.from_dict(cfg["pi"]),
                ready_prompt=llama.get("ready_prompt", "> "))
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
            if not load_out.strip():
                emit("!! nessun output all'avvio del modello: verificare il comando "
                     "o il percorso del binario/modello.")
            per_model = defaultdict(lambda: [0, 0])   # {benchmark: [corretti, totale]}
            k = 0
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
                    ok = (sc.get("score") == 1.0)
                    per_model[bname][1] += 1
                    globale[bname][1] += 1
                    if ok:
                        per_model[bname][0] += 1
                        globale[bname][0] += 1
                    shown = resp if len(resp) <= args.max_chars else resp[:args.max_chars] + " […]"
                    emit("-" * 90)
                    emit(f"[{bname}] id={s.id}  tipo={s.btype}   {'[OK]' if ok else '[NO]'}")
                    emit("PROMPT:")
                    emit(s.prompt)
                    emit("ATTESO: " + (s.expected[:200] if s.expected else "(verifica tramite test)"))
                    emit("RISPOSTA DEL MODELLO:")
                    emit(shown if shown else "(vuota / nessun output)")
                    emit(f"PUNTEGGIO: {sc.get('score')}  (parsed={sc.get('parsed')}, "
                         f"confidence={sc.get('confidence')})  "
                         f"t/s: prompt={tim.get('prompt_tps')} gen={tim.get('gen_tps')}")
            ssh.stop_model()
            tot_ok = sum(v[0] for v in per_model.values())
            tot = sum(v[1] for v in per_model.values())
            dettaglio = "  ".join(f"{b} {v[0]}/{v[1]}" for b, v in per_model.items())
            emit("-" * 90)
            emit(f"RIEPILOGO {m['file']}: {dettaglio}   |   TOTALE {tot_ok}/{tot} corretti")
            emit("")
    finally:
        try:
            ssh.stop_model()
        except Exception:
            pass
        ssh.close()

    # riepilogo globale (il file --out deve essere ancora aperto)
    if globale:
        emit("=" * 90)
        emit("RIEPILOGO GLOBALE (tutti i modelli):")
        for b, v in globale.items():
            emit(f"  {b}: {v[0]}/{v[1]} corretti")
        emit("=" * 90)
    if out:
        out.close()
    if args.out:
        print(f"\nReport salvato in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
