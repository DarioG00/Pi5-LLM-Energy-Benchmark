"""Orchestratore del benchmark energetico LLM su Raspberry Pi 5.

Flusso completo:
  1. configura e accende l'Otii Ace Pro (5 V, limite di corrente);
  2. alimenta il Pi e attende il boot, poi apre SSH (Paramiko);
  3. misura il consumo idle (bias);
  4. per ogni modello × thread × ripetizione:
       - avvia llama-cli interattivo (energia di caricamento misurata a parte),
       - invia i prompt dei 5 benchmark attendendo il prompt `>`,
       - misura energia/latenza per ciascuna inferenza,
       - calcola J/token e punteggio qualità;
  5. salva tutto in CSV (media sulle ripetizioni in fase di analisi).
"""
from __future__ import annotations

import csv
import os
import time
import logging
from typing import List, Optional

from . import scoring
from . import thermal
from .datasets_loader import Sample, load_all
from .llama_parser import parse_timings, extract_response

log = logging.getLogger("runner")

LOAD_TAG = "__load__"

CSV_FIELDS = [
    "model_file", "family", "quant", "threads", "repetition",
    "benchmark", "sample_id", "prompt", "expected", "response",
    "latency_s", "energy_total_j", "energy_idle_j", "energy_net_j",
    "avg_power_w", "gen_tokens", "j_per_token",
    "prompt_tps", "gen_tps", "score", "confidence", "needs_review", "parsed",
    "temp_c", "throttled_hex", "throttle_active", "throttle_event",
]


def build_command(llama_cfg: dict, model_file: str, threads: int) -> str:
    bin_path = llama_cfg["bin"]
    models_dir = llama_cfg["models_dir"].rstrip("/")
    model_path = f"{models_dir}/{model_file}"
    extra = " ".join(llama_cfg.get("extra_args", []))
    return (
        f"{bin_path} -m {model_path} -t {threads} "
        f"-c {llama_cfg['ctx_size']} -n {llama_cfg['n_predict']} {extra}"
    ).strip()


class BenchmarkRunner:
    def __init__(self, config: dict, base_dir: str = ".",
                 otii=None, ssh=None):
        self.cfg = config
        self.base_dir = base_dir
        self.otii = otii          # OtiiController (o simulato)
        self.ssh = ssh            # PiSSH (o simulato)
        self.samples: List[Sample] = []
        self.rows: List[dict] = []
        self.thermal_cfg = thermal.ThermalConfig.from_dict(config.get("thermal", {}))

    # --------------------------------------------------------------- setup
    def load_samples(self) -> None:
        self.samples = load_all(self.cfg["benchmarks"], self.base_dir)
        log.info("Totale campioni: %d", len(self.samples))

    def setup_hardware(self) -> None:
        # Otii
        self.otii.connect()
        self.otii.configure_power()
        self.otii.power_on()
        # Pi
        self.ssh.wait_for_boot_and_connect()
        self.ssh.open_shell()
        # bias idle
        self.otii.measure_idle_bias()

    # ----------------------------------------------------------------- run
    def run(self) -> None:
        llama_cfg = self.cfg["llama"]
        reps = int(self.cfg.get("repetitions", 3))
        tcfg = self.thermal_cfg
        for model in self.cfg["models"]:
            for threads in self.cfg["threads"]:
                for rep in range(1, reps + 1):
                    attempt = 0
                    while True:
                        attempt += 1
                        log.info("=== %s | t=%d | rip %d/%d (tentativo %d) ===",
                                 model["file"], threads, rep, reps, attempt)
                        # cooldown + attesa prima di registrare (fuori dalla
                        # finestra di misura, così non inquina l'energia)
                        if tcfg.enabled:
                            thermal.wait_until_cool(self.ssh, tcfg)
                        rows, throttled = self._run_one(model, threads, rep, llama_cfg)
                        retry = (tcfg.enabled and tcfg.abort_on_throttle
                                 and throttled and attempt <= tcfg.max_retries)
                        if retry:
                            log.warning("Throttling rilevato durante la ripetizione: "
                                        "scarto e ripeto dopo cooldown.")
                            continue
                        if throttled:
                            log.warning("Throttling rilevato: registro comunque "
                                        "(throttle_event=True nelle righe).")
                        self.rows.extend(rows)
                        break

    def _run_one(self, model: dict, threads: int, rep: int, llama_cfg: dict):
        """Esegue una registrazione completa. Ritorna (righe, throttled_durante)."""
        command = build_command(llama_cfg, model["file"], threads)
        tcfg = self.thermal_cfg
        local_rows: List[dict] = []

        # stato termico di base prima della registrazione
        base = thermal.read_state(self.ssh) if tcfg.enabled else {}
        base_occurred = bool(base.get("throttled_value", 0) & 0xF0000)

        self.otii.new_project()
        self.otii.start_recording()
        time.sleep(self.cfg["otii"].get("settle_seconds", 3))

        # --- caricamento del modello ---
        t_load0 = self.otii.mark()
        load_out = self.ssh.launch_model(command, llama_cfg.get("load_timeout", 180))
        t_load1 = self.otii.mark()

        # --- inferenze sui benchmark ---
        per_sample = []
        throttled_during = False
        for s in self.samples:
            t0 = self.otii.mark()
            wall0 = time.monotonic()
            raw = self.ssh.send_prompt(s.prompt, llama_cfg.get("infer_timeout", 240))
            latency = time.monotonic() - wall0
            t1 = self.otii.mark()
            # lettura termica DOPO la chiusura della finestra di misura
            tstate = {}
            if tcfg.enabled and tcfg.log_per_inference:
                tstate = thermal.read_state(self.ssh)
                now_occ = bool(tstate.get("throttled_value", 0) & 0xF0000)
                if tstate.get("active") or (now_occ and not base_occurred):
                    throttled_during = True
                # piccola pausa tra inferenze per limitare l'accumulo termico
                if tcfg.cooldown_min_s > 0:
                    time.sleep(min(tcfg.cooldown_min_s, 3))
            per_sample.append((s, t0, t1, latency, raw, tstate))

        # --- chiusura sessione e riepilogo timing ---
        summary = self.ssh.stop_model()
        sess_tim = parse_timings(load_out + "\n" + "\n".join(p[4] for p in per_sample) + "\n" + summary)

        rec = self.otii.stop_recording()

        # --- riga di caricamento ---
        load_e = self.otii.window_energy(rec, t_load0, t_load1)
        local_rows.append(self._mk_row(
            model, threads, rep, LOAD_TAG, "load", "", "", "",
            latency=load_e["duration_s"], energy=load_e,
            tokens=0, timings=sess_tim,
            score={"score": "", "confidence": "", "needs_review": "", "parsed": ""},
            tstate=base, throttle_event=throttled_during,
        ))

        # --- righe per ogni inferenza ---
        for (s, t0, t1, latency, raw, tstate) in per_sample:
            energy = self.otii.window_energy(rec, t0, t1)
            response = extract_response(raw, s.prompt)
            ptim = parse_timings(raw)
            # Il formato compatto dei t/s non riporta il numero di token: si
            # approssima con n_predict (-n), ossia il massimo di token generati.
            gen_tokens = ptim.get("gen_tokens") or llama_cfg.get("n_predict", 128)
            meta = dict(s.meta)
            meta.setdefault("prompt", s.prompt)
            sc = scoring.score(s.btype, response, s.expected, meta)
            timings = {
                "prompt_tps": ptim.get("prompt_tps") or sess_tim.get("prompt_tps"),
                "gen_tps": ptim.get("gen_tps") or sess_tim.get("gen_tps"),
            }
            local_rows.append(self._mk_row(
                model, threads, rep, s.benchmark, s.id, s.prompt,
                s.expected, response,
                latency=latency, energy=energy, tokens=gen_tokens,
                timings=timings, score=sc,
                tstate=tstate, throttle_event=throttled_during,
            ))
        return local_rows, throttled_during

    def _mk_row(self, model, threads, rep, benchmark, sample_id, prompt,
                expected, response, latency, energy, tokens, timings, score,
                tstate=None, throttle_event=False) -> dict:
        net = energy.get("energy_net_j", 0.0)
        jpt = (net / tokens) if tokens else ""
        tstate = tstate or {}
        temp = tstate.get("temp_c", "")
        return {
            "model_file": model["file"],
            "family": model.get("family", ""),
            "quant": model.get("quant", ""),
            "threads": threads,
            "repetition": rep,
            "benchmark": benchmark,
            "sample_id": sample_id,
            "prompt": prompt,
            "expected": expected,
            "response": response,
            "latency_s": round(latency, 4),
            "energy_total_j": round(energy.get("energy_total_j", 0.0), 6),
            "energy_idle_j": round(energy.get("energy_idle_j", 0.0), 6),
            "energy_net_j": round(net, 6),
            "avg_power_w": round(energy.get("avg_power_w", 0.0), 4),
            "gen_tokens": tokens,
            "j_per_token": round(jpt, 6) if jpt != "" else "",
            "prompt_tps": timings.get("prompt_tps"),
            "gen_tps": timings.get("gen_tps"),
            "score": score.get("score", ""),
            "confidence": score.get("confidence", ""),
            "needs_review": score.get("needs_review", ""),
            "parsed": score.get("parsed", ""),
            "temp_c": round(temp, 1) if isinstance(temp, (int, float)) and temp == temp else "",
            "throttled_hex": tstate.get("throttled_hex", ""),
            "throttle_active": tstate.get("active", ""),
            "throttle_event": throttle_event,
        }

    # ----------------------------------------------------------------- save
    def save_csv(self, path: Optional[str] = None) -> str:
        path = path or os.path.join(self.base_dir, self.cfg["output"]["results_csv"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)
        log.info("Risultati salvati in %s (%d righe)", path, len(self.rows))
        return path

    # ------------------------------------------------------------- teardown
    def teardown(self) -> None:
        try:
            self.ssh.stop_model()
        except Exception:
            pass
        try:
            self.ssh.close()
        except Exception:
            pass
        try:
            self.otii.power_off()
            self.otii.disconnect()
        except Exception:
            pass
