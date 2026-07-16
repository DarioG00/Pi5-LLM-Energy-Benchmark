"""Orchestratore del benchmark energetico LLM su Raspberry Pi 5.

Flusso completo:
  1. apre la connessione al Pi (PMIC via SSH) e misura il consumo idle (bias);
  2. avvia llama-cli interattivo (energia di caricamento misurata a parte);
  3. per ogni modello × thread × ripetizione, per ogni prompt:
       - azzera la cronologia (/clear) così ogni inferenza e' indipendente,
       - invia il prompt attendendo il prompt `>`,
       - misura energia netta (PMIC) e latenza wall-clock per l'inferenza,
       - valuta la qualita' della risposta;
  4. salva tutto in CSV (media sulle ripetizioni in fase di analisi).
"""
from __future__ import annotations

import csv
import os
import time
import logging
from typing import Any, List, Optional

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
    "avg_power_w",
    "prompt_tps", "gen_tps", "score", "confidence", "needs_review", "parsed",
    "temp_c", "throttled_hex", "throttle_active", "throttle_event",
]


def build_command(llama_cfg: dict, model_file: str, threads: int) -> str:
    """Costruisce la riga di comando di llama-cli per un modello, con numero di thread, dimensione del contesto e limite di token."""
    bin_path = llama_cfg["bin"]
    models_dir = llama_cfg["models_dir"].rstrip("/")
    model_path = f"{models_dir}/{model_file}"
    extra = " ".join(llama_cfg.get("extra_args", []))
    return (
        f"{bin_path} -m {model_path} -t {threads} "
        f"-c {llama_cfg['ctx_size']} -n {llama_cfg['n_predict']} {extra}"
    ).strip()


class BenchmarkRunner:
    """Orchestratore della campagna: coordina misura del consumo (PMIC), esecuzione dei modelli via SSH, valutazione delle risposte e salvataggio dei risultati."""
    def __init__(self, config: dict, base_dir: str = ".",
                 power: Any = None, ssh: Any = None):
        """Inizializza il runner con la configurazione e i backend di misura (power) e comunicazione (ssh)."""
        self.cfg = config
        self.base_dir = base_dir
        self.power = power        # PmicMonitor (o simulato)
        self.ssh = ssh            # PiSSH (o simulato)
        self.samples: List[Sample] = []
        self.rows: List[dict] = []
        self.thermal_cfg = thermal.ThermalConfig.from_dict(config.get("thermal", {}))

    # --------------------------------------------------------------- setup
    def load_samples(self) -> None:
        """Carica i campioni di tutti i benchmark, applicando l'eventuale limite max_samples_per_benchmark."""
        self.samples = load_all(self.cfg["benchmarks"], self.base_dir)
        cap = int(self.cfg.get("max_samples_per_benchmark", 0) or 0)
        if cap > 0:
            seen: dict = {}
            kept = []
            for s in self.samples:
                c = seen.get(s.benchmark, 0)
                if c < cap:
                    kept.append(s)
                    seen[s.benchmark] = c + 1
            self.samples = kept
            log.info("Limite campioni per benchmark: %d", cap)
        log.info("Totale campioni: %d", len(self.samples))

    def setup_hardware(self) -> None:
        # misura consumo (PMIC): apre la connessione SSH dedicata al Pi
        """Prepara l'hardware: apre le connessioni, avvia la shell, verifica l'integrita' dei modelli e misura il consumo idle di riferimento."""
        self.power.connect()
        self.power.configure_power()
        self.power.power_on()
        # Pi (shell interattiva per llama-cli)
        self.ssh.wait_for_boot_and_connect()
        self.ssh.open_shell()
        # preflight: elimina eventuali llama-cli rimasti da una campagna
        # precedente interrotta (altrimenti occupano RAM e causano swap/OOM)
        self.ssh.kill_stray_llama()
        freem = self.ssh.free_mem_mb()
        if freem == freem:  # non NaN
            log.info("RAM disponibile sul Pi all'avvio: %.0f MiB", freem)
        # verifica integrita' dei modelli (sha256), una sola volta all'avvio
        from . import integrity
        integrity.verify(self.ssh, self.cfg, self.base_dir)
        # bias idle
        self.power.measure_idle_bias()

    # ----------------------------------------------------------------- run
    def run(self) -> None:
        """Esegue la campagna completa: per ogni modello x thread x ripetizione avvia una registrazione, ripetendola in caso di throttling."""
        llama_cfg = self.cfg["llama"]
        reps = int(self.cfg.get("repetitions", 3))
        tcfg = self.thermal_cfg
        # pausa di raffreddamento tra una registrazione e la successiva: fa
        # ripartire ogni misura da una temperatura piu' bassa, senza falsare i
        # dati (la finestra di misura e' per-inferenza, la pausa e' fuori).
        cooldown = int(self.cfg.get("cooldown_seconds", 0) or 0)
        first_run = True
        for model in self.cfg["models"]:
            for threads in self.cfg["threads"]:
                for rep in range(1, reps + 1):
                    # pausa prima di ogni registrazione tranne la primissima
                    if not first_run and cooldown > 0:
                        temp = self.ssh.measure_temp() if tcfg.enabled else float("nan")
                        if temp == temp:  # non NaN
                            log.info("Pausa di raffreddamento di %ds (T attuale %.1f°C)...",
                                     cooldown, temp)
                        else:
                            log.info("Pausa di raffreddamento di %ds...", cooldown)
                        time.sleep(cooldown)
                        if tcfg.enabled:
                            log.info("Ripresa (T dopo pausa %.1f°C).", self.ssh.measure_temp())
                    first_run = False
                    attempt = 0
                    while True:
                        attempt += 1
                        log.info("=== %s | t=%d | rip %d/%d (tentativo %d) ===",
                                 model["file"], threads, rep, reps, attempt)
                        rows, throttled = self._run_one(model, threads, rep, llama_cfg)
                        retry = (tcfg.enabled and tcfg.abort_on_throttle
                                 and throttled and attempt <= tcfg.max_retries)
                        if retry:
                            log.warning("Throttling rilevato durante la ripetizione: "
                                        "scarto e ripeto la registrazione.")
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

        self.power.new_project()
        self.power.start_recording()
        time.sleep(getattr(self.power, "settle_seconds", 3))

        # --- caricamento del modello ---
        # slate pulito: nessun llama-cli residuo deve competere per la RAM
        self.ssh.kill_stray_llama()
        freem = self.ssh.free_mem_mb()
        if freem == freem and freem < 1000:  # non NaN e sotto soglia di sicurezza
            log.warning("RAM disponibile bassa sul Pi (%.0f MiB): rischio OOM/swap", freem)
        t_load0 = self.power.mark()
        load_out = self.ssh.launch_model(command, llama_cfg.get("load_timeout", 180))
        t_load1 = self.power.mark()
        log.info("modello caricato in %.0fs; eseguo %d inferenze...",
                 t_load1 - t_load0, len(self.samples))

        # --- inferenze sui benchmark ---
        per_sample = []
        throttled_during = False
        n_tot = len(self.samples)
        for k, s in enumerate(self.samples, 1):
            log.info("  [%d/%d] %s %s ...", k, n_tot, s.benchmark, s.id)
            # azzera la cronologia: ogni prompt e' indipendente e non satura il contesto
            self.ssh.clear_history()
            # prepara il prompt PRIMA di aprire la finestra (l'eventuale overhead
            # di preparazione non deve entrare nell'energia/latenza dell'inferenza)
            self.ssh.stage_prompt(s.prompt)
            t0 = self.power.mark()
            wall0 = time.monotonic()
            raw = self.ssh.submit_prompt(llama_cfg.get("infer_timeout", 240))
            latency = time.monotonic() - wall0
            t1 = self.power.mark()
            # lettura termica DOPO la chiusura della finestra di misura
            tstate = {}
            if tcfg.enabled and tcfg.log_per_inference:
                tstate = thermal.read_state(self.ssh)
                now_occ = bool(tstate.get("throttled_value", 0) & 0xF0000)
                if tstate.get("active") or (now_occ and not base_occurred):
                    throttled_during = True
            # valuta subito la risposta (fuori dalla finestra di misura) per mostrare
            # il punteggio nell'avanzamento; il risultato viene riusato per il CSV.
            response = extract_response(raw, s.prompt)
            meta = dict(s.meta)
            meta.setdefault("prompt", s.prompt)
            sc = scoring.score(s.btype, response, s.expected, meta)
            sv = sc.get("score")
            esito = "OK" if sv == 1.0 else ("n/d" if sv in ("", None) else "NO")
            temp = tstate.get("temp_c") if tstate else None
            tstr = " | T=%.1f°C" % temp if isinstance(temp, (int, float)) else ""
            log.info("  [%d/%d] %s %s -> punteggio %s [%s]  (%.1fs)%s",
                     k, n_tot, s.benchmark, s.id, sv, esito, latency, tstr)
            per_sample.append((s, t0, t1, latency, raw, tstate, response, sc))

        # --- chiusura sessione e riepilogo timing ---
        summary = self.ssh.stop_model()
        sess_tim = parse_timings(load_out + "\n" + "\n".join(p[4] for p in per_sample) + "\n" + summary)

        rec = self.power.stop_recording()

        # --- riga di caricamento ---
        load_e = self.power.window_energy(rec, t_load0, t_load1)
        local_rows.append(self._mk_row(
            model, threads, rep, LOAD_TAG, "load", "", "", "",
            latency=load_e["duration_s"], energy=load_e,
            timings=sess_tim,
            score={"score": "", "confidence": "", "needs_review": "", "parsed": ""},
            tstate=base, throttle_event=throttled_during,
        ))

        # --- righe per ogni inferenza ---
        for (s, t0, t1, latency, raw, tstate, response, sc) in per_sample:
            energy = self.power.window_energy(rec, t0, t1)
            ptim = parse_timings(raw)
            timings = {
                "prompt_tps": ptim.get("prompt_tps") or sess_tim.get("prompt_tps"),
                "gen_tps": ptim.get("gen_tps") or sess_tim.get("gen_tps"),
            }
            local_rows.append(self._mk_row(
                model, threads, rep, s.benchmark, s.id, s.prompt,
                s.expected, response,
                latency=latency, energy=energy,
                timings=timings, score=sc,
                tstate=tstate, throttle_event=throttled_during,
            ))
        return local_rows, throttled_during

    def _mk_row(self, model, threads, rep, benchmark, sample_id, prompt,
                expected, response, latency, energy, timings, score,
                tstate=None, throttle_event=False) -> dict:
        """Compone una riga del CSV a partire da misure di energia, timing, stato termico e punteggio di una singola inferenza."""
        net = energy.get("energy_net_j", 0.0)
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
        """Salva tutte le righe raccolte nel file CSV dei risultati."""
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
        """Chiude in sicurezza la sessione del modello, la connessione SSH e il backend di misura."""
        try:
            self.ssh.stop_model()
        except Exception:
            pass
        try:
            self.ssh.close()
        except Exception:
            pass
        try:
            self.power.power_off()
            self.power.disconnect()
        except Exception:
            pass
