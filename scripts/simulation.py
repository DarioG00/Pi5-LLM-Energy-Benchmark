"""Backend simulati per provare il flusso senza hardware (--simulate).

Riproducono l'interfaccia di PmicMonitor e PiSSH usata da BenchmarkRunner,
generando dati sintetici plausibili. Utile per validare la pipeline e l'analisi
prima di collegare il Raspberry Pi.
"""
from __future__ import annotations

import time
import random


class FakePmic:
    """Backend PMIC simulato: stessa interfaccia di PmicMonitor.

    Genera potenze (W) plausibili come somma grezza del PMIC
    (idle ~3-4 W, carico ~8-12 W).
    """
    def __init__(self, cfg_pmic: dict):
        self.cfg = cfg_pmic or {}
        self.settle_seconds = self.cfg.get("settle_seconds", 3)
        self.idle_power_w = 3.5
        self._t0 = time.monotonic()

    def connect(self): pass
    def configure_power(self): pass
    def power_on(self): pass
    def power_off(self): pass
    def disconnect(self): pass
    def new_project(self): return None

    def start_recording(self):
        self._t0 = time.monotonic()

    def stop_recording(self):
        return []          # i campioni non servono al backend simulato

    def mark(self) -> float:
        return time.monotonic() - self._t0

    def measure_idle_bias(self) -> float:
        self.idle_power_w = round(random.uniform(3.0, 4.0), 3)
        return self.idle_power_w

    def window_energy(self, rec, t_from: float, t_to: float) -> dict:
        duration = max(1e-3, t_to - t_from)
        avg_power = round(random.uniform(7.5, 12.0), 3)   # W (somma PMIC)
        energy_total = avg_power * duration
        energy_idle = self.idle_power_w * duration
        return {
            "duration_s": duration,
            "avg_power_w": avg_power,
            "energy_total_j": energy_total,
            "energy_idle_j": energy_idle,
            "energy_net_j": energy_total - energy_idle,
        }


# Riga dei timing emessa dal build recente di llama.cpp (parentesi, virgola).
_PERF = "[ Prompt: 16,4 t/s | Generation: 15,8 t/s ]\n"


class FakeSSH:
    def __init__(self, cfg_pi: dict, ready_prompt: str = "> "):
        self.cfg = cfg_pi
        self.ready_prompt = ready_prompt
        self._temp = 48.0          # °C simulati
        self._throttled = 0        # bitmask simulato

    def wait_for_boot_and_connect(self): time.sleep(0.05)
    def open_shell(self): pass
    def clear_history(self): pass
    def stage_prompt(self, prompt): self._staged = prompt
    def submit_prompt(self, infer_timeout): return self.send_prompt(getattr(self, '_staged', ''), infer_timeout)

    # --- comandi paralleli (vcgencmd) simulati ---
    def run_command(self, cmd: str, timeout: float = 15.0) -> str:
        if "measure_temp" in cmd:
            return f"temp={self._temp:.1f}'C"
        if "get_throttled" in cmd:
            return f"throttled=0x{self._throttled:x}"
        if "sha256sum" in cmd:
            import hashlib
            name = cmd.split()[-1]
            return hashlib.sha256(name.encode()).hexdigest() + "  " + name
        if "pmic_read_adc" in cmd:
            i = random.uniform(0.3, 1.1)   # corrente VDD_CORE simulata
            return (
                "   VDD_CORE_A current(7)=%.6fA\n"
                "   VDD_CORE_V volt(15)=0.900000V\n" % i
            )
        return ""

    def measure_temp(self) -> float:
        # la temperatura simulata cala leggermente a ogni lettura
        self._temp = max(45.0, self._temp - random.uniform(1.5, 3.5))
        return self._temp

    def get_throttled_raw(self) -> str:
        # occasionalmente simula un evento di throttling/under-voltage
        if self._temp > 80 or random.random() < 0.03:
            self._throttled |= (1 << 2) | (1 << 18)
        return f"throttled=0x{self._throttled:x}"

    def _heat(self):
        self._temp = min(88.0, self._temp + random.uniform(2.0, 6.0))

    def launch_model(self, command: str, load_timeout: float) -> str:
        time.sleep(random.uniform(0.05, 0.15))  # caricamento simulato
        self._heat()
        return f"main: loading model...\nsystem_info: ...\n> "

    def send_prompt(self, prompt: str, infer_timeout: float) -> str:
        time.sleep(random.uniform(0.02, 0.08))  # generazione simulata
        self._heat()
        fake = random.choice([
            "The answer is B", "True", "Yes", "#### 72",
            "return len(string)", "Washington, D.C.",
        ])
        return f"{prompt} {fake}\n{_PERF}> "

    def stop_model(self) -> str:
        return _PERF

    def close(self): pass
