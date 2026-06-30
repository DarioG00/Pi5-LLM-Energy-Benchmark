"""Backend simulati per provare il flusso senza hardware (--simulate).

Riproducono l'interfaccia di OtiiController e PiSSH usata da BenchmarkRunner,
generando dati sintetici plausibili. Utile per validare la pipeline e l'analisi
prima di collegare l'Otii Arc e il Raspberry Pi.
"""
from __future__ import annotations

import time
import random


class FakeOtii:
    def __init__(self, cfg_otii: dict):
        self.cfg = cfg_otii
        self.idle_power_w = 2.6
        self._t0 = time.monotonic()
        self.id = "SIM-ARC"

    def connect(self): pass
    def configure_power(self): pass
    def power_on(self): pass
    def power_off(self): pass
    def disconnect(self): pass
    def new_project(self): self._t0 = time.monotonic(); return object()

    def start_recording(self):
        self._t0 = time.monotonic()

    def stop_recording(self):
        return object()

    def mark(self) -> float:
        return time.monotonic() - self._t0

    def measure_idle_bias(self) -> float:
        self.idle_power_w = round(random.uniform(2.3, 2.9), 3)
        return self.idle_power_w

    def window_energy(self, rec, t_from: float, t_to: float) -> dict:
        duration = max(1e-3, t_to - t_from)
        avg_power = round(random.uniform(4.5, 7.5), 3)  # W in carico
        energy_total = avg_power * duration
        energy_idle = self.idle_power_w * duration
        return {
            "duration_s": duration,
            "avg_power_w": avg_power,
            "energy_total_j": energy_total,
            "energy_idle_j": energy_idle,
            "energy_net_j": energy_total - energy_idle,
        }


# Formato compatto come quello emesso dal build di llama.cpp in uso.
_PERF = "Prompt: 200.00 t/s | Generation: 80.00 t/s\n"


class FakeSSH:
    def __init__(self, cfg_pi: dict, ready_prompt: str = "> "):
        self.cfg = cfg_pi
        self.ready_prompt = ready_prompt
        self._temp = 48.0          # °C simulati
        self._throttled = 0        # bitmask simulato

    def wait_for_boot_and_connect(self): time.sleep(0.05)
    def open_shell(self): pass

    # --- comandi paralleli (vcgencmd) simulati ---
    def run_command(self, cmd: str, timeout: float = 15.0) -> str:
        if "measure_temp" in cmd:
            return f"temp={self._temp:.1f}'C"
        if "get_throttled" in cmd:
            return f"throttled=0x{self._throttled:x}"
        return ""

    def measure_temp(self) -> float:
        # raffreddamento passivo quando interrogato di seguito (cooldown)
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
