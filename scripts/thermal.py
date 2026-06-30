"""Monitoraggio termico e gestione del throttling del Raspberry Pi 5.

Decodifica `vcgencmd get_throttled` (bitmask) e implementa una strategia di
cooldown per evitare che il surriscaldamento o il throttling termico falsino le
misure di performance ed energia.

Bit di `get_throttled` (Raspberry Pi):
    0  under-voltage attiva ora
    1  frequenza ARM limitata ora
    2  throttling attivo ora
    3  soft temperature limit attivo ora
    16 under-voltage avvenuta (sticky)
    17 frequenza ARM limitata avvenuta (sticky)
    18 throttling avvenuto (sticky)
    19 soft temperature limit avvenuto (sticky)
"""
from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass

log = logging.getLogger("thermal")

_NOW_BITS = {
    0: "under_voltage_now",
    1: "arm_freq_capped_now",
    2: "throttled_now",
    3: "soft_temp_limit_now",
}
_OCC_BITS = {
    16: "under_voltage_occurred",
    17: "arm_freq_capped_occurred",
    18: "throttled_occurred",
    19: "soft_temp_limit_occurred",
}


@dataclass
class ThermalConfig:
    enabled: bool = True          # monitoraggio termico (lettura temp/throttle)
    cooldown_enabled: bool = True # attesa di raffreddamento tra configurazioni
    max_temp_c: float = 70.0
    cooldown_target_c: float = 60.0
    cooldown_min_s: float = 5.0
    cooldown_max_wait_s: float = 300.0
    poll_s: float = 5.0
    log_per_inference: bool = True
    abort_on_throttle: bool = True
    max_retries: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> "ThermalConfig":
        known = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**known)


def parse_throttled(raw: str) -> dict:
    """Decodifica l'output di get_throttled in flag booleani.

    Ritorna un dizionario con il valore esadecimale, ogni flag e due sintesi:
    `active` (un qualsiasi bit 'now' attivo) e `occurred` (un qualsiasi bit
    'sticky').
    """
    m = re.search(r"throttled=0x([0-9a-fA-F]+)", raw or "")
    value = int(m.group(1), 16) if m else 0
    out = {"throttled_hex": f"0x{value:x}", "throttled_value": value}
    active = False
    occurred = False
    for bit, name in _NOW_BITS.items():
        flag = bool(value & (1 << bit))
        out[name] = flag
        active = active or flag
    for bit, name in _OCC_BITS.items():
        flag = bool(value & (1 << bit))
        out[name] = flag
        occurred = occurred or flag
    out["active"] = active
    out["occurred"] = occurred
    return out


def read_state(ssh) -> dict:
    """Legge temperatura + stato throttling correnti dal Pi via SSH."""
    temp = ssh.measure_temp()
    thr = parse_throttled(ssh.get_throttled_raw())
    thr["temp_c"] = temp
    return thr


def wait_until_cool(ssh, cfg: ThermalConfig) -> dict:
    """Attende che la temperatura scenda sotto `cooldown_target_c`.

    Rispetta una pausa minima (`cooldown_min_s`) e un tetto massimo di attesa
    (`cooldown_max_wait_s`). Ritorna lo stato termico finale.
    """
    if not cfg.enabled or not cfg.cooldown_enabled:
        return {}
    # pausa minima sempre applicata tra una configurazione e l'altra
    if cfg.cooldown_min_s > 0:
        time.sleep(cfg.cooldown_min_s)

    deadline = time.time() + cfg.cooldown_max_wait_s
    state = read_state(ssh)
    temp = state.get("temp_c", float("nan"))
    if temp != temp:  # NaN: impossibile leggere la temperatura
        log.warning("Temperatura non leggibile: salto il cooldown attivo.")
        return state

    while temp > cfg.cooldown_target_c and time.time() < deadline:
        log.info("Cooldown: %.1f°C > target %.1f°C, attendo %.0fs...",
                 temp, cfg.cooldown_target_c, cfg.poll_s)
        time.sleep(cfg.poll_s)
        state = read_state(ssh)
        temp = state.get("temp_c", float("nan"))
        if temp != temp:
            break
    log.info("Pronto a registrare a %.1f°C (throttled=%s)",
             state.get("temp_c", float("nan")), state.get("throttled_hex"))
    return state
