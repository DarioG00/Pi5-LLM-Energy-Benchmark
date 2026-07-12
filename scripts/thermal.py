"""Monitoraggio termico e rilevazione del throttling del Raspberry Pi 5.

Decodifica `vcgencmd get_throttled` (bitmask) e legge la temperatura della SoC,
così da registrare lo stato termico di ogni inferenza e riconoscere eventuali
episodi di throttling (che falserebbero latenza e velocità).

Il Raspberry Pi 5 usato monta una ventola di raffreddamento attiva: la temperatura
resta ampiamente sotto le soglie di throttling e non è quindi previsto alcun
cool-down attivo tra le configurazioni. Il modulo si limita a monitorare.

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
    """Parametri di gestione termica: monitoraggio, log per inferenza, ripetizione su throttling e numero massimo di tentativi."""
    enabled: bool = True           # monitoraggio termico (lettura temp/throttle)
    log_per_inference: bool = True # registra temp/throttle a ogni inferenza
    abort_on_throttle: bool = True # scarta e ripete la registrazione se throttling
    max_retries: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> "ThermalConfig":
        """Crea una ThermalConfig da un dizionario, ignorando le chiavi non pertinenti."""
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
