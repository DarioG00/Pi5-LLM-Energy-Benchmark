"""Verifica di integrita' dei file GGUF via sha256.

Serve ad accorgersi se un modello si corrompe (o viene sostituito) tra
un'esecuzione e l'altra. Non richiede di inserire a mano gli hash di Hugging
Face: alla PRIMA esecuzione calcola gli sha256 dei file presenti sul Pi e li
salva come riferimento (models.sha256.json); nelle esecuzioni successive li
riconfronta e segnala (o interrompe) se un hash e' cambiato.

E' un controllo eseguito UNA sola volta all'avvio della campagna, non a ogni
caricamento del modello: l'impatto sui tempi e' trascurabile.
"""
from __future__ import annotations

import json
import os
import re
import logging

log = logging.getLogger("integrity")

_HASH_RE = re.compile(r"([0-9a-fA-F]{64})")


def remote_sha256(ssh, path: str, timeout: float = 600.0):
    """Calcola lo sha256 di un file sul Pi (via `sha256sum`). None se non leggibile."""
    try:
        out = ssh.run_command(f"sha256sum {path}", timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        log.warning("sha256 non calcolabile per %s: %s", path, exc)
        return None
    m = _HASH_RE.match(out.strip())
    return m.group(1).lower() if m else None


def verify(ssh, cfg: dict, base_dir: str = ".") -> bool:
    """Verifica gli sha256 dei modelli rispetto alla baseline salvata.

    Ritorna True se tutto coincide (o alla prima esecuzione). Se `abort_on_mismatch`
    e' attivo e un hash e' cambiato, solleva RuntimeError.
    """
    vc = cfg.get("verify_models", {})
    if not vc.get("enabled", False):
        return True

    ref_path = os.path.join(base_dir, vc.get("reference", "models.sha256.json"))
    models_dir = cfg["llama"]["models_dir"].rstrip("/")

    ref = {}
    if os.path.exists(ref_path):
        try:
            ref = json.load(open(ref_path, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            ref = {}

    current, mismatches, missing = {}, [], []
    for m in cfg["models"]:
        f = m["file"]
        log.info("Verifica integrita' di %s ...", f)
        h = remote_sha256(ssh, f"{models_dir}/{f}")
        if h is None:
            missing.append(f)
            continue
        current[f] = h
        if f in ref and ref[f] != h:
            mismatches.append(f)

    # aggiorna la baseline aggiungendo SOLO i file nuovi (non sovrascrive quelli
    # gia' noti: un file corrotto continua a essere segnalato finche' non lo si
    # ripristina o non si aggiorna a mano la baseline).
    updated = dict(ref)
    for f, h in current.items():
        updated.setdefault(f, h)
    if updated != ref:
        try:
            json.dump(updated, open(ref_path, "w", encoding="utf-8"), indent=2)
            log.info("Baseline hash salvata/aggiornata in %s", ref_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Impossibile salvare la baseline: %s", exc)

    if missing:
        log.warning("Modelli non trovati/illeggibili sul Pi: %s", ", ".join(missing))
    if mismatches:
        log.error("HASH CAMBIATO (file GGUF corrotto o sostituito): %s", ", ".join(mismatches))
        if vc.get("abort_on_mismatch", True):
            raise RuntimeError(
                "Verifica integrita' fallita per: " + ", ".join(mismatches) +
                ". Riscarica il/i modello/i (o, se il cambiamento e' voluto, "
                f"elimina/aggiorna {ref_path}).")
        return False

    log.info("Integrita' dei modelli OK (%d file verificati).", len(current))
    return True
