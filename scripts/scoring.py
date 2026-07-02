"""Scoring semplice e per-benchmark della qualità delle risposte.

Filosofia: scoring di base automatico + revisione manuale
successiva. Ogni funzione restituisce un dizionario con:
  - score       : 0.0/1.0 (o frazione) confronto con l'atteso
  - confidence  : "high" se il match è affidabile, "low" se euristico
  - needs_review: True se conviene rivedere a mano / con un LLM giudice
"""
from __future__ import annotations

import re
import textwrap
from typing import Optional


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _first_choice_letter(text: str) -> Optional[str]:
    m = re.search(r"\b([A-E])\b", text.upper())
    return m.group(1) if m else None


def _last_number(text: str) -> Optional[float]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def _hashed_number(text: str) -> Optional[float]:
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text.replace(",", ""))
    if m:
        return float(m.group(1))
    return _last_number(text)


def score_multiple_choice(response: str, expected: str) -> dict:
    got = _first_choice_letter(response)
    ok = got is not None and got == expected.strip().upper()[:1]
    return {"score": 1.0 if ok else 0.0,
            "confidence": "high" if got else "low",
            "needs_review": got is None,
            "parsed": got}


def score_exact_match(response: str, expected: str) -> dict:
    r, e = _norm(response), _norm(expected)
    ok = e in r or r == e or r.startswith(e)
    return {"score": 1.0 if ok else 0.0,
            "confidence": "high",
            "needs_review": not ok,
            "parsed": r[:80]}


def score_numeric(response: str, expected: str) -> dict:
    got = _hashed_number(response)
    exp = _hashed_number(expected) if expected else None
    if got is None or exp is None:
        return {"score": 0.0, "confidence": "low",
                "needs_review": True, "parsed": got}
    ok = abs(got - exp) < 1e-6
    return {"score": 1.0 if ok else 0.0,
            "confidence": "high",
            "needs_review": False,
            "parsed": got}


def score_open_ended(response: str, expected: str, incorrect: Optional[list] = None) -> dict:
    """Euristica a bassa confidenza per TruthfulQA: overlap + check distrattori.

    Va sempre rivista a mano o con un LLM-giudice.
    """
    r = _norm(response)
    e = _norm(expected)
    exp_tokens = set(e.split())
    overlap = len(exp_tokens & set(r.split())) / max(1, len(exp_tokens))
    hit_wrong = any(_norm(w) in r for w in (incorrect or []))
    score = 0.0
    if overlap >= 0.5 and not hit_wrong:
        score = 1.0
    elif overlap >= 0.3 and not hit_wrong:
        score = 0.5
    return {"score": score, "confidence": "low",
            "needs_review": True, "parsed": round(overlap, 2)}


def score_code(response: str, sample_meta: dict) -> dict:
    """Esegue il test HumanEval in un sandbox locale (best-effort, pass@1).

    Costruisce: firma (dal prompt) + corpo generato + test. Se non compila o
    fallisce viene 0. Marcato comunque needs_review per controllo manuale.
    """
    test = sample_meta.get("test")
    prompt = sample_meta.get("prompt", "")
    entry = sample_meta.get("entry_point", "")
    if not test:
        return {"score": 0.0, "confidence": "low",
                "needs_review": True, "parsed": "no-test"}

    body = _extract_code(response, entry)
    # se il modello ha già incluso la firma def, la usiamo intera; altrimenti la
    # ricostruiamo dal prompt (firma + docstring) normalizzando l'indentazione
    # del corpo a 4 spazi (il modello può restituirlo già indentato o meno).
    if f"def {entry}" in body:
        program = body
    else:
        body = textwrap.indent(textwrap.dedent(body), "    ")
        program = prompt + body
    full = program + "\n\n" + test + "\n"
    ok = _run_sandbox(full)
    return {"score": 1.0 if ok else 0.0,
            "confidence": "high" if ok else "low",
            "needs_review": not ok,
            "parsed": "pass" if ok else "fail"}


def _extract_code(response: str, entry: str) -> str:
    """Estrae il codice da una risposta, gestendo i blocchi markdown ```."""
    m = re.search(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip("\n")
    return response


def _run_sandbox(source: str, timeout: int = 8) -> bool:
    import subprocess
    import sys
    import tempfile
    import os
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(source)
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, timeout=timeout, text=True,
        )
        return proc.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def score(btype: str, response: str, expected: str, meta: Optional[dict] = None) -> dict:
    meta = meta or {}
    if btype == "multiple_choice":
        return score_multiple_choice(response, expected)
    if btype == "exact_match":
        return score_exact_match(response, expected)
    if btype == "numeric":
        return score_numeric(response, expected)
    if btype == "open_ended":
        return score_open_ended(response, expected, meta.get("incorrect"))
    if btype == "code":
        return score_code(response, meta)
    return {"score": 0.0, "confidence": "low", "needs_review": True, "parsed": None}
