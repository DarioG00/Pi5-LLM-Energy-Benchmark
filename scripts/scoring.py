"""Scoring semplice e per-benchmark della qualita' delle risposte.

Filosofia: scoring di base automatico + revisione manuale
successiva. Ogni funzione restituisce un dizionario con:
  - score       : 0.0/1.0 (o frazione) confronto con l'atteso
  - confidence  : "high" se il match e' affidabile, "low" se euristico
  - needs_review: True se conviene rivedere a mano
"""
from __future__ import annotations

import re
import textwrap
from typing import Optional


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _first_choice_letter(text: str) -> Optional[str]:
    """Estrae la lettera dell'opzione scelta (A-E), con euristiche robuste.

    Cerca prima formulazioni esplicite ("Answer: B", "the answer is B",
    "(B)", "B)"); se non le trova, ripiega sulla prima lettera A-E isolata.
    """
    up = text.upper()
    patterns = [
        r"ANSWER\s*(?:IS|:)?\s*\(?([A-E])\)?",
        r"\b(?:OPTION|RISPOSTA)\s*\(?([A-E])\)?",
        r"\(([A-E])\)",
        r"\b([A-E])[.)]",
    ]
    for pat in patterns:
        m = re.search(pat, up)
        if m:
            return m.group(1)
    m = re.search(r"\b([A-E])\b", up)
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


def _code_prefix(prompt: str) -> str:
    """Parte di codice del prompt: dalla prima riga import/from/def in poi.

    Scarta le eventuali righe di istruzioni in linguaggio naturale che precedono
    il codice (es. "Complete the Python function..."), che altrimenti renderebbero
    non valido il programma ricostruito.
    """
    lines = prompt.splitlines()
    for i, l in enumerate(lines):
        if re.match(r"^\s*(from\s|import\s|def\s)", l):
            return "\n".join(lines[i:])
    return prompt


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

    code_prompt = _code_prefix(prompt)          # import + firma + docstring
    imports = "\n".join(l for l in code_prompt.splitlines()
                        if re.match(r"^\s*(from\s|import\s)", l))
    body = _extract_code(response, entry)
    indented = textwrap.indent(textwrap.dedent(body), "    ")
    sep = "" if code_prompt.endswith("\n") else "\n"
    # Il modello puo' rispondere in modi diversi: con la funzione completa, col
    # solo corpo (indentato o meno). Proviamo piu' ricostruzioni e consideriamo
    # superato il test (pass@1) se almeno una compila ed esegue correttamente.
    candidates = []
    if re.search(rf"(?m)^\s*def\s+{re.escape(entry)}\b", body):
        candidates.append((imports + "\n" + body) if imports else body)
    candidates.append(code_prompt + sep + indented)   # firma+docstring + corpo indentato
    candidates.append(code_prompt + sep + body)        # corpo gia' indentato dal modello
    # Se la risposta rieccheggia firma+docstring (magari su una sola riga, perche'
    # il prompt e' stato inviato senza a-capo), il completamento vero e' cio' che
    # segue l'ultima docstring: lo si combina col prompt originale ben formattato.
    if '\"\"\"' in body:
        completion = body.rsplit('\"\"\"', 1)[-1].strip("\n")
        if completion.strip():
            candidates.append(code_prompt + sep +
                              textwrap.indent(textwrap.dedent(completion), "    "))
    ok = any(_run_sandbox(prog + "\n\n" + test + "\n") for prog in candidates)
    return {"score": 1.0 if ok else 0.0,
            "confidence": "high" if ok else "low",
            "needs_review": not ok,
            "parsed": "pass" if ok else "fail"}


def _extract_code(response: str, entry: str) -> str:
    """Estrae il codice da una risposta, gestendo i blocchi markdown ```.

    Se non c'e' un blocco markdown, prova a partire dalla riga della firma
    ``def <entry>`` (scartando l'eventuale testo introdotto dal modello); in
    ultima istanza restituisce la risposta cosi' com'e'.
    """
    # consuma solo la riga della fence (```python), preservando l'indentazione
    # della prima riga di codice.
    m = re.search(r"```(?:python)?[^\n]*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip("\n")
    m = re.search(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip("\n")
    if entry:
        m = re.search(rf"(?ms)^\s*def\s+{re.escape(entry)}\b.*", response)
        if m:
            return m.group(0).strip("\n")
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
    if btype == "code":
        return score_code(response, meta)
    return {"score": 0.0, "confidence": "low", "needs_review": True, "parsed": None}
