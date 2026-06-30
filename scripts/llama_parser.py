"""Parsing dell'output di llama-cli: statistiche di timing e testo generato."""
from __future__ import annotations

import re
from typing import Optional

# Funziona sia con i vecchi "llama_print_timings:" sia con i nuovi
# "llama_perf_context_print:".
_PROMPT_RE = re.compile(
    r"(?:prompt eval time)\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens?.*?([\d.]+)\s*tokens per second",
    re.IGNORECASE | re.DOTALL,
)
_EVAL_RE = re.compile(
    r"(?<!prompt )(?:eval time)\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs?.*?([\d.]+)\s*tokens per second",
    re.IGNORECASE | re.DOTALL,
)
_LOAD_RE = re.compile(r"load time\s*=\s*([\d.]+)\s*ms", re.IGNORECASE)


def parse_timings(text: str) -> dict:
    """Estrae prompt/gen tokens-per-second, token e tempi dall'output.

    I campi mancanti vengono restituiti come None.
    """
    res = {
        "prompt_tps": None,
        "gen_tps": None,
        "prompt_tokens": None,
        "gen_tokens": None,
        "prompt_eval_ms": None,
        "gen_eval_ms": None,
        "load_ms": None,
    }
    m = _PROMPT_RE.search(text)
    if m:
        res["prompt_eval_ms"] = float(m.group(1))
        res["prompt_tokens"] = int(m.group(2))
        res["prompt_tps"] = float(m.group(3))
    m = _EVAL_RE.search(text)
    if m:
        res["gen_eval_ms"] = float(m.group(1))
        res["gen_tokens"] = int(m.group(2))
        res["gen_tps"] = float(m.group(3))
    m = _LOAD_RE.search(text)
    if m:
        res["load_ms"] = float(m.group(1))
    return res


def extract_response(raw: str, prompt: str) -> str:
    """Ricava il testo generato dal blocco catturato per un prompt.

    Rimuove l'eco del prompt, le righe di log di llama.cpp e il prompt `>`.
    """
    text = raw
    single = prompt.replace("\n", " ").strip()
    idx = text.find(single)
    if idx != -1:
        text = text[idx + len(single):]
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s == ">" or s.endswith("> ") and len(s) <= 2:
            continue
        # scarta le righe di log/diagnostica di llama.cpp
        if re.match(r"^(llama_|llm_|ggml_|main:|system_info|sampler|generate:|build:)", s):
            continue
        lines.append(line.rstrip())
    out = "\n".join(lines).strip()
    # rimuove un eventuale '>' finale residuo
    out = re.sub(r"\n?>\s*$", "", out).strip()
    return out


def estimate_tokens(text: str) -> int:
    """Stima grezza dei token generati (fallback se mancano le statistiche).

    Regola pratica ~ 0.75 parole/token -> token ≈ parole / 0.75.
    """
    words = len(text.split())
    return int(round(words / 0.75)) if words else 0
