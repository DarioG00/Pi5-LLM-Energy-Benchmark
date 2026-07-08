"""Parsing dell'output di llama-cli: statistiche di timing e testo generato."""
from __future__ import annotations

import re
from typing import Optional

# Formato compatto effettivamente prodotto dal build di llama.cpp in uso, es.:
#   "[ Prompt: 16,4 t/s | Generation: 15,8 t/s ]"  (build recente, b8989)
# (gestisce anche la virgola come separatore decimale).
_COMPACT_RE = re.compile(
    r"Prompt:\s*([\d.,]+)\s*t/s\s*\|\s*Generation:\s*([\d.,]+)\s*t/s",
    re.IGNORECASE,
)

# Fallback per il formato classico ("llama_print_timings:" /
# "llama_perf_context_print:"), che fornisce anche il conteggio dei token.
_PROMPT_RE = re.compile(
    r"(?:prompt eval time)\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens?.*?([\d.]+)\s*tokens per second",
    re.IGNORECASE | re.DOTALL,
)
_EVAL_RE = re.compile(
    r"(?<!prompt )(?:eval time)\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs?.*?([\d.]+)\s*tokens per second",
    re.IGNORECASE | re.DOTALL,
)
_LOAD_RE = re.compile(r"load time\s*=\s*([\d.]+)\s*ms", re.IGNORECASE)


def _to_float(s: Optional[str]) -> Optional[float]:
    """Converte una stringa numerica in float gestendo la virgola decimale."""
    if s is None:
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def parse_timings(text: str) -> dict:
    """Estrae prompt/gen tokens-per-second, token e tempi dall'output.

    Prova prima il formato compatto del build in uso ("Prompt: x t/s |
    Generation: y t/s") e, in subordine, il formato classico di llama.cpp
    (che fornisce anche il numero di token). I campi mancanti restano None.
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

    # 1) formato compatto (prioritario: è quello realmente emesso)
    m = _COMPACT_RE.search(text)
    if m:
        res["prompt_tps"] = _to_float(m.group(1))
        res["gen_tps"] = _to_float(m.group(2))

    # 2) formato classico: completa i t/s mancanti e i conteggi di token
    m = _PROMPT_RE.search(text)
    if m:
        res["prompt_eval_ms"] = _to_float(m.group(1))
        res["prompt_tokens"] = int(m.group(2))
        if res["prompt_tps"] is None:
            res["prompt_tps"] = _to_float(m.group(3))
    m = _EVAL_RE.search(text)
    if m:
        res["gen_eval_ms"] = _to_float(m.group(1))
        res["gen_tokens"] = int(m.group(2))
        if res["gen_tps"] is None:
            res["gen_tps"] = _to_float(m.group(3))
    m = _LOAD_RE.search(text)
    if m:
        res["load_ms"] = _to_float(m.group(1))
    return res


def extract_response(raw: str, prompt: str) -> str:
    """Ricava il testo generato dal blocco catturato per un prompt.

    Rimuove l'eco del prompt, le righe di log di llama.cpp e il prompt `>`.
    """
    text = raw
    # 1) rimuove il blocco del prompt rieccheggiato (multi-riga o collassato)
    for p in (prompt.strip(), prompt.replace("\n", " ").strip()):
        if p:
            idx = text.find(p)
            if idx != -1:
                text = text[idx + len(p):]
                break
    # 2) insieme delle righe del prompt: se rieccheggiate singolarmente vanno
    #    scartate (lo scoring del codice ricostruisce comunque firma+docstring dal
    #    prompt originale; il completamento non coincide con righe del prompt).
    prompt_lines = {l.strip() for l in prompt.splitlines() if l.strip()}
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s == ">" or (s.endswith(">") and len(s) <= 2):
            continue
        if s.startswith("/read"):                # eco del comando /read
            continue
        if s in prompt_lines:                    # riga del prompt rieccheggiata
            continue
        # scarta le righe di log/diagnostica di llama.cpp
        if re.match(r"^(llama_|llm_|ggml_|main:|system_info|sampler|generate:|build:|srv |print_info|load[_:]|register_|common_)", s):
            continue
        # scarta la riga dei timing [ Prompt: ... | Generation: ... t/s ]
        if _COMPACT_RE.search(s):
            continue
        lines.append(line.rstrip())
    out = "\n".join(lines).strip()
    out = re.sub(r"\n?>\s*$", "", out).strip()
    return out


def estimate_tokens(text: str) -> int:
    """Stima grezza dei token generati (fallback se mancano le statistiche).

    Regola pratica ~ 0.75 parole/token -> token ≈ parole / 0.75.
    """
    words = len(text.split())
    return int(round(words / 0.75)) if words else 0
