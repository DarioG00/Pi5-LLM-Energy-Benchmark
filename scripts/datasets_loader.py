"""Caricamento dei dataset/benchmark in formato JSONL."""
from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass, field
from typing import List

log = logging.getLogger("datasets")


@dataclass
class Sample:
    benchmark: str
    btype: str
    id: str
    prompt: str
    expected: str = ""
    meta: dict = field(default_factory=dict)


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{ln} JSON non valido: {exc}") from exc
    return rows


def load_benchmark(name: str, btype: str, path: str, base_dir: str = ".") -> List[Sample]:
    full = path if os.path.isabs(path) else os.path.join(base_dir, path)
    if not os.path.exists(full):
        raise FileNotFoundError(f"Dataset mancante: {full}")
    samples: List[Sample] = []
    for row in load_jsonl(full):
        meta = {k: v for k, v in row.items()
                if k not in ("id", "prompt", "expected")}
        samples.append(Sample(
            benchmark=name,
            btype=btype,
            id=str(row.get("id", "")),
            prompt=row["prompt"],
            expected=str(row.get("expected", "")),
            meta=meta,
        ))
    log.info("Benchmark '%s': %d campioni caricati da %s", name, len(samples), full)
    return samples


def load_all(benchmarks_cfg: List[dict], base_dir: str = ".") -> List[Sample]:
    out: List[Sample] = []
    for b in benchmarks_cfg:
        out.extend(load_benchmark(b["name"], b["type"], b["file"], base_dir))
    return out
