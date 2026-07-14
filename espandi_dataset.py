#!/usr/bin/env python3
"""Amplia i dataset dei benchmark a N prompt ciascuno (default 30).

Usa la libreria ufficiale `datasets` di Hugging Face (canale affidabile, non il
datasets-server). MANTIENE i prompt gia' presenti e ne aggiunge di nuovi fino a
N per benchmark, con lo STESSO formato e criterio di scoring. Prima di
sovrascrivere fa un backup dei dataset attuali in `datasets_bak/`.

Prerequisito (una volta sola, sul PC host con internet):
    pip install datasets

Uso:
    python espandi_dataset.py            # porta tutti i dataset a 30
    python espandi_dataset.py --n 50     # a 50
    python espandi_dataset.py --check    # mostra alcuni item generati e non scrive
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil

try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit(
        "Manca la libreria 'datasets'. Installala con:  pip install datasets")

DATA_DIR = "datasets"
BAK_DIR = "datasets_bak"


# --------------------------------------------------------------- formattatori
def fmt_gsm8k(row, idx):
    q = row["question"].strip()
    ans = row["answer"]
    if "####" not in ans:
        return None
    expected = ans.split("####")[-1].strip().replace(",", "")
    prompt = ("Solve the math word problem. End your answer with a line of the "
              "form '#### <number>'.\n" + q + "\nAnswer:")
    return {"id": f"gsm8k-{idx}", "prompt": prompt, "expected": expected}


def fmt_csqa(row, idx):
    q = row["question"].strip()
    labels = row["choices"]["label"]
    texts = row["choices"]["text"]
    key = (row.get("answerKey") or "").strip()
    if key not in labels:
        return None
    choices = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
    prompt = ("Answer with only the single letter of the correct option.\n"
              f"Question: {q}\nChoices:\n{choices}\nAnswer:")
    return {"id": f"csqa-{idx}", "prompt": prompt, "answer_key": key, "expected": key}


def fmt_truthfulqa(row, idx):
    q = row["question"].strip()
    choices = row["mc1_targets"]["choices"]
    labels = row["mc1_targets"]["labels"]
    correct = [c for c, l in zip(choices, labels) if l == 1]
    distract = [c for c, l in zip(choices, labels) if l == 0]
    if len(correct) != 1 or len(distract) < 3:
        return None
    opts = [correct[0]] + distract[:3]
    random.Random(1000 + idx).shuffle(opts)
    letters = ["A", "B", "C", "D"]
    expected = letters[opts.index(correct[0])]
    body = "\n".join(f"{L}) {o}" for L, o in zip(letters, opts))
    prompt = ("Answer with only the single letter of the correct option.\n"
              f"Question: {q}\nChoices:\n{body}\nAnswer:")
    return {"id": f"tqa-{idx}", "prompt": prompt, "answer_key": expected, "expected": expected}


def fmt_humaneval(row, idx):
    prompt = "Complete the Python function. Output only the code.\n" + row["prompt"]
    return {"id": row.get("task_id", f"HumanEval-{idx}").replace("/", "-"),
            "entry_point": row["entry_point"],
            "prompt": prompt,
            "test": row["test"],
            "expected": row.get("canonical_solution", "").strip()}


BBH_TASKS = {
    "boolean_expressions": "Evaluate the result of the following boolean expression. Answer with only True or False.",
    "navigate": "Answer with only Yes or No.",
    "word_sorting": "Sort the following words alphabetically. Output only the sorted words separated by spaces.",
    "date_understanding": "Answer with only the single letter of the correct option.",
}


def fmt_bbh(row, task, idx):
    inp = row["input"].strip()
    target = row["target"].strip().strip("()")
    prompt = f"{BBH_TASKS[task]}\n{inp}\nAnswer:"
    return {"id": f"bbh-{task[:4]}-{idx}", "task": task, "prompt": prompt, "expected": target}


# --------------------------------------------------------------- sorgenti
def rows_of(name):
    if name == "gsm8k":
        return load_dataset("openai/gsm8k", "main", split="test")
    if name == "commonsenseqa":
        return load_dataset("tau/commonsense_qa", split="validation")
    if name == "truthfulqa":
        return load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    if name == "humaneval":
        return load_dataset("openai/openai_humaneval", split="test")
    raise ValueError(name)


def load_existing(name):
    path = os.path.join(DATA_DIR, f"{name}.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def key_of(item):
    return item["prompt"].strip()


def expand_simple(name, fmt, N):
    existing = load_existing(name)
    have = {key_of(i) for i in existing}
    items = list(existing)
    idx = len(existing) + 1
    for row in rows_of(name):
        if len(items) >= N:
            break
        it = fmt(dict(row), idx)
        if it is None or key_of(it) in have:
            continue
        items.append(it); have.add(key_of(it)); idx += 1
    return items[:N]


def expand_bbh(N):
    existing = load_existing("bbh")
    have = {key_of(i) for i in existing}
    items = list(existing)
    tasks = list(BBH_TASKS)
    per_task = max(0, N - len(items)) // len(tasks) + 1
    idx = len(existing) + 1
    for task in tasks:
        added = 0
        for row in load_dataset("lukaemon/bbh", task, split="test"):
            if len(items) >= N or added >= per_task:
                break
            it = fmt_bbh(dict(row), task, idx)
            if key_of(it) in have:
                continue
            items.append(it); have.add(key_of(it)); idx += 1; added += 1
    return items[:N]


def write(name, items, check):
    if check:
        print(f"\n=== {name}: {len(items)} item (ultimi 2) ===")
        for it in items[-2:]:
            print(json.dumps(it, ensure_ascii=False)[:300])
        return
    path = os.path.join(DATA_DIR, f"{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"  {name}: scritti {len(items)} item -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not args.check:
        os.makedirs(BAK_DIR, exist_ok=True)
        for f in os.listdir(DATA_DIR):
            if f.endswith(".jsonl"):
                shutil.copy(os.path.join(DATA_DIR, f), os.path.join(BAK_DIR, f))
        print(f"Backup dei dataset attuali in {BAK_DIR}/")

    N = args.n
    plan = [
        ("gsm8k",         lambda: expand_simple("gsm8k", fmt_gsm8k, N)),
        ("commonsenseqa", lambda: expand_simple("commonsenseqa", fmt_csqa, N)),
        ("truthfulqa",    lambda: expand_simple("truthfulqa", fmt_truthfulqa, N)),
        ("humaneval",     lambda: expand_simple("humaneval", fmt_humaneval, N)),
        ("bbh",           lambda: expand_bbh(N)),
    ]
    for name, fn in plan:
        try:
            write(name, fn(), args.check)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {name}: ERRORE ({exc}). Dataset lasciato invariato.")
    print("\nFatto. Controlla qualche item, poi lancia i due config _ext sul Pi.")


if __name__ == "__main__":
    main()
