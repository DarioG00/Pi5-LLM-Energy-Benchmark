"""Analisi dei risultati e generazione dei grafici di confronto.

Usa pandas (aggregazione), seaborn/matplotlib (grafici) e scikit-learn
(normalizzazione per uno score composito di efficienza/qualità/latenza).
Legge il CSV prodotto dal runner e salva i grafici in recordings/plots.
"""
from __future__ import annotations

import os
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

log = logging.getLogger("analysis")
sns.set_theme(style="whitegrid")

LOAD_TAG = "__load__"


def _model_label(row) -> str:
    """Etichetta con famiglia e quantizzazione del modello, usata come categoria nei grafici."""
    return f"{row['family']}\n{row['quant']}"


def load_results(csv_path: str) -> pd.DataFrame:
    """Legge il CSV dei risultati e ne normalizza i tipi; azzera (NaN) le metriche energetiche delle righe con energia non valida (<=0), cosi' da escluderle dai grafici."""
    df = pd.read_csv(csv_path)
    for col in ["latency_s", "energy_net_j", "avg_power_w",
                "score", "prompt_tps", "gen_tps", "temp_c"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "throttle_event" in df:
        df["throttle_event"] = df["throttle_event"].astype(str).str.lower().isin(["true", "1"])
    if "throttle_active" in df:
        df["throttle_active"] = df["throttle_active"].astype(str).str.lower().isin(["true", "1"])
    # Righe con energia totale <= 0 = lettura PMIC mancante (es. connessione
    # caduta): le si esclude dalle metriche energetiche impostandole a NaN, cosi'
    # non falsano medie e grafici (niente barre negative spurie).
    if "energy_total_j" in df:
        bad = pd.to_numeric(df["energy_total_j"], errors="coerce").fillna(0) <= 0
        n_bad = int((bad & (df["benchmark"] != LOAD_TAG)).sum())
        if n_bad:
            log.warning("Escludo %d inferenze senza misura di energia valida "
                        "(energia<=0: probabile lettura PMIC mancante).", n_bad)
        df.loc[bad, ["energy_total_j", "energy_net_j", "avg_power_w"]] = np.nan
    df["model"] = df.apply(lambda r: f"{r['family']}-{r['quant']}", axis=1)
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Media sulle ripetizioni per (modello, thread, benchmark)."""
    infer = df[df["benchmark"] != LOAD_TAG].copy()
    grp = (infer.groupby(["model", "family", "quant", "threads", "benchmark"])
           .agg(latency_s=("latency_s", "mean"),
                energy_net_j=("energy_net_j", "mean"),
                avg_power_w=("avg_power_w", "mean"),
                gen_tps=("gen_tps", "mean"),
                score=("score", "mean"))
           .reset_index())
    return grp


def per_config(df: pd.DataFrame) -> pd.DataFrame:
    """Media su tutti i benchmark per (modello, thread): vista di sintesi."""
    infer = df[df["benchmark"] != LOAD_TAG].copy()
    grp = (infer.groupby(["model", "family", "quant", "threads"])
           .agg(latency_s=("latency_s", "mean"),
                energy_net_j=("energy_net_j", "mean"),
                avg_power_w=("avg_power_w", "mean"),
                prompt_tps=("prompt_tps", "mean"),
                gen_tps=("gen_tps", "mean"),
                score=("score", "mean"))
           .reset_index())
    return grp


def total_energy_per_config(df: pd.DataFrame) -> pd.DataFrame:
    """Energia totale consumata da ciascuna configurazione (modello x thread)
    per completare l'intera suite di benchmark.

    Per ogni ripetizione somma l'energia netta di tutte le inferenze, poi media
    sulle ripetizioni: si ottiene l'energia media per una passata completa dei
    benchmark. E' una misura di consumo *complessivo*, complementare all'energia
    media per singola inferenza.
    """
    infer = df[df["benchmark"] != LOAD_TAG].copy()
    per_rep = (infer.groupby(["model", "family", "quant", "threads", "repetition"])
               .agg(energy_total_j=("energy_net_j", lambda s: s.sum(min_count=1)),
                    n_inferenze=("energy_net_j", "count"))
               .reset_index())
    grp = (per_rep.groupby(["model", "family", "quant", "threads"])
           .agg(energy_total_j=("energy_total_j", "mean"),
                n_inferenze=("n_inferenze", "mean"))
           .reset_index())
    return grp.sort_values(["model", "threads"])


def composite_score(cfg_df: pd.DataFrame) -> pd.DataFrame:
    """Score composito normalizzato (sklearn): alta qualità, bassa energia/latenza."""
    d = cfg_df.copy()
    feats = d[["energy_net_j", "latency_s", "score"]].fillna(d[["energy_net_j", "latency_s", "score"]].mean())
    norm = MinMaxScaler().fit_transform(feats)
    d["n_energy"], d["n_lat"], d["n_score"] = norm[:, 0], norm[:, 1], norm[:, 2]
    # peso: 40% efficienza (energia), 20% latenza, 40% qualità (energia/latenza invertite)
    d["composite"] = (0.4 * (1 - d["n_energy"]) + 0.2 * (1 - d["n_lat"]) + 0.4 * d["n_score"])
    return d.sort_values("composite", ascending=False)


# --------------------------------------------------------------------- grafici
def _save(fig, path):
    """Salva la figura su file (con layout compatto) e chiude il plot."""
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log.info("Grafico salvato: %s", path)


def plot_efficiency(cfg_df, out):
    """Grafico a barre dell'energia netta media per inferenza, per modello e numero di thread."""
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=cfg_df, x="model", y="energy_net_j", hue="threads", ax=ax)
    ax.set_title("Energia netta media per inferenza (J) per modello e thread")
    ax.set_xlabel(""); ax.set_ylabel("Energia netta (J) (minore = meglio)")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, out)


def plot_energy_per_config(tot_df, out):
    """Grafico dell'energia totale consumata da ciascuna configurazione."""
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=tot_df, x="model", y="energy_total_j", hue="threads", ax=ax)
    ax.set_title("Energia totale consumata per completare i benchmark (J) "
                 "per configurazione")
    ax.set_xlabel(""); ax.set_ylabel("Energia totale (J) (minore = meglio)")
    ax.tick_params(axis="x", rotation=20)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.0f", fontsize=8, padding=2)
    _save(fig, out)


def plot_latency(cfg_df, out):
    """Grafico a barre della latenza media di inferenza, per modello e numero di thread."""
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=cfg_df, x="model", y="latency_s", hue="threads", ax=ax)
    ax.set_title("Latenza media di inferenza (s) per modello e thread")
    ax.set_xlabel(""); ax.set_ylabel("Latenza (s)")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, out)


def plot_quality(agg_df, out):
    """Mappa di calore dell'accuratezza media (0-1) per modello e benchmark."""
    fig, ax = plt.subplots(figsize=(12, 6))
    piv = agg_df.pivot_table(index="model", columns="benchmark",
                             values="score", aggfunc="mean")
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
    ax.set_title("Qualità media (score 0-1) per modello e benchmark")
    ax.set_xlabel(""); ax.set_ylabel("")
    _save(fig, out)


def plot_power(cfg_df, out):
    """Grafico a barre della potenza media assorbita, per modello e numero di thread."""
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=cfg_df, x="model", y="avg_power_w", hue="threads", ax=ax)
    ax.set_title("Potenza media assorbita (W) per modello e thread")
    ax.set_xlabel(""); ax.set_ylabel("Potenza media (W)")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, out)


def plot_throughput(cfg_df, out):
    """Grafici a barre della velocita' di prompt processing e di generazione (token/s)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(data=cfg_df, x="model", y="prompt_tps", hue="threads", ax=axes[0])
    axes[0].set_title("Prompt processing (t/s) per modello e thread")
    axes[0].set_xlabel(""); axes[0].set_ylabel("token/s")
    axes[0].tick_params(axis="x", rotation=20)
    sns.barplot(data=cfg_df, x="model", y="gen_tps", hue="threads", ax=axes[1])
    axes[1].set_title("Generation (t/s) per modello e thread")
    axes[1].set_xlabel(""); axes[1].set_ylabel("token/s")
    axes[1].tick_params(axis="x", rotation=20)
    _save(fig, out)


def plot_tradeoff(cfg_df, out):
    """Scatter del compromesso tra energia per inferenza e qualita', per configurazione."""
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=cfg_df, x="energy_net_j", y="score",
                    hue="model", style="threads", s=140, ax=ax)
    ax.set_title("Trade-off efficienza vs qualità")
    ax.set_xlabel("Energia netta per inferenza (J) (minore = meglio)")
    ax.set_ylabel("Score qualità (maggiore = meglio)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _save(fig, out)


def plot_composite(comp_df, out):
    """Grafico della classifica delle configurazioni secondo lo score composito."""
    fig, ax = plt.subplots(figsize=(11, 6))
    comp_df = comp_df.copy()
    comp_df["cfg"] = comp_df["model"] + " t" + comp_df["threads"].astype(str)
    sns.barplot(data=comp_df, x="composite", y="cfg",
                hue="cfg", palette="viridis", legend=False, ax=ax)
    ax.set_title("Classifica configurazioni (score composito: efficienza+qualità+latenza)")
    ax.set_xlabel("Score composito (maggiore = meglio)"); ax.set_ylabel("")
    _save(fig, out)


def plot_thermal(df, out):
    """Boxplot della temperatura del SoC durante le inferenze, per modello e thread."""
    infer = df[(df["benchmark"] != LOAD_TAG) & df["temp_c"].notna()].copy()
    if infer.empty:
        log.info("Nessun dato di temperatura: salto il grafico termico.")
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=infer, x="model", y="temp_c", hue="threads", ax=ax)
    ax.axhline(80, ls="--", color="red", lw=1, label="soft limit ~80C")
    ax.set_title("Temperatura SoC durante le inferenze per modello e thread")
    ax.set_xlabel(""); ax.set_ylabel("Temperatura (C)")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, out)


def thermal_report(df) -> str:
    """Restituisce una riga di riepilogo su temperatura massima ed eventi di throttling."""
    n_throttle = int(df.get("throttle_event", pd.Series(dtype=bool)).sum())
    n_active = int(df.get("throttle_active", pd.Series(dtype=bool)).sum())
    tmax = df["temp_c"].max() if "temp_c" in df and df["temp_c"].notna().any() else float("nan")
    msg = (f"Termico - T max: {tmax:.1f}C | righe con evento throttling: "
           f"{n_throttle} | letture con throttling attivo: {n_active}")
    if n_throttle:
        msg += "  [!] alcune misure sono state acquisite con throttling: verificale."
    return msg


def run_analysis(csv_path: str, cfg: dict, base_dir: str = ".") -> None:
    """Esegue l'intera analisi: aggrega i risultati del CSV, salva le tabelle e genera tutti i grafici di confronto in recordings/plots."""
    plots_dir = os.path.join(base_dir, cfg["output"]["plots_dir"])
    os.makedirs(plots_dir, exist_ok=True)

    df = load_results(csv_path)
    if df.empty:
        log.warning("CSV vuoto: nessuna analisi.")
        return

    agg = aggregate(df)
    cfgv = per_config(df)
    tot_e = total_energy_per_config(df)
    comp = composite_score(cfgv)

    # salva tabelle aggregate
    agg.to_csv(os.path.join(base_dir, cfg["output"]["raw_dir"], "aggregated_by_benchmark.csv"), index=False)
    tot_e.to_csv(os.path.join(base_dir, cfg["output"]["raw_dir"], "energy_per_config.csv"), index=False)
    comp.to_csv(os.path.join(base_dir, cfg["output"]["raw_dir"], "ranking_composite.csv"), index=False)

    plot_efficiency(cfgv, os.path.join(plots_dir, "efficiency_jpt.png"))
    plot_energy_per_config(tot_e, os.path.join(plots_dir, "energy_per_config.png"))
    plot_latency(cfgv, os.path.join(plots_dir, "latency.png"))
    plot_quality(agg, os.path.join(plots_dir, "quality_heatmap.png"))
    plot_power(cfgv, os.path.join(plots_dir, "power.png"))
    plot_throughput(cfgv, os.path.join(plots_dir, "throughput.png"))
    plot_tradeoff(cfgv, os.path.join(plots_dir, "tradeoff.png"))
    plot_composite(comp, os.path.join(plots_dir, "ranking_composite.png"))
    plot_thermal(df, os.path.join(plots_dir, "thermal.png"))
    log.info(thermal_report(df))
    log.info("Analisi completata. Grafici in %s", plots_dir)


if __name__ == "__main__":
    import sys
    import json
    logging.basicConfig(level=logging.INFO)
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "recordings/results.csv"
    with open("config.json", encoding="utf-8") as fh:
        cfg = json.load(fh)
    run_analysis(csv_path, cfg, ".")
