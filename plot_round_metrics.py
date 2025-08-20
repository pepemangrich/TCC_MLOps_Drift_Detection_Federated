#!/usr/bin/env python3
"""
Gera gráficos a partir dos logs do servidor (Flower).

Saídas (por padrão em ./logs):
- global_accuracy.png: acurácia global por rodada (a partir de round_metrics.csv)
- drift_count.png: número de clientes com drift por rodada (a partir de round_metrics.csv)
- per_client_mean_accuracy.png: acurácia média por cliente (a partir de per_client_metrics.csv)
- per_client_heatmap.png: heatmap acurácia por cliente x rodada (a partir de per_client_metrics.csv)

Uso:
  python plot_round_metrics.py
  python plot_round_metrics.py --csv logs/round_metrics.csv --per-client-csv logs/per_client_metrics.csv --outdir logs --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot dos logs do servidor (Flower).")
    p.add_argument("--csv", default="logs/round_metrics.csv", help="Caminho do CSV global (round_metrics.csv).")
    p.add_argument("--per-client-csv", default="logs/per_client_metrics.csv", help="Caminho do CSV por-cliente.")
    p.add_argument("--outdir", default="logs", help="Diretório de saída dos PNGs.")
    p.add_argument("--show", action="store_true", help="Mostra os gráficos na tela (além de salvar).")
    return p.parse_args()


def ensure_exists(path: Path) -> None:
    if not path.exists():
        sys.exit(f"[ERRO] Arquivo não encontrado: {path}")

def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def parse_clients_with_drift(series: pd.Series) -> pd.Series:
    # Converte "a;b;c" -> contagem
    def _count_str(x: str) -> int:
        if pd.isna(x) or x == "":
            return 0
        return len([t for t in str(x).split(";") if t])
    return series.astype(str).map(_count_str)

def plot_global_metrics(df: pd.DataFrame, outdir: Path) -> None:
    # Acurácia global por rodada
    fig1 = plt.figure()
    plt.plot(df["round"], df["global_accuracy"], marker="o")
    plt.xlabel("Rodada")
    plt.ylabel("Acurácia global")
    plt.title("Acurácia global por rodada")
    acc_png = outdir / "global_accuracy.png"
    fig1.tight_layout()
    fig1.savefig(acc_png, dpi=150)
    plt.close(fig1)

    # Contagem de clientes com drift
    if "drift_count" not in df.columns:
        df = df.copy()
        df["drift_count"] = parse_clients_with_drift(df.get("clients_with_drift", pd.Series(dtype=str)))
    fig2 = plt.figure()
    plt.plot(df["round"], df["drift_count"], marker="o")
    plt.xlabel("Rodada")
    plt.ylabel("Clientes com drift")
    plt.title("Clientes com drift por rodada")
    drift_png = outdir / "drift_count.png"
    fig2.tight_layout()
    fig2.savefig(drift_png, dpi=150)
    plt.close(fig2)

    print("[OK] Gráficos globais salvos:")
    print(f" - {acc_png}")
    print(f" - {drift_png}")

def plot_per_client_mean(per_client: pd.DataFrame, outdir: Path) -> None:
    # Acúmulo e média por cliente
    g = per_client.groupby("cid", as_index=False).agg(
        mean_accuracy=("accuracy", "mean"),
        rounds=("round", "nunique"),
        total_examples=("num_examples", "sum"),
    ).sort_values("mean_accuracy", ascending=False)

    # Bar plot (acurácia média por cliente)
    fig = plt.figure(figsize=(10, max(4, 0.35 * len(g))))
    plt.barh(g["cid"], g["mean_accuracy"])
    plt.xlabel("Acurácia média")
    plt.ylabel("Cliente (cid)")
    plt.title("Acurácia média por cliente")
    plt.gca().invert_yaxis()  # cliente com maior média no topo
    out_png = outdir / "per_client_mean_accuracy.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # Salva também um CSV resumo
    out_csv = outdir / "per_client_summary.csv"
    g.to_csv(out_csv, index=False)

    print("[OK] Per-client (média) salvo:")
    print(f" - {out_png}")
    print(f" - {out_csv}")

def plot_per_client_heatmap(per_client: pd.DataFrame, outdir: Path) -> None:
    # Pivot: linhas=cliente, colunas=rodada, valores=acurácia
    pivot = per_client.pivot_table(index="cid", columns="round", values="accuracy", aggfunc="mean")
    pivot = pivot.sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    # Heatmap com matplotlib (sem seaborn, sem cores customizadas)
    fig = plt.figure(figsize=(max(6, 0.4 * pivot.shape[1]), max(4, 0.4 * pivot.shape[0])))
    im = plt.imshow(pivot.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.4)
    plt.yticks(np.arange(pivot.shape[0]), pivot.index)
    plt.xticks(np.arange(pivot.shape[1]), pivot.columns)
    plt.xlabel("Rodada")
    plt.ylabel("Cliente (cid)")
    plt.title("Heatmap de acurácia por cliente x rodada")
    out_png = outdir / "per_client_heatmap.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print("[OK] Per-client (heatmap) salvo:")
    print(f" - {out_png}")

def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    # --------- Globais ---------
    csv_path = Path(args.csv)
    ensure_exists(csv_path)
    df = pd.read_csv(csv_path)

    # cria coluna drift_count se faltando
    if "drift_count" not in df.columns:
        df = df.copy()
        df["drift_count"] = parse_clients_with_drift(df.get("clients_with_drift", pd.Series(dtype=str)))

    # Ordena por rodada (caso necessário)
    if "round" not in df.columns:
        sys.exit("[ERRO] CSV global precisa conter coluna 'round'.")
    df = df.sort_values("round", kind="stable")

    plot_global_metrics(df, outdir)

    # --------- Por-cliente ---------
    per_client_path = Path(args.per_client_csv)
    if per_client_path.exists():
        per_client = pd.read_csv(per_client_path)
        required_cols = {"round", "cid", "accuracy", "num_examples", "drift"}
        if not required_cols.issubset(per_client.columns):
            print(f"[AVISO] CSV por-cliente não contém colunas esperadas {required_cols}. Pulando per-client plots.")
        else:
            per_client = per_client.sort_values(["cid", "round"], kind="stable")
            plot_per_client_mean(per_client, outdir)
            plot_per_client_heatmap(per_client, outdir)
    else:
        print(f"[INFO] Arquivo por-cliente não encontrado: {per_client_path}. Plots per-client não gerados.")

    # Resumo no terminal
    print("\nResumo global (primeiras linhas):")
    cols = [c for c in ["round", "global_accuracy", "drift_count"] if c in df.columns]
    print(df[cols].head(10).to_string(index=False))

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()