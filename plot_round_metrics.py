#!/usr/bin/env python3
"""
Gera gráficos a partir de logs/round_metrics.csv:

- logs/global_accuracy.png: acurácia global por rodada
- logs/drift_count.png: nº de clientes com drift por rodada

Uso:
  python plot_round_metrics.py
  python plot_round_metrics.py --csv logs/round_metrics.csv --outdir logs --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot dos logs de rodada do servidor (Flower).")
    p.add_argument("--csv", default="logs/round_metrics.csv", help="Caminho do CSV de logs.")
    p.add_argument("--outdir", default="logs", help="Diretório de saída dos PNGs.")
    p.add_argument("--show", action="store_true", help="Mostra os gráficos na tela (além de salvar).")
    return p.parse_args()


def ensure_exists(path: Path) -> None:
    if not path.exists():
        sys.exit(f"[ERRO] Arquivo não encontrado: {path}")


def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Converte round para int caso venha como string
    if "round" in df.columns:
        df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    else:
        sys.exit("[ERRO] Coluna 'round' ausente no CSV.")

    if "global_accuracy" not in df.columns:
        sys.exit("[ERRO] Coluna 'global_accuracy' ausente no CSV.")

    # Conta clientes com drift (lista separada por ';')
    def count_drift(s) -> int:
        s = "" if pd.isna(s) else str(s).strip()
        if not s:
            return 0
        return len([x for x in s.split(";") if x])

    df["drift_count"] = df.get("clients_with_drift", "").apply(count_drift)
    return df.sort_values("round").reset_index(drop=True)


def plot_global_accuracy(df: pd.DataFrame, outdir: Path) -> Path:
    out = outdir / "global_accuracy.png"
    plt.figure()
    plt.plot(df["round"], df["global_accuracy"], marker="o")
    plt.title("Acurácia global por rodada")
    plt.xlabel("Rodada")
    plt.ylabel("Acurácia")
    plt.grid(True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_drift_count(df: pd.DataFrame, outdir: Path) -> Path:
    out = outdir / "drift_count.png"
    plt.figure()
    plt.bar(df["round"].astype(str), df["drift_count"])
    plt.title("Clientes com drift por rodada")
    plt.xlabel("Rodada")
    plt.ylabel("Nº de clientes com drift")
    plt.grid(axis="y")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    ensure_exists(csv_path)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_df(csv_path)

    acc_png = plot_global_accuracy(df, outdir)
    drift_png = plot_drift_count(df, outdir)

    print("✅ Gráficos gerados:")
    print(f" - {acc_png}")
    print(f" - {drift_png}")

    # Resumo no terminal
    print("\nResumo (primeiras linhas):")
    cols = ["round", "global_accuracy", "drift_count"]
    print(df[cols].head(10).to_string(index=False))

    if args.show:
        # Se o usuário quiser visualizar interativamente
        plt.show()


if __name__ == "__main__":
    main()