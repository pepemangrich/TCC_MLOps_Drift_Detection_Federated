#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, ast, json, os
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser("Plot robusto dos logs")
    p.add_argument("--results_dir", help="Dir com round_metrics.csv e per_client_metrics.csv")
    p.add_argument("--csv", help="Caminho para round_metrics.csv (opcional)")
    p.add_argument("--per-client-csv", help="Caminho para per_client_metrics.csv (opcional)")
    p.add_argument("--outdir", help="Dir de saída (default = próprio results_dir)")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _num(s): return pd.to_numeric(s, errors="coerce")

def _detect_global_acc_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["global_accuracy","accuracy","acc","val_accuracy","val_acc"]:
        if c in df.columns: return c
    for c in df.columns:
        cl=c.lower()
        if "global" in cl and ("acc" in cl or "accuracy" in cl):
            return c
    return None

def _detect_cid_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["client_id","cid","client","partition_id"]:
        if c in df.columns: return c
    for c in df.columns:
        if "client" in c.lower() or "cid" in c.lower(): return c
    return None

def _detect_client_acc_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["accuracy","acc","val_accuracy","val_acc"]:
        if c in df.columns: return c
    # derivar de correct/total
    if all(x in df.columns for x in ("correct","total")):
        df["__derived_acc__"] = _num(df["correct"])/_num(df["total"])
        return "__derived_acc__"
    for c in df.columns:
        cl=c.lower()
        if ("acc" in cl or "accuracy" in cl) and "global" not in cl:
            return c
    return None

def _parse_listish(x) -> List[str]:
    if x is None or (isinstance(x,float) and np.isnan(x)): return []
    if isinstance(x,(list,tuple,set)): return [str(z) for z in x]
    s=str(x).strip()
    if not s: return []
    # JSON / literal
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(s)
            if isinstance(obj,(list,tuple,set)): return [str(z) for z in obj]
        except Exception: pass
    # {1,2}
    if s.startswith("{") and s.endswith("}"):
        inner=s[1:-1]
        return [t.strip() for t in inner.split(",") if t.strip()]
    # CSV simples
    if "," in s: return [t.strip() for t in s.split(",") if t.strip()]
    return [s]

def _compute_drift_count(df: pd.DataFrame) -> pd.Series:
    # 1) já tem drift_count numérico confiável?
    if "drift_count" in df.columns:
        dc=_num(df["drift_count"])
        if dc.notna().sum()>=max(1,int(0.5*len(df))):  # tolerante
            return dc.fillna(0)

    # 2) construir a partir de clients_with_drift
    if "clients_with_drift" in df.columns:
        if "num_clients" in df.columns:
            def row_count(row):
                v = row.get("clients_with_drift")
                if isinstance(v, str) and v.strip().upper()=="GLOBAL":
                    # evento GLOBAL: conta todos os clientes
                    return int(_num(row.get("num_clients",0)) or 0)
                return len(_parse_listish(v))
            return df.apply(row_count, axis=1).astype(int)
        else:
            return df["clients_with_drift"].apply(lambda v: len(_parse_listish(v))).astype(int)

    # 3) como fallback, usar global_drift_flag (0/1)
    for col in ["global_drift_flag","flag_global","drift_flag","flag"]:
        if col in df.columns:
            s=df[col]
            if s.dtype==bool: return s.astype(int)
            return _num(s).fillna(0).astype(int)

    return pd.Series([0]*len(df), index=df.index)

def plot_global_accuracy(df: pd.DataFrame, outdir: Path):
    acc_col=_detect_global_acc_col(df)
    if acc_col is None:
        print("[WARN] Sem coluna de acurácia global.")
        return
    r=_num(df["round"]) if "round" in df.columns else pd.Series(range(1,len(df)+1))
    acc=_num(df[acc_col])
    plt.figure()
    plt.plot(r, acc, marker="o")
    plt.xlabel("Rodada"); plt.ylabel("Acurácia global")
    plt.title("Acurácia global por rodada")
    plt.grid(True, alpha=0.3)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(outdir/"global_accuracy.png", dpi=160); plt.close()

def plot_drift_count(df: pd.DataFrame, outdir: Path):
    r=_num(df["round"]) if "round" in df.columns else pd.Series(range(1,len(df)+1))
    dc=_compute_drift_count(df)
    plt.figure()
    plt.step(r, dc, where="mid")
    plt.xlabel("Rodada"); plt.ylabel("Nº de clientes com drift")
    plt.title("Contagem de flags de drift por rodada")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(outdir/"drift_count.png", dpi=160); plt.close()

def plot_per_client_mean(per_client: pd.DataFrame, outdir: Path):
    cid=_detect_cid_col(per_client); acc=_detect_client_acc_col(per_client)
    if cid is None or acc is None:
        print("[WARN] Sem colunas de cliente/acurácia para média por cliente.")
        return
    g=per_client[[cid,acc]].copy()
    g[acc]=_num(g[acc])
    g=g.groupby(cid, as_index=False)[acc].mean().sort_values(acc)
    plt.figure(figsize=(6.5, max(3.0, 0.28*len(g))))
    plt.barh(g[cid].astype(str), g[acc])
    plt.xlabel("Acurácia média"); plt.ylabel("Cliente")
    plt.title("Acurácia média por cliente")
    plt.tight_layout(); plt.savefig(outdir/"per_client_mean_accuracy.png", dpi=160); plt.close()

def plot_per_client_heatmap(per_client: pd.DataFrame, outdir: Path):
    cid=_detect_cid_col(per_client); acc=_detect_client_acc_col(per_client)
    if cid is None or acc is None or "round" not in per_client.columns:
        print("[WARN] Sem round/cliente/acurácia para heatmap.")
        return
    df=per_client[[cid,"round",acc]].copy()
    df["round"]=_num(df["round"]).astype(int)
    df[acc]=_num(df[acc])
    pivot=df.pivot_table(index=cid, columns="round", values=acc, aggfunc="mean")
    plt.figure(figsize=(7.5, max(3.0, 0.28*pivot.shape[0])))
    im=plt.imshow(pivot.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="Acurácia")
    plt.yticks(range(pivot.shape[0]), [str(x) for x in pivot.index])
    plt.xticks(range(pivot.shape[1]), [str(x) for x in pivot.columns])
    plt.xlabel("Rodada"); plt.ylabel("Cliente")
    plt.title("Heatmap: acurácia por cliente × rodada")
    plt.tight_layout(); plt.savefig(outdir/"per_client_heatmap.png", dpi=160); plt.close()

def main():
    args=parse_args()
    if args.results_dir:
        base=Path(args.results_dir)
        csv_path=base/"round_metrics.csv"
        per_client_path=base/"per_client_metrics.csv"
        outdir=Path(args.outdir) if args.outdir else base
    else:
        csv_path=Path(args.csv); per_client_path=Path(args.per_client_csv) if args.per_client_csv else None
        outdir=Path(args.outdir) if args.outdir else (csv_path.parent if csv_path else Path("results"))

    if not csv_path or not csv_path.exists():
        raise SystemExit(f"[ERRO] round_metrics.csv não encontrado ({csv_path})")

    df=pd.read_csv(csv_path)
    plot_global_accuracy(df, outdir)
    plot_drift_count(df, outdir)

    if per_client_path and per_client_path.exists():
        pc=pd.read_csv(per_client_path)
        plot_per_client_mean(pc, outdir)
        plot_per_client_heatmap(pc, outdir)
    else:
        print(f"[INFO] per_client_metrics.csv não encontrado em {per_client_path}")

    # resumo útil
    cols=[c for c in ["round","global_accuracy","num_clients","clients_with_drift","global_drift_flag"] if c in df.columns]
    if cols:
        print("\nResumo global (head):")
        print(df[cols].head(10).to_string(index=False))

    if args.show: plt.show()

if __name__=="__main__":
    main()