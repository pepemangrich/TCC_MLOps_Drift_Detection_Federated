# plot_round_metrics.py
import argparse, json, ast, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_listish(x):
    if pd.isna(x): return []
    s=str(x).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            o=parser(s)
            if isinstance(o,(list,tuple,set)): return list(o)
        except: pass
    if ";" in s: return [t for t in s.split(";") if t]
    if "," in s: return [t for t in s.split(",") if t]
    return [s] if s else []

def ensure_outdir(d):
    os.makedirs(d, exist_ok=True)

def plot_global_accuracy(df, out):
    col = next((c for c in ["global_accuracy","accuracy","acc"] if c in df.columns), None)
    if col is None: return
    plt.figure(figsize=(10,7))
    plt.plot(df["round"], df[col], marker="o")
    plt.title("Acurácia global por rodada")
    plt.xlabel("Rodada"); plt.ylabel("Acurácia global")
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "global_accuracy.png"), dpi=150)
    plt.close()

def plot_drift_count(df, out):
    if "clients_with_drift" not in df.columns: return
    counts=[]
    for _,r in df.iterrows():
        items = parse_listish(r["clients_with_drift"])
        if "__GLOBAL__" in items:
            # caso exótico, manter compatibilidade
            n = int(r.get("num_clients",0) or 0)
        else:
            n = len(items)
        counts.append(n)
    x = df["round"].to_numpy()
    y = np.array(counts)
    plt.figure(figsize=(10,7))
    plt.step(x, y, where="post")
    plt.title("Contagem de flags de drift por rodada")
    plt.xlabel("Rodada"); plt.ylabel("Nº de clientes com drift")
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "drift_count.png"), dpi=150)
    plt.close()

def plot_global_flag(df, out):
    if "global_drift_flag" not in df.columns: return
    x = df["round"].to_numpy()
    y = df["global_drift_flag"].astype(int).to_numpy()
    plt.figure(figsize=(10,3.5))
    plt.step(x, y, where="post")
    plt.yticks([0,1], ["0","1"])
    plt.title("Flag de drift GLOBAL por rodada")
    plt.xlabel("Rodada"); plt.ylabel("Global flag")
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "global_drift_flag.png"), dpi=150)
    plt.close()

def plot_delta_band(df, out):
    need = {"delta_t","band_lo","band_hi"}
    if not need.issubset(df.columns): return
    x = df["round"].to_numpy()
    dt = pd.to_numeric(df["delta_t"], errors="coerce").to_numpy()
    lo = pd.to_numeric(df["band_lo"], errors="coerce").to_numpy()
    hi = pd.to_numeric(df["band_hi"], errors="coerce").to_numpy()
    plt.figure(figsize=(10,7))
    plt.plot(x, dt, label="Δₜ")
    plt.fill_between(x, lo, hi, alpha=.2, label="Banda")
    plt.title("Sinal Δₜ e banda de tolerância (Wilbik)")
    plt.xlabel("Rodada"); plt.ylabel("Δ")
    plt.legend()
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "delta_band.png"), dpi=150)
    plt.close()

def plot_per_client_heatmap(per_client, out):
    if per_client is None: return
    req = {"round","cid","accuracy"}
    if not req.issubset(per_client.columns): return
    pvt = per_client.pivot_table(index="cid", columns="round", values="accuracy", aggfunc="mean")
    pvt = pvt.sort_index()
    plt.figure(figsize=(12,4))
    im = plt.imshow(pvt.values, aspect="auto")
    plt.colorbar(im, label="Acurácia")
    plt.title("Heatmap: acurácia por cliente × rodada")
    plt.xlabel("Rodada"); plt.ylabel("Cliente")
    plt.xticks(range(len(pvt.columns)), pvt.columns)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "per_client_heatmap.png"), dpi=150)
    plt.close()

def plot_per_client_mean(per_client, out):
    if per_client is None: return
    req = {"cid","accuracy"}
    if not req.issubset(per_client.columns): return
    means = per_client.groupby("cid")["accuracy"].mean().sort_values(ascending=False)
    plt.figure(figsize=(10,4))
    plt.barh(list(map(str, means.index)), means.values)
    plt.gca().invert_yaxis()
    plt.xlabel("Acurácia média"); plt.ylabel("Cliente")
    plt.title("Acurácia média por cliente")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "per_client_mean_accuracy.png"), dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="results/<exec>/round_metrics.csv")
    ap.add_argument("--per-client-csv", default=None)
    ap.add_argument("--results_dir", default=None, help="pasta com os CSVs (alternativa aos argumentos acima)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    if args.results_dir:
        csv = os.path.join(args.results_dir, "round_metrics.csv")
        pcsv = os.path.join(args.results_dir, "per_client_metrics.csv")
        out = args.results_dir if args.outdir is None else args.outdir
    else:
        csv = args.csv
        pcsv = args.per_client_csv
        out = args.outdir or (os.path.dirname(csv) if csv else ".")

    ensure_outdir(out)
    df = pd.read_csv(csv)
    per_client = pd.read_csv(pcsv) if (pcsv and os.path.exists(pcsv)) else None

    plot_global_accuracy(df, out)
    plot_drift_count(df, out)
    plot_global_flag(df, out)
    plot_delta_band(df, out)
    plot_per_client_heatmap(per_client, out)
    plot_per_client_mean(per_client, out)

if __name__ == "__main__":
    main()