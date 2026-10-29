import csv
import glob
import os
import re

import numpy as np
import pandas as pd
import torch

from src import data
from src.inference import load_model
from src.stats import P_TAR_LIST, PREDS_DIR, RESULTS_DIR, cascade, load, temperatures

SAVE_DIR = os.environ.get("SAVE_DIR", "saved_model_v5")
SEEDS = (1, 2, 3, 4, 5)
MAIN = "last"
SELECTIONS = ("last", "acc", "loss")
MODES = ("none", "overall", "branches")
BUDGETS = (10, 12, 14, 16, 18, 20)
N_BINS = 15
CONFIGS = ([(f"b{n} {w}", "", n, w) for n in (2, 3, 4, 5) for w in ("equal", "linspace")]
           + [("b3 inv_linspace", "", 3, "inv_linspace"), ("v3b", "s7-14", 3, "equal")])
GRID = [c for c in CONFIGS if c[1] == "" and c[3] in ("equal", "linspace")]
BASELINES = (("b0", ""), ("b0 sem pré-treino", "scratch"))
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042, 40: 2.021}


def variant(arch, seed, sel=MAIN):
    v = f"{arch}-r{seed}" if arch else f"r{seed}"
    return v if sel == "loss" else f"{v}-{sel}"


def tag(n, w, arch, seed, sel=MAIN):
    return f"mobilenet_{variant(arch, seed, sel)}_b{n}_{w}_best"


def summarize(values):
    v = np.array([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, 0
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else np.nan), len(v)


def t_quantile(df):
    keys = sorted(T_975)
    return T_975[max(k for k in keys if k <= df)] if df >= 1 else np.nan


def compare(label, metric, deltas):
    mean, std, k = summarize(deltas)
    half = t_quantile(k - 1) * std / np.sqrt(k) if k > 1 else np.nan
    return {"comparacao": label, "metrica": metric, "delta_mean": mean, "delta_std": std,
            "ic95_inf": mean - half, "ic95_sup": mean + half, "n": k,
            "positivos": int(sum(d > 0 for d in deltas)),
            "ic_exclui_zero": int(k > 1 and (mean - half > 0 or mean + half < 0)),
            "deltas": " ".join(f"{d:+.2f}" for d in deltas)}


def exit_accs(t):
    z = np.load(os.path.join(PREDS_DIR, f"{t}_test.npz"))
    return (z["logits"].argmax(2) == z["labels"][:, None]).mean(0) * 100


def exits_table():
    rows = []
    for label, arch, n, w in CONFIGS:
        for sel in SELECTIONS:
            arr = np.array([exit_accs(tag(n, w, arch, s, sel)) for s in SEEDS])
            row = {"config": label, "selecao": sel, "sementes": len(arr)}
            for e in range(arr.shape[1]):
                name = "final" if e == arr.shape[1] - 1 else f"saida{e + 1}"
                row[f"{name}_mean"] = arr[:, e].mean()
                row[f"{name}_std"] = arr[:, e].std(ddof=1)
            row["media_saidas_mean"] = arr.mean(1).mean()
            rows.append(row)
    return rows


@torch.no_grad()
def baselines():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, rows = {}, []
    for label, arch in BASELINES:
        for sel in SELECTIONS:
            accs = []
            for s in SEEDS:
                path = os.path.join(SAVE_DIR, f"{tag(0, 'equal', arch, s, sel)}.pth")
                model, ckpt = load_model(path, device)
                mean, std = data.norm_from_ckpt(ckpt)
                loader = torch.utils.data.DataLoader(
                    data.test_dataset("./data", mean, std), batch_size=1000, num_workers=0)
                correct = sum((model(x.to(device))[-1].argmax(1).cpu() == y).sum().item()
                              for x, y in loader)
                accs.append(correct / 100)
                rows.append({"config": label, "selecao": sel, "semente": s,
                             "acc": correct / 100, "epoca": ckpt["epoch"]})
            results[(label, sel)] = accs
    return results, rows


def operating_points(op, b0_macs):
    idx = op.set_index(["tag", "mode"])
    rows, per_seed = [], {}
    for label, arch, n, w in CONFIGS:
        for mode in MODES:
            recs = []
            for s in SEEDS:
                t = tag(n, w, arch, s)
                r = idx.loc[(t, mode)]
                _, _, e = cascade(load(t, "test"), mode, r["p_tar"])
                recs.append({**r.to_dict(), "exits": [100 * np.mean(e == i) for i in range(n + 1)]})
            per_seed[(label, mode)] = recs
            row = {"config": label, "mode": mode,
                   "p_tar": " / ".join(f"{r['p_tar']:.2f}" for r in recs)}
            for m in ("acc", "macs", "frac_edge", "acc_edge", "acc_cloud"):
                mean, std, _ = summarize([r[m] for r in recs])
                row[f"{m}_mean"], row[f"{m}_std"] = mean, std
            row["macs_rel_b0"] = 100 * row["macs_mean"] / b0_macs
            ex = np.array([r["exits"] for r in recs]).mean(0)
            row["saidas"] = " / ".join(f"{x:.1f}" for x in ex)
            rows.append(row)
    return rows, per_seed


def fixed_exit(per_seed, mode="overall"):
    rows = []
    for label, arch, n, w in CONFIGS:
        picks, fixed_acc, fixed_macs, deltas = [], [], [], []
        recs = per_seed[(label, mode)]
        for s, r in zip(SEEDS, recs):
            t = tag(n, w, arch, s)
            cum = load(t, "test")["cum_macs"] / 1e6
            k = int(np.argmax(cum >= r["macs"] - 1e-9))
            acc = exit_accs(t)[k]
            picks.append("final" if k == n else str(k + 1))
            fixed_acc.append(acc)
            fixed_macs.append(cum[k])
            deltas.append(r["acc"] - acc)
        rows.append({**compare(f"{label}: cascata − saída fixa", "acc", deltas),
                     "saida_fixa": " / ".join(picks),
                     "cascata_acc_mean": np.mean([r["acc"] for r in recs]),
                     "cascata_macs_mean": np.mean([r["macs"] for r in recs]),
                     "fixa_acc_mean": np.mean(fixed_acc),
                     "fixa_macs_mean": np.mean(fixed_macs)})
    return rows


def curve(t, mode):
    d = load(t, "test")
    pts = []
    for p in P_TAR_LIST:
        correct, macs, e = cascade(d, mode, p)
        edge = e < d["logits"].shape[1] - 1
        pts.append({"p_tar": p, "acc": 100 * correct.mean(), "macs": macs,
                    "frac_edge": 100 * edge.mean(),
                    "acc_edge": 100 * correct[edge].mean() if edge.any() else np.nan})
    return pts


def best_within(pts, budget):
    ok = [p["acc"] for p in pts if p["macs"] <= budget]
    return max(ok) if ok else np.nan


def curves_and_budgets():
    curve_rows, budget_rows, budget_seed = [], [], {}
    for label, arch, n, w in CONFIGS:
        for mode in MODES:
            per_seed = [curve(tag(n, w, arch, s), mode) for s in SEEDS]
            for i, p in enumerate(P_TAR_LIST):
                row = {"config": label, "mode": mode, "p_tar": p}
                for m in ("acc", "macs", "frac_edge", "acc_edge"):
                    mean, std, _ = summarize([c[i][m] for c in per_seed])
                    row[f"{m}_mean"], row[f"{m}_std"] = mean, std
                curve_rows.append(row)
            for b in BUDGETS:
                vals = [best_within(c, b) for c in per_seed]
                budget_seed[(label, mode, b)] = vals
                mean, std, k = summarize(vals)
                budget_rows.append({"config": label, "mode": mode, "orcamento_M": b,
                                    "acc_mean": mean, "acc_std": std, "sementes_que_alcancam": k})
    return curve_rows, budget_rows, budget_seed


def ece(conf, correct):
    bins = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
    total = 0.0
    for b in range(N_BINS):
        m = bins == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return 100 * total


def ece_table():
    rows = []
    for label, arch, n, w in CONFIGS:
        for mode in MODES:
            per_seed = []
            for s in SEEDS:
                d = load(tag(n, w, arch, s), "test")
                z = d["logits"] / temperatures(d, mode).reshape(1, -1, 1)
                z = z - z.max(axis=2, keepdims=True)
                p = np.exp(z)
                p /= p.sum(axis=2, keepdims=True)
                per_seed.append([ece(p[:, e].max(1), p[:, e].argmax(1) == d["labels"])
                                 for e in range(n + 1)])
            arr = np.array(per_seed)
            row = {"config": label, "mode": mode}
            for e in range(n + 1):
                name = "final" if e == n else f"saida{e + 1}"
                row[f"{name}_mean"], row[f"{name}_std"] = arr[:, e].mean(), arr[:, e].std(ddof=1)
            rows.append(row)
    return rows


def temperature_table():
    rows = []
    for label, arch, n, w in CONFIGS:
        ds = [load(tag(n, w, arch, s), "test") for s in SEEDS]
        tb = np.array([d["T_branches"] for d in ds])
        to = np.array([d["T_overall"] for d in ds])
        rows.append({"config": label,
                     "T_por_saida": " / ".join(f"{m:.2f}" for m in tb.mean(0)),
                     "T_overall_mean": to.mean(), "T_overall_std": to.std(ddof=1)})
    return rows


def history_minima(arch, n, w, seed):
    path = os.path.join(SAVE_DIR, "history",
                        f"mobilenet_{variant(arch, seed, 'loss')}_b{n}_{w}.csv")
    h = pd.read_csv(path)
    cols = [c for c in h.columns if c.startswith("val_loss_")]
    return {c.removeprefix("val_loss_"): int(h.loc[h[c].idxmin(), "epoch"]) for c in cols}


def selection():
    rows = []
    for label, arch, n, w in CONFIGS:
        for s in SEEDS:
            rec = {"config": label, "semente": s}
            for sel in SELECTIONS:
                t = tag(n, w, arch, s, sel)
                ck = torch.load(os.path.join(SAVE_DIR, f"{t}.pth"), map_location="cpu",
                                weights_only=False)
                acc = exit_accs(t)
                rec[f"epoca_{sel}"] = ck["epoch"]
                rec[f"final_{sel}"] = acc[-1]
                rec[f"media_{sel}"] = acc.mean()
                rec[f"saida1_{sel}"] = acc[0]
            rec["delta_final_acc_loss"] = rec["final_acc"] - rec["final_loss"]
            rec["delta_media_acc_loss"] = rec["media_acc"] - rec["media_loss"]
            rec["delta_final_last_acc"] = rec["final_last"] - rec["final_acc"]
            minima = history_minima(arch, n, w, s)
            rec["minimo_val_loss"] = " / ".join(f"{k} {v}" for k, v in minima.items())
            rows.append(rec)
    return rows


def training_times():
    pattern = re.compile(r"OK: TREINO b(\d+) (\S+) (\S+) \(semente \d+, divisão \d+\) \((\d+) min\)")
    times = {}
    for f in glob.glob("logs/retrain_v5_*.log"):
        for line in open(f, errors="ignore"):
            m = pattern.search(line)
            if m:
                n, w, v, minutes = m.groups()
                arch = v.rsplit("r", 1)[0].rstrip("-")
                label = {"s7-14": "v3b", "scratch": "b0 sem pré-treino"}.get(
                    arch, "b0" if n == "0" else f"b{n} {w}")
                times.setdefault(label, []).append(int(minutes))
    rows = [{"config": k, "treinos": len(v), "min_mean": np.mean(v), "min_total": sum(v)}
            for k, v in sorted(times.items())]
    rows.append({"config": "total", "treinos": sum(r["treinos"] for r in rows),
                 "min_mean": np.nan, "min_total": sum(r["min_total"] for r in rows)})
    return rows


def comparisons(per_seed, exits_seed, b0, budget_seed):
    rows = []

    def op(label, metric="acc", mode="overall"):
        return [r[metric] for r in per_seed[(label, mode)]]

    def diff(a, b):
        return [x - y for x, y in zip(a, b)]

    for n in (2, 3, 4, 5):
        rows.append(compare(f"b{n}: linspace − equal", "acc", diff(op(f"b{n} linspace"), op(f"b{n} equal"))))
        rows.append(compare(f"b{n}: linspace − equal", "macs", diff(op(f"b{n} linspace", "macs"), op(f"b{n} equal", "macs"))))
    for w in ("equal", "linspace"):
        for n in (3, 4, 5):
            rows.append(compare(f"b{n} − b2 ({w})", "acc", diff(op(f"b{n} {w}"), op(f"b2 {w}"))))
    rows.append(compare("b3: inv_linspace − equal", "acc", diff(op("b3 inv_linspace"), op("b3 equal"))))
    rows.append(compare("b3: inv_linspace − linspace", "acc", diff(op("b3 inv_linspace"), op("b3 linspace"))))
    rows.append(compare("v3b − b3 equal", "acc", diff(op("v3b"), op("b3 equal"))))
    rows.append(compare("v3b ÷ b3 equal", "macs", [a / b for a, b in zip(op("v3b", "macs"), op("b3 equal", "macs"))]))
    for label, *_ in CONFIGS:
        rows.append(compare(f"{label}: cascata − b0", "acc", diff(op(label), b0[("b0", MAIN)])))
        rows.append(compare(f"{label}: saída final − b0", "acc", diff(exits_seed[(label, MAIN)], b0[("b0", MAIN)])))
    rows.append(compare("b0 − b0 sem pré-treino", "acc", diff(b0[("b0", MAIN)], b0[("b0 sem pré-treino", MAIN)])))
    for label, *_ in CONFIGS:
        rows.append(compare(f"{label}: overall − none", "acc", diff(op(label), op(label, mode="none"))))
        rows.append(compare(f"{label}: overall − branches", "acc", diff(op(label), op(label, mode="branches"))))
    for n in (2, 3, 4, 5):
        for b in BUDGETS:
            a, c = budget_seed[(f"b{n} linspace", "overall", b)], budget_seed[(f"b{n} equal", "overall", b)]
            if not any(np.isnan(a)) and not any(np.isnan(c)):
                rows.append(compare(f"b{n}: linspace − equal até {b} M MACs", "acc", diff(a, c)))
            a, c = budget_seed[(f"b{n} equal", "overall", b)], budget_seed[(f"b{n} equal", "none", b)]
            if not any(np.isnan(a)) and not any(np.isnan(c)):
                rows.append(compare(f"b{n} equal: overall − none até {b} M MACs", "acc", diff(a, c)))
    return rows


def write(rows, name):
    if not rows:
        return
    fields = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    out = os.path.join(RESULTS_DIR, name)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: round(float(v), 3) if isinstance(v, (float, np.floating)) else v
                        for k, v in r.items()})
    print(f"{len(rows)} linhas em {out}")


def main():
    op = pd.read_csv(os.path.join(RESULTS_DIR, "stats_operating_points.csv"))
    b0_macs = float(load(tag(2, "equal", "", SEEDS[0]), "test")["cum_macs"][-1]) / 1e6

    exits = exits_table()
    write(exits, "replicas_exits.csv")
    exits_seed = {(label, MAIN): [exit_accs(tag(n, w, arch, s))[-1] for s in SEEDS]
                  for label, arch, n, w in CONFIGS}

    b0, b0_rows = baselines()
    write(b0_rows, "replicas_baseline.csv")

    ops, per_seed = operating_points(op, b0_macs)
    for label, _ in BASELINES:
        mean, std, k = summarize(b0[(label, MAIN)])
        ops.append({"config": label, "mode": "", "acc_mean": mean, "acc_std": std,
                    "macs_mean": b0_macs, "macs_rel_b0": 100.0, "frac_edge_mean": 0.0})
    write(ops, "replicas_operating_points.csv")
    write(fixed_exit(per_seed), "replicas_fixed_exit.csv")

    curve_rows, budget_rows, budget_seed = curves_and_budgets()
    write(curve_rows, "replicas_curves.csv")
    write(budget_rows, "replicas_budget.csv")

    write(ece_table(), "replicas_ece.csv")
    write(temperature_table(), "replicas_temperatures.csv")

    sel = selection()
    write(sel, "replicas_selection.csv")
    ep = np.array([r["epoca_loss"] for r in sel], dtype=float)
    d = np.array([r["delta_final_acc_loss"] for r in sel])
    print(f"seleção: correlação época do mínimo da loss x ganho na final = {np.corrcoef(ep, d)[0, 1]:.2f}")

    write(training_times(), "replicas_times.csv")
    write(comparisons(per_seed, exits_seed, b0, budget_seed), "replicas_comparisons.csv")
    for label, _ in BASELINES:
        for s in SELECTIONS:
            mean, std, _ = summarize(b0[(label, s)])
            print(f"{label} [{s}]: {mean:.2f} ± {std:.2f}")


if __name__ == "__main__":
    main()
