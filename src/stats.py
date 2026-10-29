"""Consolidacao estatistica dos resultados da cascata.

Duas correcoes de protocolo sobre a analise anterior:

1. **p_tar escolhido na validacao.** Antes o melhor ponto de operacao era lido
   direto no test set — com 14 limiares x 3 calibracoes x varias configuracoes,
   isso e' data snooping e infla o numero reportado. Aqui o limiar e' escolhido
   na validacao (o menor que nao perde para a saida final sozinha) e o teste e'
   reportado so' nesse ponto.

2. **McNemar pareado**, so' entre modos de calibracao do mesmo modelo, com
   correcao de Holm. As mesmas amostras passam pelo mesmo modelo, entao a
   comparacao correta e' pareada: olha so' onde eles
   discordam (b = A acerta e B erra; c = o inverso) e testa se o desequilibrio
   entre b e c e' maior que o acaso. Bem mais sensivel que comparar intervalos,
   que tratam as duas medidas como independentes.

Uso:
    uv run python -m src.stats
"""

import csv
import glob
import math
import os

import numpy as np

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
PREDS_DIR = os.path.join(RESULTS_DIR, "preds")
CALIB_MODES = ["none", "overall", "branches"]
P_TAR_LIST = [round(0.50 + 0.05 * i, 2) for i in range(10)] + [0.96, 0.97, 0.98, 0.99]


def load(tag, split):
    f = np.load(os.path.join(PREDS_DIR, f"{tag}_{split}.npz"))
    return {
        "logits": f["logits"].astype(np.float32),
        "labels": f["labels"].astype(np.int64),
        "cum_macs": f["cum_macs"],
        "T_branches": f["temperatures_branches"],
        "T_overall": float(f["temperature_overall"]),
    }


def temperatures(d, mode):
    n = d["logits"].shape[1]
    if mode == "none":
        return np.ones(n)
    if mode == "overall":
        return np.full(n, d["T_overall"])
    return d["T_branches"]


def cascade(d, mode, p_tar):
    """(acerto por amostra [N], MACs medios) para um ponto de operacao."""
    T = temperatures(d, mode).reshape(1, -1, 1)
    z = d["logits"] / T
    z = z - z.max(axis=2, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=2, keepdims=True)
    confs, preds = p.max(axis=2), p.argmax(axis=2)

    above = confs >= p_tar
    hit = above.any(axis=1)
    exit_idx = np.where(hit, above.argmax(axis=1), confs.shape[1] - 1)
    pick = np.where(hit, exit_idx, confs.argmax(axis=1))
    chosen = preds[np.arange(len(pick)), pick]
    return chosen == d["labels"], d["cum_macs"][exit_idx].mean() / 1e6, exit_idx


def select_ptar(tag, mode):
    """Menor limiar cuja acuracia na VALIDACAO nao fica abaixo da saida final
    sozinha; se nenhum chega la, o de maior acuracia."""
    dv = load(tag, "val")
    final = (dv["logits"][:, -1].argmax(axis=1) == dv["labels"]).mean()
    accs = [(cascade(dv, mode, p)[0].mean(), p) for p in P_TAR_LIST]
    ok = [p for acc, p in accs if acc >= final]
    return min(ok) if ok else max(accs, key=lambda a: (a[0], -a[1]))[1]


def mcnemar(a, b):
    """Teste exato de McNemar. Retorna (n_a_only, n_b_only, p_valor)."""
    n_a = int(np.sum(a & ~b))
    n_b = int(np.sum(~a & b))
    n = n_a + n_b
    if n == 0:
        return n_a, n_b, 1.0
    k = min(n_a, n_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return n_a, n_b, min(1.0, 2 * tail)


def ci95(correct):
    p = correct.mean()
    return 100 * 1.96 * math.sqrt(p * (1 - p) / len(correct))


def evaluate(tag, mode):
    """Protocolo honesto: escolhe p_tar na val, reporta teste nesse ponto."""
    p_tar = select_ptar(tag, mode)
    dt = load(tag, "test")
    correct, macs, exit_idx = cascade(dt, mode, p_tar)
    edge = exit_idx < dt["logits"].shape[1] - 1
    return {"tag": tag, "mode": mode, "p_tar": p_tar, "correct": correct,
            "acc": 100 * correct.mean(), "ci": ci95(correct), "macs": macs,
            "frac_edge": 100 * edge.mean(), "frac_edge_ci": ci95(edge),
            "frac_cloud": 100 * (1 - edge.mean()),
            "acc_edge": 100 * correct[edge].mean() if edge.any() else float("nan"),
            "acc_cloud": 100 * correct[~edge].mean() if (~edge).any() else float("nan")}


def fmt(r):
    return f"{r['acc']:.2f}±{r['ci']:.2f} @{r['macs']:5.1f}M (p={r['p_tar']:.2f})"


def paired_ci95(n_a, n_b, n_total):
    """IC95% de Wald para a diferenca pareada de proporcoes (Agresti)."""
    d = n_a - n_b
    var = n_a + n_b - d * d / n_total
    return 100 * 1.96 * math.sqrt(max(var, 0.0)) / n_total


def holm(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def main():
    tags = sorted({os.path.basename(f).rsplit("_", 1)[0]
                   for f in glob.glob(f"{PREDS_DIR}/*_test.npz")})
    print(f"{len(tags)} modelos em {PREDS_DIR}\n")

    R = {(tag, mode): evaluate(tag, mode) for tag in tags for mode in CALIB_MODES}
    for tag in tags:
        print(f"{tag:<44} " + " | ".join(
            f"{m}: {fmt(R[(tag, m)])}" for m in ("none", "overall")))

    op_fields = ["tag", "mode", "p_tar", "acc", "ci", "macs", "frac_edge",
                 "frac_edge_ci", "frac_cloud", "acc_edge", "acc_cloud"]
    op_out = os.path.join(RESULTS_DIR, "stats_operating_points.csv")
    with open(op_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=op_fields)
        w.writeheader()
        for r in R.values():
            w.writerow({k: round(float(r[k]), 4) if k not in ("tag", "mode") else r[k]
                        for k in op_fields})
    print(f"\n{len(R)} pontos de operacao salvos em {op_out}")

    rows = []
    for tag in (t for t in tags if "-last_" in t):
        for a, b in (("overall", "none"), ("overall", "branches")):
            ra, rb = R[(tag, a)], R[(tag, b)]
            n_a, n_b, p = mcnemar(ra["correct"], rb["correct"])
            rows.append({
                "tag": tag, "comparacao": f"{a} vs {b}",
                "delta": round(ra["acc"] - rb["acc"], 3),
                "ci95": round(paired_ci95(n_a, n_b, len(ra["correct"])), 3),
                "n_a": n_a, "n_b": n_b, "p_value": p,
                "p_tar_a": ra["p_tar"], "p_tar_b": rb["p_tar"],
            })
    if not rows:
        return
    adjusted = holm(np.array([r["p_value"] for r in rows]))
    for r, adj in zip(rows, adjusted):
        r["p_holm"] = adj
        r["significant"] = int(adj < 0.05)
    out = os.path.join(RESULTS_DIR, "stats_comparisons.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    sig = sum(r["significant"] for r in rows)
    print(f"McNemar dentro do mesmo modelo: {len(rows)} testes, "
          f"{sig} significativos depois da correcao de Holm -> {out}")


if __name__ == "__main__":
    main()
