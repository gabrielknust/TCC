"""Consolidacao estatistica dos resultados da cascata.

Duas correcoes de protocolo sobre a analise anterior:

1. **p_tar escolhido na validacao.** Antes o melhor ponto de operacao era lido
   direto no test set — com 14 limiares x 3 calibracoes x varias configuracoes,
   isso e' data snooping e infla o numero reportado. Aqui o limiar e' escolhido
   maximizando a acuracia na validacao e o teste e' reportado so' nesse ponto.

2. **McNemar pareado** no lugar de comparar IC95%. Os modelos classificam as
   MESMAS amostras, entao a comparacao correta e' pareada: olha so' onde eles
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

RESULTS_DIR = "results"
PREDS_DIR = "results/preds"
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
    exit_idx = np.where(above.any(axis=1), above.argmax(axis=1), confs.shape[1] - 1)
    chosen = preds[np.arange(len(exit_idx)), exit_idx]
    return chosen == d["labels"], d["cum_macs"][exit_idx].mean() / 1e6


def select_ptar(tag, mode):
    """Limiar que maximiza a acuracia na VALIDACAO."""
    dv = load(tag, "val")
    accs = [(cascade(dv, mode, p)[0].mean(), p) for p in P_TAR_LIST]
    return max(accs)[1]


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
    correct, macs = cascade(dt, mode, p_tar)
    return {"tag": tag, "mode": mode, "p_tar": p_tar, "correct": correct,
            "acc": 100 * correct.mean(), "ci": ci95(correct), "macs": macs}


def fmt(r):
    return f"{r['acc']:.2f}±{r['ci']:.2f} @{r['macs']:5.1f}M (p={r['p_tar']:.2f})"


ROWS = []
GROUP = {"A": None}


def paired_ci95(n_a, n_b, n_total):
    """IC95% de Wald para a diferenca pareada de proporcoes (Agresti)."""
    d = n_a - n_b
    var = n_a + n_b - d * d / n_total
    return 100 * 1.96 * math.sqrt(max(var, 0.0)) / n_total


def compare(label, ra, rb):
    n_a, n_b, p = mcnemar(ra["correct"], rb["correct"])
    delta = ra["acc"] - rb["acc"]
    sig = "SIM" if p < 0.05 else "nao"
    print(f"  {label:<34} Δ={delta:+6.2f}pp  discordam {n_a + n_b:>4} "
          f"({n_a} vs {n_b})  p={p:.4f}  significativo: {sig}")
    ROWS.append({
        "group": GROUP["A"], "label": label, "delta": round(delta, 3),
        "ci95": round(paired_ci95(n_a, n_b, len(ra["correct"])), 3),
        "n_a": n_a, "n_b": n_b, "discordantes": n_a + n_b,
        "p_value": p, "significant": int(p < 0.05),
        "p_tar_a": ra["p_tar"], "p_tar_b": rb["p_tar"],
    })


def main():
    tags = sorted({os.path.basename(f).rsplit("_", 1)[0]
                   for f in glob.glob(f"{PREDS_DIR}/*_test.npz")})
    print(f"{len(tags)} modelos em {PREDS_DIR}\n")

    print("=" * 96)
    print("TABELA 1 — p_tar escolhido na VALIDACAO, acuracia reportada no TESTE")
    print("=" * 96)
    R = {}
    for tag in tags:
        for mode in CALIB_MODES:
            R[(tag, mode)] = evaluate(tag, mode)
        print(f"{tag:<40} " + " | ".join(
            f"{m}: {fmt(R[(tag, m)])}" for m in ("none", "overall")))

    def get(tag, mode="overall"):
        return R.get((f"mobilenet_{tag}", mode))

    print()
    print("=" * 96)
    print("TABELA 2 — McNemar pareado (mesmas 10k amostras)")
    print("=" * 96)

    GROUP["A"] = "A"
    print("\n[A] linspace vs equal (calib overall, checkpoint por val acc)")
    for n in (2, 3, 4, 5):
        a, b = get(f"acc_b{n}_linspace_best"), get(f"acc_b{n}_equal_best")
        if a and b:
            compare(f"b{n}: linspace vs equal", a, b)

    GROUP["A"] = "B"
    print("\n[B] overall vs branches (equal, checkpoint por val acc)")
    for n in (2, 3, 4, 5):
        a, b = R.get((f"mobilenet_acc_b{n}_equal_best", "overall")), R.get((f"mobilenet_acc_b{n}_equal_best", "branches"))
        if a and b:
            compare(f"b{n}: overall vs branches", a, b)

    GROUP["A"] = "C"
    print("\n[C] calibracao vs nenhuma (equal, val acc)")
    for n in (2, 3, 4, 5):
        a, b = R.get((f"mobilenet_acc_b{n}_equal_best", "overall")), R.get((f"mobilenet_acc_b{n}_equal_best", "none"))
        if a and b:
            compare(f"b{n}: overall vs sem calib", a, b)

    GROUP["A"] = "D"
    print("\n[D] arquitetura (b3 equal, overall, val acc)")
    base = get("acc_b3_equal_best")
    for tag, lbl in [("s14-acc_b3_equal_best", "v3a (4x4) vs padrao"),
                     ("s7-14-acc_b3_equal_best", "v3b (8x8) vs padrao")]:
        r = get(tag)
        if r and base:
            compare(lbl, r, base)

    GROUP["A"] = "E"
    print("\n[E] criterio de selecao de checkpoint (val acc vs val loss, overall)")
    for n in (2, 3, 4, 5):
        for st in ("equal", "linspace"):
            a, b = get(f"acc_b{n}_{st}_best"), get(f"b{n}_{st}_best")
            if a and b:
                compare(f"b{n} {st}: sel acc vs sel loss", a, b)
    for tag, lbl in [("s14", "v3a"), ("s7-14", "v3b")]:
        a, b = get(f"{tag}-acc_b3_equal_best"), get(f"{tag}_b3_equal_best")
        if a and b:
            compare(f"{lbl}: sel acc vs sel loss", a, b)

    GROUP["A"] = "F"
    print("\n[F] numero de ramos (equal, overall, val acc)")
    for n in (3, 4, 5):
        a, b = get(f"acc_b{n}_equal_best"), get("acc_b2_equal_best")
        if a and b:
            compare(f"b{n} vs b2", a, b)

    out = os.path.join(RESULTS_DIR, "stats_comparisons.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ROWS[0].keys()))
        w.writeheader()
        w.writerows(ROWS)
    print(f"\n{len(ROWS)} comparacoes salvas em {out}")


if __name__ == "__main__":
    main()
