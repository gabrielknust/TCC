import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex

from src.plots import CAT, INK, INK_2, MUTED, OUT_DIR, SURFACE, ink_on, new_fig
from src.stats import RESULTS_DIR, cascade, load, select_ptar

CLASSES = ["avião", "automóvel", "pássaro", "gato", "veado",
           "cachorro", "sapo", "cavalo", "navio", "caminhão"]
MODE = "overall"
MODELS = [f"mobilenet_r{s}-last_b3_linspace_best" for s in (1, 2, 3, 4, 5)]
FIGURE_MODEL = "mobilenet_r1-last_b3_linspace_best"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def per_class(tag):
    p_tar = select_ptar(tag, MODE)
    d = load(tag, "test")
    correct, _, exit_idx = cascade(d, MODE, p_tar)
    n_exits = d["logits"].shape[1]
    rows = []
    for c, name in enumerate(CLASSES):
        m = d["labels"] == c
        e, ok = exit_idx[m], correct[m]
        edge = e < n_exits - 1
        row = {"tag": tag, "p_tar": p_tar, "classe": name, "n": int(m.sum())}
        for i in range(n_exits - 1):
            row[f"frac_exit{i + 1}"] = 100 * np.mean(e == i)
        row["frac_cloud"] = 100 * np.mean(~edge)
        row["acc"] = 100 * ok.mean()
        row["acc_edge"] = 100 * ok[edge].mean() if edge.any() else float("nan")
        row["acc_cloud"] = 100 * ok[~edge].mean() if (~edge).any() else float("nan")
        row["acc_final_only"] = 100 * np.mean(d["logits"][m, -1].argmax(1) == c)
        rows.append(row)
    return rows


def plot(rows):
    exit_cols = [k for k in rows[0] if k.startswith("frac_exit")] + ["frac_cloud"]
    order = sorted(rows, key=lambda r: r["frac_cloud"])
    mat = np.array([[r[c] for c in exit_cols] for r in order])
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    norm = Normalize(0, 100)

    fig, axes = new_fig(1, 2, w=4.4, h=4.6, gridspec_kw={"width_ratios": [1.5, 1]})
    ax, ax2 = axes[0]
    ax.grid(False)
    ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color=ink_on(to_hex(cmap(norm(mat[i, j])))))
    ax.set_xticks(range(len(exit_cols)))
    ax.set_xticklabels([f"saída {i + 1}" for i in range(len(exit_cols) - 1)] + ["nuvem"])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([r["classe"] for r in order], color=INK_2, fontsize=9)
    ax.set_title("% das amostras da classe resolvidas em cada saída", fontsize=10)
    for side in ("bottom", "left"):
        ax.spines[side].set_visible(False)

    y = np.arange(len(order))
    accs = [r["acc"] for r in order]
    mean_acc = float(np.mean([r["acc"] for r in rows]))
    ax2.axvline(mean_acc, color=MUTED, lw=1, ls="--")
    ax2.scatter(accs, y, s=56, color=CAT[2], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for yi, a in zip(y, accs):
        ax2.text(a + 0.6, yi, f"{a:.1f}", va="center", fontsize=8, color=INK_2)
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlim(min(accs) - 3, 100)
    ax2.set_xlabel("acurácia da cascata na classe (%)", fontsize=9)
    ax2.set_title(f"acurácia por classe (tracejado: média {mean_acc:.1f}%)", fontsize=10)

    r0 = rows[0]
    fig.suptitle(
        f"Por classe (b3 linspace, semente 1, calibração {MODE}, p_tar {r0['p_tar']:.2f})",
        fontsize=11, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT_DIR, "per_class_exits.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def main():
    all_rows = []
    for tag in MODELS:
        all_rows.extend(per_class(tag))
    fields = []
    for r in all_rows:
        fields.extend(k for k in r if k not in fields)
    out = os.path.join(RESULTS_DIR, "per_class.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()})
    print(f"{len(all_rows)} linhas salvas em {out}")
    print(plot([r for r in all_rows if r["tag"] == FIGURE_MODEL]))


if __name__ == "__main__":
    main()
