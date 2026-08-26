import os

import torch

from src import data
from src.inference import (
    BATCH_SIZE,
    CALIB_MODES,
    DATA_ROOT,
    MODEL_PATHS,
    collect_all_logits,
    load_model,
    resolve_temperatures,
)
from src.plots import (
    AXIS,
    CALIB_COLOR,
    CALIB_LABEL,
    INK,
    INK_2,
    MUTED,
    OUT_DIR,
    SURFACE,
    new_fig,
)

N_BINS = 15


def reliability_stats(confs, correct, n_bins=N_BINS):
    edges = torch.linspace(0, 1, n_bins + 1)
    idx = torch.bucketize(confs, edges[1:-1])
    n = confs.numel()
    centers, accs, ece = [], [], 0.0
    for b in range(n_bins):
        mask = idx == b
        nb = mask.sum().item()
        if nb == 0:
            continue
        acc = correct[mask].float().mean().item()
        conf = confs[mask].mean().item()
        ece += nb / n * abs(acc - conf)
        centers.append((edges[b].item() + edges[b + 1].item()) / 2)
        accs.append(acc)
    return centers, accs, ece


def plot_model(model_path, device):
    print("=" * 60)
    print(f"Reliability: {model_path}")
    print("=" * 60)

    model, ckpt = load_model(model_path, device)
    n_exits = model.n_branches + 1
    mean, std = data.norm_from_ckpt(ckpt)
    testset = data.test_dataset(DATA_ROOT, mean, std)
    loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    logits, labels = collect_all_logits(model, loader, device)

    fig, axes = new_fig(len(CALIB_MODES), n_exits, w=2.7, h=2.6,
                        sharex=True, sharey=True)
    for r, mode in enumerate(CALIB_MODES):
        temperatures = resolve_temperatures(ckpt, n_exits, mode)
        T = torch.tensor(temperatures).view(1, n_exits, 1)
        probs = torch.softmax(logits / T, dim=2)
        confs, preds = probs.max(dim=2)
        eces = []
        for e in range(n_exits):
            correct = preds[:, e] == labels
            centers, accs, ece = reliability_stats(confs[:, e], correct)
            eces.append(ece)
            ax = axes[r][e]
            ax.plot([0, 1], [0, 1], ls="--", lw=1, color=AXIS)
            ax.bar(centers, accs, width=0.92 / N_BINS, color=CALIB_COLOR[mode],
                   edgecolor=SURFACE, linewidth=0.5)
            ax.text(0.05, 0.92, f"ECE {100 * ece:.1f}%", fontsize=8,
                    color=INK, transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_yticks([0, 0.5, 1.0])
            if r == 0:
                ax.set_title("final" if e == n_exits - 1 else f"ramo {e + 1}",
                             fontsize=10)
            if r == len(CALIB_MODES) - 1:
                ax.set_xlabel("confiança", fontsize=9)
            if e == 0:
                ax.set_ylabel(f"{CALIB_LABEL[mode]}\nacurácia no bin", fontsize=9)
        print(f"[{mode}] ECE por saída: "
              + ", ".join(f"{100 * e:.2f}%" for e in eces))

    tag = os.path.splitext(os.path.basename(model_path))[0]
    fig.suptitle(
        f"Reliability diagrams (test set) — {tag}\n"
        "diagonal = calibração perfeita; barras abaixo = overconfidence",
        fontsize=11, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = os.path.join(OUT_DIR, f"reliability_{tag}.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"Salvo em: {out}\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    os.makedirs(OUT_DIR, exist_ok=True)
    for model_path in MODEL_PATHS:
        plot_model(model_path, device)


if __name__ == "__main__":
    main()
