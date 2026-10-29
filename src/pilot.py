import csv
import os

import numpy as np
import pandas as pd
import torch

from src import data
from src.inference import load_model
from src.model import acc_variant_tag, checkpoint_filename

OLD_DIR = os.environ.get("OLD_DIR", "saved_model")
NEW_DIR = os.environ.get("NEW_DIR", "saved_model_v5/pilot")
VARIANT = os.environ.get("VARIANT", "r1")
CONFIGS = [(0, "equal"), (3, "equal"), (3, "linspace")]
TOLERANCE = 0.3
OUT = os.path.join("logs", "pilot_v5_comparacao.csv")


@torch.no_grad()
def exit_accuracies(path, device):
    model, ckpt = load_model(path, device)
    mean, std = data.norm_from_ckpt(ckpt)
    loader = torch.utils.data.DataLoader(
        data.test_dataset("./data", mean, std), batch_size=512, num_workers=0)
    correct, total = None, 0
    for images, labels in loader:
        outs = model(images.to(device))
        hits = torch.stack([(o.argmax(1).cpu() == labels) for o in outs])
        correct = hits.sum(1) if correct is None else correct + hits.sum(1)
        total += labels.numel()
    return (100 * correct.float() / total).tolist(), ckpt["epoch"]


def history_summary(n, w):
    path = os.path.join(NEW_DIR, "history",
                        checkpoint_filename(n, w, VARIANT).replace("_best.pth", ".csv"))
    if not os.path.exists(path):
        return ""
    h = pd.read_csv(path)
    loss_cols = [c for c in h.columns if c.startswith("val_loss_")]
    acc_cols = [c for c in h.columns if c.startswith("val_acc_")]
    minima = ", ".join(
        f"{c.removeprefix('val_loss_')} {int(h.loc[h[c].idxmin(), 'epoch'])}" for c in loss_cols)
    best_mean = int(h.loc[h[acc_cols].mean(axis=1).idxmax(), "epoch"])
    return (f"época do mínimo da loss de validação por saída: {minima}; "
            f"melhor média de acurácia na época {best_mean}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, alerts = [], []
    for n, w in CONFIGS:
        for sel, variant in (("loss", VARIANT), ("acc", acc_variant_tag(VARIANT))):
            name = checkpoint_filename(n, w, variant)
            old_path, new_path = os.path.join(OLD_DIR, name), os.path.join(NEW_DIR, name)
            if not os.path.exists(new_path):
                print(f"faltando: {new_path}")
                continue
            new, new_ep = exit_accuracies(new_path, device)
            old, old_ep = exit_accuracies(old_path, device) if os.path.exists(old_path) else ([np.nan] * len(new), None)
            labels = [f"S{i+1}" for i in range(len(new) - 1)] + ["Final"]
            row = {"config": f"b{n} {w}", "selecao": sel, "epoca_antiga": old_ep, "epoca_nova": new_ep}
            for lab, o, nw in zip(labels, old, new):
                row[f"{lab}_antiga"], row[f"{lab}_nova"] = round(o, 2), round(nw, 2)
            row["delta_final"] = round(new[-1] - old[-1], 2)
            row["delta_media"] = round(float(np.mean(new) - np.mean(old)), 2)
            rows.append(row)
            exits = " | ".join(f"{lab} {o:.2f} → {nw:.2f}" for lab, o, nw in zip(labels, old, new))
            print(f"b{n} {w:<8} sel. {sel:<4} épocas {old_ep} → {new_ep}: {exits} "
                  f"(Δ final {row['delta_final']:+.2f}, Δ média {row['delta_media']:+.2f})")
            if sel == "acc" and row["delta_final"] < -TOLERANCE:
                alerts.append(f"b{n} {w}: saída final {row['delta_final']:+.2f} pp")
        summary = history_summary(n, w)
        if summary:
            print(f"   b{n} {w}: {summary}")
    if rows:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nTabela salva em {OUT}")
    if alerts:
        print(f"\nATENÇÃO: a receita nova piorou mais de {TOLERANCE} pp em: " + "; ".join(alerts))
    elif rows:
        print(f"\nReceita nova sem piora acima de {TOLERANCE} pp na saída final (checkpoint por acurácia).")


if __name__ == "__main__":
    main()
