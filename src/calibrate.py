"""Temperature scaling para early-exit DNN.

Replica a receita do orientador (Guo et al., 2017 + variação per-branch):
  - Per-branch: 1 temperatura T_i por saída (total: n_exits temperaturas)
  - Overall:    1 temperatura T única aplicada em todas as saídas

Usa a metade de CALIBRAÇÃO da validação (src/data.py) — disjunta da metade que
o train.py usa para early stopping e seleção de checkpoint. Assim as amostras
que ajustam a temperatura não são as mesmas que escolheram o modelo, e nenhuma
delas foi vista no treino. A normalização é lida do checkpoint (legados: 0.5).

Temperaturas são salvas de volta no próprio checkpoint sob as chaves:
  - temperatures_branches: list[float]
  - temperature_overall:   float

Uso:
    uv run python -m src.calibrate
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from src import data
from src.inference import load_model
from src.model import checkpoint_filename, parse_extra_stride1, variant_tag


# --- Configuração (edite aqui ou sobreponha via env vars) ---
N_BRANCHES = int(os.environ.get("N_BRANCHES", "2"))
LOSS_WEIGHTS_LIST = os.environ.get("LOSS_WEIGHTS_LIST", "equal,linspace").split(",")
SAVE_DIR = os.environ.get("SAVE_DIR", "saved_model")
EXTRA_STRIDE1_BLOCKS = parse_extra_stride1(os.environ.get("EXTRA_STRIDE1", ""))
VARIANT = os.environ.get("VARIANT", variant_tag(EXTRA_STRIDE1_BLOCKS))
MODEL_PATHS = [
    os.path.join(SAVE_DIR,
                 checkpoint_filename(N_BRANCHES, w.replace(",", "-"), VARIANT))
    for w in LOSS_WEIGHTS_LIST
]

# --- Hiperparâmetros fixos ---
BATCH_SIZE = 128
DATA_ROOT = "./data"
LBFGS_LR = 0.01
LBFGS_MAX_ITER = 50
T_INIT = 1.5               # convenção de Guo et al.


@torch.no_grad()
def collect_logits(model, loader, device):
    """Roda o modelo em todo o loader e devolve logits por saída + labels."""
    model.eval()
    n_exits = model.n_branches + 1
    logits_per_exit = [[] for _ in range(n_exits)]
    labels_all = []
    for images, labels in loader:
        images = images.to(device)
        outs = model(images)
        for i, o in enumerate(outs):
            logits_per_exit[i].append(o.cpu())
        labels_all.append(labels)
    return (
        [torch.cat(lg) for lg in logits_per_exit],
        torch.cat(labels_all),
    )


def fit_temperature(logits, labels, device):
    """Otimiza 1 T escalar via LBFGS pra minimizar NLL. Retorna (T, nll_antes, nll_depois)."""
    logits = logits.to(device)
    labels = labels.to(device)
    T = nn.Parameter(torch.ones(1, device=device) * T_INIT)
    nll = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([T], lr=LBFGS_LR, max_iter=LBFGS_MAX_ITER)

    with torch.no_grad():
        nll_before = nll(logits / T_INIT, labels).item()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        nll_after = nll(logits / T, labels).item()

    return T.item(), nll_before, nll_after


def build_val_loader(mean, std):
    """Metade de calibração da validação — DISJUNTA da usada para seleção."""
    _, _, val_calib = data.train_val_datasets(DATA_ROOT, mean, std)
    return torch.utils.data.DataLoader(
        val_calib, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    ), len(val_calib)


def calibrate_one(model_path, device):
    print("=" * 60)
    print(f"Calibrando: {model_path}")
    print("=" * 60)

    model, ckpt_meta = load_model(model_path, device)
    n_exits = model.n_branches + 1
    mean, std = data.norm_from_ckpt(ckpt_meta)
    val_loader, val_size = build_val_loader(mean, std)
    print(f"Modelo: n_branches={model.n_branches}  n_exits={n_exits}")
    print(f"Normalização: mean={mean} std={std}")
    print(f"Val set: {val_size} amostras")

    logits_per_exit, labels = collect_logits(model, val_loader, device)

    # --- Per-branch: 1 T por saída ---
    print("\n[Per-branch]")
    temperatures_branches = []
    for i, logits in enumerate(logits_per_exit):
        T, nll_before, nll_after = fit_temperature(logits, labels, device)
        temperatures_branches.append(T)
        print(f"  Saída {i}: T = {T:.4f}  |  NLL: {nll_before:.4f} → {nll_after:.4f}")

    # --- Overall: 1 T única para todas as saídas ---
    # Concatena logits e repete labels → otimiza 1 T que minimize NLL global.
    print("\n[Overall]")
    all_logits = torch.cat(logits_per_exit, dim=0)   # [N * n_exits, C]
    all_labels = labels.repeat(n_exits)              # [N * n_exits]
    T_overall, nll_before, nll_after = fit_temperature(all_logits, all_labels, device)
    print(f"  T_overall = {T_overall:.4f}  |  NLL: {nll_before:.4f} → {nll_after:.4f}")

    # Salva no checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    ckpt["temperatures_branches"] = temperatures_branches
    ckpt["temperature_overall"] = T_overall
    torch.save(ckpt, model_path)
    print(f"\nSalvo em: {model_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    for path in MODEL_PATHS:
        calibrate_one(path, device)
        print()


if __name__ == "__main__":
    main()
