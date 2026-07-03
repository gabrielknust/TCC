"""Inferência com cascata edge→cloud.

Avalia um modelo treinado em todo o test set para uma lista de thresholds (p_tar)
e gera tabela com: acurácia global (± IC95%), fração borda vs nuvem, % de amostras
e acurácia por saída.

Suporta 3 modos de calibração via CALIB_MODE:
  - "none":     usa logits crus (sem T)
  - "overall":  divide logits por temperature_overall (1 T para todas as saídas)
  - "branches": divide logits por temperatures_branches[i] (1 T por saída)

Os modos overall/branches só funcionam se o checkpoint tiver passado por src/calibrate.py.

Convenção borda/nuvem: os N ramos antecipados rodam na borda (edge); a saída
final representa o offload para a nuvem (cloud).

O IC95% é a meia-largura do intervalo de confiança binomial (aproximação normal):
1.96 * sqrt(p*(1-p)/n). Saídas com poucas amostras têm intervalos largos — é a
"variância" esperada da estimativa de acurácia, não instabilidade do modelo.

Uso:
    uv run python -m src.inference
"""

import math
import os
from datetime import datetime

import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms

from src.model import MobileNetV2EarlyExit


# --- Configuração do experimento (edite aqui ou sobreponha via env vars) ---
# Cada combinação (modelo × modo) gera 1 CSV. Total = len(MODEL_PATHS) × len(CALIB_MODES).
N_BRANCHES = int(os.environ.get("N_BRANCHES", "2"))
LOSS_WEIGHTS_LIST = os.environ.get("LOSS_WEIGHTS_LIST", "equal,linspace").split(",")
MODEL_PATHS = [
    f"saved_model/mobilenet_b{N_BRANCHES}_{w.replace(',', '-')}_best.pth"
    for w in LOSS_WEIGHTS_LIST
]
CALIB_MODES = ["none", "overall", "branches"]
P_TAR_LIST = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50 … 0.95

# --- Hiperparâmetros fixos ---
BATCH_SIZE = 128
DATA_ROOT = "./data"


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MobileNetV2EarlyExit(
        n_branches=ckpt["n_branches"],
        exit_positions=ckpt["exit_positions"],
        num_classes=ckpt["num_classes"],
        input_size=ckpt["input_size"],
        pretrained=False,
    ).to(device)
    # strict=False tolera buffers extras de thop (total_ops/total_params) em
    # checkpoints antigos salvos antes do fix de _linear_flop_positions.
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    leaked = [k for k in unexpected if "total_ops" not in k and "total_params" not in k]
    if missing or leaked:
        raise RuntimeError(
            f"state_dict incompatível. missing={missing} unexpected={leaked}"
        )
    model.eval()
    return model, ckpt


def resolve_temperatures(ckpt, n_exits, mode):
    """Devolve lista [T_0, ..., T_{n_exits-1}] de acordo com o modo escolhido."""
    if mode == "none":
        return [1.0] * n_exits
    if mode == "overall":
        T = ckpt.get("temperature_overall")
        if T is None:
            raise RuntimeError(
                "CALIB_MODE='overall' mas checkpoint não tem 'temperature_overall'. "
                "Rode src/calibrate.py primeiro."
            )
        return [float(T)] * n_exits
    if mode == "branches":
        Ts = ckpt.get("temperatures_branches")
        if Ts is None:
            raise RuntimeError(
                "CALIB_MODE='branches' mas checkpoint não tem 'temperatures_branches'. "
                "Rode src/calibrate.py primeiro."
            )
        if len(Ts) != n_exits:
            raise RuntimeError(
                f"temperatures_branches tem {len(Ts)} valores, modelo tem {n_exits} saídas"
            )
        return [float(t) for t in Ts]
    raise ValueError(f"CALIB_MODE inválido: {mode}")


@torch.no_grad()
def collect_all_logits(model, loader, device):
    """Roda o modelo batched UMA vez e guarda os logits de todas as saídas.

    Os modos de calibração diferem só na temperatura aplicada aos logits, então
    coletar os logits uma única vez evita 1 forward pass completo por modo.
    Retorna logits [N, n_exits, C] (CPU) e labels [N].
    """
    logits_all, labels_all = [], []
    for images, labels in loader:
        images = images.to(device)
        outs = model(images)
        logits_all.append(torch.stack(outs, dim=1).cpu())  # [B, E, C]
        labels_all.append(labels)
    return torch.cat(logits_all), torch.cat(labels_all)


def apply_temperatures(logits, temperatures):
    """softmax(logits / T_i) por saída → (confs [N, E], preds [N, E])."""
    n_exits = len(temperatures)
    T = torch.tensor(temperatures).view(1, n_exits, 1)  # broadcast em [N, E, C]
    probs = torch.softmax(logits / T, dim=2)
    confs, preds = probs.max(dim=2)
    return confs, preds


def _ci95_half_width(correct_mask):
    """Meia-largura do IC95% binomial (aprox. normal) da acurácia, em pontos %."""
    n = correct_mask.numel()
    if n == 0:
        return float("nan")
    p = correct_mask.float().mean().item()
    return 100 * 1.96 * math.sqrt(p * (1 - p) / n)


def cascade_metrics(confs, preds, labels, p_tar):
    """Para cada amostra, escolhe a primeira saída cuja conf >= p_tar.
    Se nenhuma satisfaz, usa a saída final (nuvem).
    """
    n_samples, n_exits = confs.shape
    above = confs >= p_tar  # [N, E]
    has_above = above.any(dim=1)
    first_above = above.float().argmax(dim=1)
    exit_idx = torch.where(has_above, first_above, torch.tensor(n_exits - 1))

    chosen_pred = preds[torch.arange(n_samples), exit_idx]
    correct = (chosen_pred == labels)

    metrics = {
        "p_tar": p_tar,
        "acc_global_%": 100 * correct.float().mean().item(),
        "acc_global_ci95_%": _ci95_half_width(correct),
    }

    # Agregado borda (ramos 0..E-2) vs nuvem (saída final E-1).
    edge_mask = exit_idx < n_exits - 1
    cloud_mask = ~edge_mask
    metrics["frac_edge_%"] = 100 * edge_mask.float().mean().item()
    metrics["acc_edge_%"] = (
        100 * correct[edge_mask].float().mean().item()
        if edge_mask.any() else float("nan")
    )
    metrics["frac_cloud_%"] = 100 * cloud_mask.float().mean().item()
    metrics["acc_cloud_%"] = (
        100 * correct[cloud_mask].float().mean().item()
        if cloud_mask.any() else float("nan")
    )

    for e in range(n_exits):
        mask = exit_idx == e
        n = mask.sum().item()
        metrics[f"frac_exit{e}_%"] = 100 * n / n_samples
        if n > 0:
            metrics[f"acc_exit{e}_%"] = 100 * correct[mask].float().mean().item()
            metrics[f"acc_exit{e}_ci95_%"] = _ci95_half_width(correct[mask])
        else:
            metrics[f"acc_exit{e}_%"] = float("nan")
            metrics[f"acc_exit{e}_ci95_%"] = float("nan")
    return metrics


def run_model(model_path, loader, n_test, device):
    """Coleta logits do modelo uma vez e gera 1 CSV por modo de calibração."""
    print("=" * 60)
    print(f"Modelo: {model_path}")
    print("=" * 60)

    model, ckpt = load_model(model_path, device)
    n_exits = model.n_branches + 1
    print(f"n_branches={model.n_branches}  positions={model.exit_positions}")
    print(f"Coletando logits para {n_test} amostras...")
    logits, labels = collect_all_logits(model, loader, device)

    model_tag = os.path.splitext(os.path.basename(model_path))[0]
    for calib_mode in CALIB_MODES:
        temperatures = resolve_temperatures(ckpt, n_exits, calib_mode)
        print(f"\n[calib={calib_mode}]  T={[f'{t:.4f}' for t in temperatures]}")
        confs, preds = apply_temperatures(logits, temperatures)

        rows = [cascade_metrics(confs, preds, labels, p) for p in P_TAR_LIST]
        df = pd.DataFrame(rows)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = f"results/cascade_{model_tag}_{calib_mode}_{timestamp}.csv"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False, float_format="%.2f")

        cols = ["p_tar", "acc_global_%", "acc_global_ci95_%",
                "frac_edge_%", "acc_edge_%", "frac_cloud_%", "acc_cloud_%"]
        print(df[cols].to_string(index=False))
        print(f"Salvo em: {out}")
    print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    for model_path in MODEL_PATHS:
        run_model(model_path, loader, len(testset), device)


if __name__ == "__main__":
    main()
