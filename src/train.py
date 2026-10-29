import csv
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src import data
from src.model import (
    MobileNetV2EarlyExit,
    acc_variant_tag,
    checkpoint_filename,
    parse_extra_stride1,
    variant_tag,
)


# --- Configuração do experimento (edite aqui ou sobreponha via env vars) ---
# Os defaults rodam manualmente; o run_experiments.sh sobrepõe via env.
N_BRANCHES = int(os.environ.get("N_BRANCHES", "2"))     # nº de saídas antecipadas (1-5)
EPOCHS = int(os.environ.get("EPOCHS", "200"))           # máx. épocas (early stopping pode encerrar antes)
# Pesos da loss por saída. Cada item gera um treino separado.
# Opções: "equal" | "linspace" | "inv_linspace" | CSV (ex: "0.4,0.3,0.3")
LOSS_WEIGHTS_LIST = os.environ.get("LOSS_WEIGHTS_LIST", "equal,linspace").split(",")
SAVE_DIR = os.environ.get("SAVE_DIR", "saved_model")
EXTRA_STRIDE1_BLOCKS = parse_extra_stride1(os.environ.get("EXTRA_STRIDE1", ""))
VARIANT = os.environ.get("VARIANT", variant_tag(EXTRA_STRIDE1_BLOCKS))
TRAIN_SEED = os.environ.get("TRAIN_SEED")
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", data.SEED))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.00005"))
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "cosine")
EXIT_HEAD = os.environ.get("EXIT_HEAD", "bnpool")
PATIENCE = int(os.environ.get("PATIENCE", "0"))
GPU_DATA = os.environ.get("GPU_DATA", "1") == "1"
PRETRAINED = os.environ.get("PRETRAINED", "1") == "1"

# --- Hiperparâmetros fixos (receita v1.1: Adam da v1 + fixes de dados) ---
WARMUP_EPOCHS = 5
BATCH_SIZE = 128
LR_WARMUP = 1e-3
LR_FINETUNE = 1e-4
NUM_CLASSES = 10       # CIFAR-10
INPUT_SIZE = 32        # CIFAR-10
DATA_ROOT = "./data"
# Melhora mínima para contar como progresso. Sem isso, ganhos irrisórios (1e-5)
# resetam a paciência e o early stopping nunca dispara.
MIN_DELTA_LOSS = 1e-4
MIN_DELTA_ACC = 0.05   # pontos percentuais


def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )
    return timestamp


def get_loss_weights(spec, n_exits):
    if spec == "equal":
        return [1.0] * n_exits
    if spec == "linspace":
        return np.linspace(0.15, 1.0, n_exits).tolist()
    if spec == "inv_linspace":
        return np.linspace(1.0, 0.15, n_exits).tolist()
    weights = [float(w) for w in spec.split(",")]
    if len(weights) != n_exits:
        raise ValueError(
            f"loss_weights tem {len(weights)} valores mas modelo tem {n_exits} saídas"
        )
    return weights


def evaluate(model, loader, criterion, weights, device):
    model.eval()
    n_exits = model.n_branches + 1
    correct = torch.zeros(n_exits, device=device)
    exit_losses = torch.zeros(n_exits, device=device)
    total_loss = torch.zeros((), device=device)
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outs = model(images)
            losses = [criterion(o, labels) for o in outs]
            loss = sum(w * l for w, l in zip(weights, losses))
            total_loss += loss * images.size(0)
            for i, (o, l) in enumerate(zip(outs, losses)):
                correct[i] += (o.argmax(1) == labels).sum()
                exit_losses[i] += l * images.size(0)
            total += labels.size(0)
    accs = (100 * correct / total).tolist()
    return total_loss.item() / total, accs, (exit_losses / total).tolist()


def run_training(loss_weights_spec, trainloader, valloader, testloader,
                 n_train, device):
    n_exits = N_BRANCHES + 1
    weights = get_loss_weights(loss_weights_spec, n_exits)
    if LR_SCHEDULE not in ("cosine", "constant"):
        raise ValueError(f"LR_SCHEDULE inválido: {LR_SCHEDULE} (cosine ou constant)")
    if TRAIN_SEED is not None:
        seed = int(TRAIN_SEED)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    logging.info("=" * 60)
    logging.info(f"TREINO: MobileNetV2 + {N_BRANCHES} early exits (n_exits={n_exits})")
    logging.info(f"Estratégia: {loss_weights_spec}")
    logging.info(f"Variante: {VARIANT or 'arquitetura padrão'} | "
                 f"extra_stride1_blocks={EXTRA_STRIDE1_BLOCKS}")
    logging.info(f"Pesos de loss: {[f'{w:.3f}' for w in weights]}")
    logging.info(f"Semente de treino: {TRAIN_SEED}")
    logging.info(f"Semente da divisão dos dados: {SPLIT_SEED}")
    logging.info(f"Receita: Adam, weight decay {WEIGHT_DECAY}, lr {LR_SCHEDULE}, "
                 f"{EPOCHS} épocas, paciência {PATIENCE or 'desligada'}, "
                 f"cabeça {EXIT_HEAD}, pesos do ImageNet: {'sim' if PRETRAINED else 'não'}")
    logging.info("=" * 60)

    model = MobileNetV2EarlyExit(
        n_branches=N_BRANCHES,
        num_classes=NUM_CLASSES,
        input_size=INPUT_SIZE,
        pretrained=PRETRAINED,
        extra_stride1_blocks=EXTRA_STRIDE1_BLOCKS,
        exit_head=EXIT_HEAD,
    ).to(device)
    logging.info(f"Posições dos ramos (índices em features[]): {model.exit_positions}")

    criterion = nn.CrossEntropyLoss()

    logging.info("--> Fase 1: WARMUP (backbone congelado)")
    model.freeze_backbone()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_WARMUP,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = None

    tag = loss_weights_spec.replace(",", "-")
    # Dois checkpoints da MESMA trajetória: um pelo melhor val loss, outro pela
    # melhor val acc. Assim o efeito do critério de seleção fica isolado da
    # variância entre treinos (antes eram runs diferentes).
    selections = {
        "loss": {
            "path": os.path.join(SAVE_DIR,
                                 checkpoint_filename(N_BRANCHES, tag, VARIANT)),
            "best": float("inf"), "delta": MIN_DELTA_LOSS,
        },
        "acc": {
            "path": os.path.join(
                SAVE_DIR,
                checkpoint_filename(N_BRANCHES, tag, acc_variant_tag(VARIANT))),
            "best": float("inf"), "delta": MIN_DELTA_ACC,
        },
    }
    for key, sel in selections.items():
        logging.info(f"Checkpoint [{key}]: {sel['path']}")

    exit_names = [f"S{i+1}" for i in range(n_exits - 1)] + ["Final"]
    history_path = os.path.join(
        SAVE_DIR, "history",
        checkpoint_filename(N_BRANCHES, tag, VARIANT).replace("_best.pth", ".csv"))
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w", newline="") as fh:
        csv.writer(fh).writerow(
            ["epoch", "phase", "lr", "train_loss", "val_loss"]
            + [f"train_loss_{e}" for e in exit_names]
            + [f"val_loss_{e}" for e in exit_names]
            + [f"val_acc_{e}" for e in exit_names])
    logging.info(f"Histórico por época: {history_path}")

    def save_checkpoint(path, key, epoch_done):
        torch.save({
            "state_dict": model.state_dict(),
            "n_branches": N_BRANCHES,
            "exit_positions": model.exit_positions,
            "num_classes": NUM_CLASSES,
            "input_size": INPUT_SIZE,
            "extra_stride1_blocks": model.extra_stride1_blocks,
            "norm_mean": data.IMAGENET_MEAN,
            "norm_std": data.IMAGENET_STD,
            "train_recipe": f"adam-wd{WEIGHT_DECAY:g}-{LR_SCHEDULE}-{EPOCHS}ep",
            "weight_decay": WEIGHT_DECAY,
            "lr_schedule": LR_SCHEDULE,
            "max_epochs": EPOCHS,
            "exit_head": EXIT_HEAD,
            "pretrained": PRETRAINED,
            "select_metric": key,
            "epoch": epoch_done,
            "train_seed": None if TRAIN_SEED is None else int(TRAIN_SEED),
            "split_seed": SPLIT_SEED,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }, path)

    no_improve = 0
    is_finetuning = False

    for epoch in range(EPOCHS):
        if epoch == WARMUP_EPOCHS and not is_finetuning:
            logging.info("*" * 50)
            logging.info(f"Fim do warmup. FINE-TUNING (lr={LR_FINETUNE})")
            logging.info("*" * 50)
            model.unfreeze_all()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE,
                                         weight_decay=WEIGHT_DECAY)
            if LR_SCHEDULE == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
            is_finetuning = True

        model.train()
        running_loss = torch.zeros((), device=device)
        running_exit_losses = torch.zeros(n_exits, device=device)
        lr = optimizer.param_groups[0]["lr"]
        phase = "FineTune" if is_finetuning else "Warmup"
        bar = tqdm(trainloader, desc=f"[{tag}|{phase}] Ep {epoch+1}/{EPOCHS}")
        for images, labels in bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outs = model(images)
            losses = [criterion(o, labels) for o in outs]
            loss = sum(w * l for w, l in zip(weights, losses))
            loss.backward()
            optimizer.step()
            running_loss += loss.detach() * images.size(0)
            running_exit_losses += torch.stack(losses).detach() * images.size(0)
            if not bar.disable:
                bar.set_postfix(loss=loss.item())

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss.item() / n_train
        train_exit_losses = (running_exit_losses / n_train).tolist()
        val_loss, val_accs, val_exit_losses = evaluate(
            model, valloader, criterion, weights, device)

        accs_str = " | ".join(f"S{i+1}: {a:.1f}%" for i, a in enumerate(val_accs[:-1]))
        logging.info(
            f"Ep {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"{accs_str} | Final: {val_accs[-1]:.1f}% | lr: {lr:.2e}"
        )
        with open(history_path, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [epoch + 1, phase, f"{lr:.3e}", f"{train_loss:.5f}", f"{val_loss:.5f}"]
                + [f"{l:.5f}" for l in train_exit_losses]
                + [f"{l:.5f}" for l in val_exit_losses]
                + [f"{a:.3f}" for a in val_accs])

        # Ambos como "menor é melhor": a acurácia entra negada.
        scores = {"loss": val_loss, "acc": -sum(val_accs) / len(val_accs)}
        improved = []
        for key, sel in selections.items():
            if scores[key] < sel["best"] - sel["delta"]:
                sel["best"] = scores[key]
                improved.append(key)
                save_checkpoint(sel["path"], key, epoch + 1)

        if improved:
            no_improve = 0
            marks = ", ".join(
                f"{k}={abs(selections[k]['best']):.4f}" for k in improved
            )
            logging.info(f"   --> SALVO [{'/'.join(improved)}] ({marks})")
        else:
            # Só para quando NENHUM dos dois critérios melhora, para que ambos
            # os checkpoints tenham a mesma chance.
            no_improve += 1
            logging.info(f"   --> Sem melhora. Paciência: {no_improve}/{PATIENCE or '-'}")
            if PATIENCE and no_improve >= PATIENCE:
                logging.info("EARLY STOPPING")
                break

    last_path = os.path.join(SAVE_DIR, checkpoint_filename(
        N_BRANCHES, tag, f"{VARIANT}-last" if VARIANT else "last"))
    save_checkpoint(last_path, "last", epoch + 1)
    logging.info(f"Checkpoint [last]: {last_path}")

    finals = [(key, sel["path"]) for key, sel in selections.items()]
    for key, path in finals + [("last", last_path)]:
        logging.info(f"Avaliando checkpoint [{key}]...")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        test_loss, test_accs, _ = evaluate(model, testloader, criterion, weights, device)
        logging.info("-" * 60)
        logging.info(
            f"RESULTADOS FINAIS (TEST) — {loss_weights_spec} "
            f"[seleção: {key}, época {ckpt['epoch']}]:"
        )
        logging.info(f"Loss global: {test_loss:.4f}")
        for i, a in enumerate(test_accs[:-1]):
            logging.info(f"Acc Saída {i+1}: {a:.2f}%")
        logging.info(f"Acc Final:    {test_accs[-1]:.2f}%")
        logging.info("-" * 60)


def main():
    setup_logging()
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Split compartilhado com calibrate.py (ver src/data.py): treino com
    # augmentation, validação limpa e dividida — aqui usamos só a metade de
    # seleção; a outra metade fica reservada para a calibração.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if GPU_DATA:
        trainloader, valloader, testloader = data.gpu_loaders(
            DATA_ROOT, device, BATCH_SIZE, split_seed=SPLIT_SEED)
        n_train = len(trainloader.labels)
        logging.info("Dados e augmentation na GPU")
    else:
        train_dataset, val_dataset, _val_calib = data.train_val_datasets(
            DATA_ROOT, split_seed=SPLIT_SEED)
        testset = data.test_dataset(DATA_ROOT)
        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        valloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )
        n_train = len(train_dataset)

    for spec in LOSS_WEIGHTS_LIST:
        logging.info("#" * 60)
        logging.info(f"# INICIANDO RUN: loss_weights={spec}")
        logging.info("#" * 60)
        run_training(spec, trainloader, valloader, testloader, n_train, device)


if __name__ == "__main__":
    main()
