import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from tqdm import tqdm

from src.model import MobileNetV2EarlyExit


# --- Configuração do experimento (edite aqui ou sobreponha via env vars) ---
# Os defaults rodam manualmente; o run_experiments.sh sobrepõe via env.
N_BRANCHES = int(os.environ.get("N_BRANCHES", "2"))     # nº de saídas antecipadas (1-5)
EPOCHS = int(os.environ.get("EPOCHS", "150"))           # máx. épocas (early stopping pode encerrar antes)
# Pesos da loss por saída. Cada item gera um treino separado.
# Opções: "equal" | "linspace" | "inv_linspace" | CSV (ex: "0.4,0.3,0.3")
LOSS_WEIGHTS_LIST = os.environ.get("LOSS_WEIGHTS_LIST", "equal,linspace").split(",")

# --- Hiperparâmetros fixos ---
WARMUP_EPOCHS = 5
PATIENCE = 30
BATCH_SIZE = 128
LR_WARMUP = 1e-3
LR_FINETUNE = 1e-4
NUM_CLASSES = 10       # CIFAR-10
INPUT_SIZE = 32        # CIFAR-10
SAVE_DIR = "saved_model"
DATA_ROOT = "./data"
SEED = 42              # trava o split train/val p/ reprodutibilidade (usado tb. em calibrate.py)


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
    correct = [0] * n_exits
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outs = model(images)
            losses = [criterion(o, labels) for o in outs]
            loss = sum(w * l for w, l in zip(weights, losses))
            total_loss += loss.item() * images.size(0)
            for i, o in enumerate(outs):
                correct[i] += (o.argmax(1) == labels).sum().item()
            total += labels.size(0)
    accs = [100 * c / total for c in correct]
    return total_loss / total, accs


def run_training(loss_weights_spec, trainloader, valloader, testloader,
                 train_dataset, device):
    n_exits = N_BRANCHES + 1
    weights = get_loss_weights(loss_weights_spec, n_exits)

    logging.info("=" * 60)
    logging.info(f"TREINO: MobileNetV2 + {N_BRANCHES} early exits (n_exits={n_exits})")
    logging.info(f"Estratégia: {loss_weights_spec}")
    logging.info(f"Pesos de loss: {[f'{w:.3f}' for w in weights]}")
    logging.info("=" * 60)

    model = MobileNetV2EarlyExit(
        n_branches=N_BRANCHES,
        num_classes=NUM_CLASSES,
        input_size=INPUT_SIZE,
        pretrained=True,
    ).to(device)
    logging.info(f"Posições dos ramos (índices em features[]): {model.exit_positions}")

    criterion = nn.CrossEntropyLoss()

    logging.info("--> Fase 1: WARMUP (backbone congelado)")
    model.freeze_backbone()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_WARMUP
    )

    tag = loss_weights_spec.replace(",", "-")
    save_path = os.path.join(
        SAVE_DIR, f"mobilenet_b{N_BRANCHES}_{tag}_best.pth"
    )

    best_val_loss = float("inf")
    no_improve = 0
    is_finetuning = False

    for epoch in range(EPOCHS):
        if epoch == WARMUP_EPOCHS and not is_finetuning:
            logging.info("*" * 50)
            logging.info(f"Fim do warmup. FINE-TUNING (lr={LR_FINETUNE})")
            logging.info("*" * 50)
            model.unfreeze_all()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)
            is_finetuning = True

        model.train()
        running_loss = 0.0
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
            running_loss += loss.item() * images.size(0)
            bar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_dataset)
        val_loss, val_accs = evaluate(model, valloader, criterion, weights, device)

        accs_str = " | ".join(f"S{i+1}: {a:.1f}%" for i, a in enumerate(val_accs[:-1]))
        logging.info(
            f"Ep {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"{accs_str} | Final: {val_accs[-1]:.1f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save({
                "state_dict": model.state_dict(),
                "n_branches": N_BRANCHES,
                "exit_positions": model.exit_positions,
                "num_classes": NUM_CLASSES,
                "input_size": INPUT_SIZE,
            }, save_path)
            logging.info(f"   --> SALVO (loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            logging.info(f"   --> Sem melhora. Paciência: {no_improve}/{PATIENCE}")
            if no_improve >= PATIENCE:
                logging.info("EARLY STOPPING")
                break

    logging.info("Carregando melhor modelo para teste final...")
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_loss, test_accs = evaluate(model, testloader, criterion, weights, device)
    logging.info("-" * 60)
    logging.info(f"RESULTADOS FINAIS (TEST) — {loss_weights_spec}:")
    logging.info(f"Loss global: {test_loss:.4f}")
    for i, a in enumerate(test_accs[:-1]):
        logging.info(f"Acc Saída {i+1}: {a:.2f}%")
    logging.info(f"Acc Final:    {test_accs[-1]:.2f}%")
    logging.info("-" * 60)


def main():
    setup_logging()
    os.makedirs(SAVE_DIR, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(INPUT_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    full_trainset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform_train
    )
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    split_gen = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(
        full_trainset, [train_size, val_size], generator=split_gen
    )
    testset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform_test
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    valloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for spec in LOSS_WEIGHTS_LIST:
        logging.info("#" * 60)
        logging.info(f"# INICIANDO RUN: loss_weights={spec}")
        logging.info("#" * 60)
        run_training(spec, trainloader, valloader, testloader, train_dataset, device)


if __name__ == "__main__":
    main()
