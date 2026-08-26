"""Datasets CIFAR-10 compartilhados por train/calibrate/inference.

Centraliza duas decisões que antes estavam duplicadas (e divergentes) nos
scripts:

1. **Split train/val por índices, com transforms distintos.** O bug da versão
   anterior: `random_split` sobre um único dataset fazia a validação herdar o
   augmentation de treino (RandomCrop/Flip), então early stopping e seleção de
   checkpoint rodavam sobre imagens aumentadas — e a calibração usava as mesmas
   amostras SEM augmentation. Aqui o split é feito por permutação de índices
   (mesma semântica/SEED do `random_split` antigo) sobre duas instâncias do
   dataset: treino com augmentation, validação com transform de avaliação.

2. **Normalização ImageNet.** O backbone é pré-treinado em ImageNet; usar
   mean/std do ImageNet preserva a distribuição de entrada que os pesos
   esperam. Checkpoints antigos (normalização 0.5) continuam avaliáveis via
   `norm_from_ckpt`, que lê as estatísticas salvas no checkpoint e cai no
   legado quando ausentes.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Checkpoints treinados antes da receita v2 usavam normalização 0.5.
LEGACY_MEAN = (0.5, 0.5, 0.5)
LEGACY_STD = (0.5, 0.5, 0.5)

SEED = 42          # trava o split train/val (compartilhado com calibrate.py)
TRAIN_FRACTION = 0.8
# A validação é dividida em duas metades DISJUNTAS: uma para early stopping e
# seleção de checkpoint, outra para ajustar as temperaturas. Sem isso o mesmo
# conjunto seria usado três vezes (parada, seleção e calibração), e a
# temperatura seria ajustada sobre os dados que escolheram o próprio modelo.
VAL_SELECT_FRACTION = 0.5
INPUT_SIZE = 32


def transform_train(mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return transforms.Compose([
        transforms.RandomCrop(INPUT_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def transform_eval(mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def train_val_datasets(root, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """(treino com augmentation, val_select limpa, val_calib limpa).

    A partição treino/validação é a mesma de sempre (mesmo SEED e mesmas 10k
    amostras de validação); o que muda é que a validação vem dividida em duas
    metades disjuntas — `val_select` para early stopping/seleção de checkpoint
    e `val_calib` para o temperature scaling.
    """
    aug = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train(mean, std)
    )
    clean = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_eval(mean, std)
    )
    gen = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(aug), generator=gen).tolist()
    n_train = int(TRAIN_FRACTION * len(aug))
    val_idx = perm[n_train:]
    n_select = int(VAL_SELECT_FRACTION * len(val_idx))
    return (
        Subset(aug, perm[:n_train]),
        Subset(clean, val_idx[:n_select]),
        Subset(clean, val_idx[n_select:]),
    )


def test_dataset(root, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_eval(mean, std)
    )


def norm_from_ckpt(ckpt):
    """Estatísticas de normalização com que o checkpoint foi treinado."""
    mean = ckpt.get("norm_mean", LEGACY_MEAN)
    std = ckpt.get("norm_std", LEGACY_STD)
    return tuple(mean), tuple(std)
