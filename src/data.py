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

import math

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


def train_val_datasets(root, mean=IMAGENET_MEAN, std=IMAGENET_STD, split_seed=SEED):
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
    gen = torch.Generator().manual_seed(split_seed)
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


def random_crop_flip(x, padding=4):
    b, c, h, w = x.shape
    padded = torch.nn.functional.pad(x, (padding,) * 4)
    oy = torch.randint(0, 2 * padding + 1, (b, 1), device=x.device)
    ox = torch.randint(0, 2 * padding + 1, (b, 1), device=x.device)
    rows = (oy + torch.arange(h, device=x.device)).view(b, 1, h, 1)
    x = padded.gather(2, rows.expand(b, c, h, w + 2 * padding))
    cols = (ox + torch.arange(w, device=x.device)).view(b, 1, 1, w)
    x = x.gather(3, cols.expand(b, c, h, w))
    flip = (torch.rand(b, device=x.device) < 0.5).view(b, 1, 1, 1)
    return torch.where(flip, x.flip(3), x)


class GpuLoader:
    def __init__(self, images, labels, batch_size, mean, std, augment=False,
                 shuffle=False):
        self.images, self.labels = images, labels
        self.batch_size = batch_size
        self.mean = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
        self.augment, self.shuffle = augment, shuffle

    def __len__(self):
        return math.ceil(len(self.labels) / self.batch_size)

    def __iter__(self):
        n = len(self.labels)
        order = (torch.randperm(n) if self.shuffle else torch.arange(n)).to(
            self.images.device)
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            x = self.images[idx].float().div_(255)
            if self.augment:
                x = random_crop_flip(x)
            yield (x - self.mean) / self.std, self.labels[idx]


def gpu_loaders(root, device, batch_size, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                split_seed=SEED):
    train = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test = torchvision.datasets.CIFAR10(root=root, train=False, download=True)

    def tensors(ds, idx=None):
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
        y = torch.tensor(ds.targets)
        if idx is not None:
            x, y = x[idx], y[idx]
        return x.to(device), y.to(device)

    gen = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(train), generator=gen)
    n_train = int(TRAIN_FRACTION * len(train))
    val_idx = perm[n_train:]
    n_select = int(VAL_SELECT_FRACTION * len(val_idx))
    return (
        GpuLoader(*tensors(train, perm[:n_train]), batch_size, mean, std,
                  augment=True, shuffle=True),
        GpuLoader(*tensors(train, val_idx[:n_select]), batch_size, mean, std),
        GpuLoader(*tensors(test), batch_size, mean, std),
    )


def split_seed_from_ckpt(ckpt):
    return int(ckpt.get("split_seed", SEED))


def norm_from_ckpt(ckpt):
    """Estatísticas de normalização com que o checkpoint foi treinado."""
    mean = ckpt.get("norm_mean", LEGACY_MEAN)
    std = ckpt.get("norm_std", LEGACY_STD)
    return tuple(mean), tuple(std)
