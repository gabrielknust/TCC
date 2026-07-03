import copy

import torch
import torch.nn as nn
import torchvision.models as models
from thop import profile


class ExitHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden=128, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class MobileNetV2EarlyExit(nn.Module):
    """MobileNetV2 com N saídas antecipadas posicionadas por distribuição de FLOPs.

    Args:
        n_branches: número de saídas antecipadas (não conta a final).
        exit_positions: índices em features[] onde cada ramo é inserido.
            Cada índice é o último layer da stage que precede o ramo.
            Se None, usa distribuição linear em FLOPs cumulativos.
        num_classes: nº de classes do dataset.
        input_size: lado da imagem de entrada (ex.: 32 para CIFAR).
        pretrained: se True, carrega pesos ImageNet.
        head_dropout: dropout dos exit heads.
    """

    def __init__(
        self,
        n_branches=2,
        exit_positions=None,
        num_classes=10,
        input_size=32,
        pretrained=True,
        head_dropout=0.5,
    ):
        super().__init__()
        self.n_branches = n_branches
        self.num_classes = num_classes
        self.input_size = input_size

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        # Para entradas pequenas (CIFAR 32x32), trocamos stride do primeiro conv para 1
        # e mantemos os pesos pré-treinados das demais camadas.
        if input_size <= 64:
            backbone.features[0][0] = nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3,
                stride=1, padding=1, bias=False,
            )

        features_list = list(backbone.features.children())
        self.last_channel = backbone.last_channel

        if exit_positions is None:
            exit_positions = self._linear_flop_positions(
                n_branches, input_size, features_list
            )
        if len(exit_positions) != n_branches:
            raise ValueError(
                f"exit_positions deve ter {n_branches} índices, recebeu {exit_positions}"
            )
        self.exit_positions = sorted(set(exit_positions))
        if len(self.exit_positions) != n_branches:
            raise ValueError(f"exit_positions com duplicatas: {exit_positions}")
        if max(self.exit_positions) >= len(features_list) - 1:
            raise ValueError(
                "exit_positions não pode incluir a última camada (sobraria stage final vazia)"
            )

        # Constrói stages: cada stage agrupa as camadas até (e incluindo) a posição do ramo.
        # A última stage contém todas as camadas após o último ramo.
        self.stages = nn.ModuleList()
        prev = 0
        for pos in self.exit_positions:
            self.stages.append(nn.Sequential(*features_list[prev:pos + 1]))
            prev = pos + 1
        self.stages.append(nn.Sequential(*features_list[prev:]))

        # Constrói heads inferindo nº de canais via dummy input.
        self.exits = nn.ModuleList()
        with torch.no_grad():
            self.eval()
            x = torch.zeros(1, 3, input_size, input_size)
            for i in range(n_branches):
                x = self.stages[i](x)
                ch = x.shape[1]
                self.exits.append(
                    ExitHead(ch, num_classes, hidden=128, dropout=head_dropout)
                )
            self.train()

        self.final_classifier = nn.Linear(self.last_channel, num_classes)

    def _linear_flop_positions(self, n_branches, input_size, features_list):
        """Distribui ramos linearmente nos FLOPs cumulativos do backbone.

        Nota: thop.profile adiciona buffers (total_ops, total_params) nos módulos
        perfilados. Para não poluir os módulos reais (que seriam salvos no
        state_dict), perfilamos sobre uma cópia descartável.
        """
        x = torch.zeros(1, 3, input_size, input_size)
        cum = []
        total = 0.0
        copied = [copy.deepcopy(layer).eval() for layer in features_list]
        with torch.no_grad():
            for layer in copied:
                f, _ = profile(layer, inputs=(x,), verbose=False)
                x = layer(x)
                total += float(f)
                cum.append(total)

        positions = []
        n_layers = len(features_list)
        for i in range(1, n_branches + 1):
            target = total * i / (n_branches + 1)
            for idx, c in enumerate(cum):
                if c >= target and idx < n_layers - 1 and idx not in positions:
                    positions.append(idx)
                    break
        if len(positions) != n_branches:
            raise RuntimeError(
                f"Não foi possível posicionar {n_branches} ramos (obteve {positions})"
            )
        return positions

    def freeze_backbone(self):
        """Congela tudo exceto o primeiro conv (que foi modificado) e os heads."""
        for stage in self.stages:
            for p in stage.parameters():
                p.requires_grad = False
        for p in self.final_classifier.parameters():
            p.requires_grad = False
        # Descongela o primeiro conv (perdeu pesos pré-treinados na adaptação).
        for p in self.stages[0][0].parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def _final_logits(self, x):
        x = self.stages[-1](x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.final_classifier(x)

    def forward(self, x):
        """Retorna lista [logits_exit1, ..., logits_exitN, logits_final]."""
        outputs = []
        for i in range(self.n_branches):
            x = self.stages[i](x)
            outputs.append(self.exits[i](x))
        outputs.append(self._final_logits(x))
        return outputs

    @torch.no_grad()
    def forward_cascade(self, x, p_tar):
        """Inferência por amostra com threshold (simula edge→cloud).

        Espera batch_size=1. Retorna (pred_class, conf, exit_idx)
        onde exit_idx ∈ [0, n_branches] (n_branches = saída final).
        """
        for i in range(self.n_branches):
            x = self.stages[i](x)
            logits = self.exits[i](x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            if conf.item() >= p_tar:
                return pred.item(), conf.item(), i
        logits = self._final_logits(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return pred.item(), conf.item(), self.n_branches
