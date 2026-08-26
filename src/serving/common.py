"""Peças compartilhadas pelos servidores de borda e nuvem.

O modelo é o mesmo dos experimentos offline (`MobileNetV2EarlyExit`), carregado
do checkpoint com `src.inference.load_model` — inclusive a variante de stride e
as temperaturas, que vêm gravadas no próprio arquivo.

Serialização: o mapa de features intermediário vai como bytes de um `.npy`
(float16), não como lista JSON. A diferença importa — JSON de floats infla o
payload em ~10x, e o custo de transmissão é exatamente uma das grandezas que
este experimento existe para medir.
"""

import io
import os

import numpy as np
import torch

from src import data
from src.inference import load_model, resolve_temperatures

CHECKPOINT = os.environ.get("CHECKPOINT",
                            "saved_model/mobilenet_acc_b3_equal_best.pth")
CALIB_MODE = os.environ.get("CALIB_MODE", "overall")
DEVICE = torch.device(os.environ.get("SERVING_DEVICE",
                                     "cuda" if torch.cuda.is_available() else "cpu"))


class Served:
    """Modelo carregado uma vez por processo."""

    def __init__(self, ckpt_path=CHECKPOINT, calib_mode=CALIB_MODE, device=DEVICE):
        self.model, self.ckpt = load_model(ckpt_path, device)
        self.model.eval()
        self.device = device
        self.n_branches = self.model.n_branches
        self.n_exits = self.n_branches + 1
        self.temperatures = resolve_temperatures(self.ckpt, self.n_exits, calib_mode)
        self.mean, self.std = data.norm_from_ckpt(self.ckpt)
        self.calib_mode = calib_mode
        self.name = os.path.basename(ckpt_path)

    def describe(self):
        return {"checkpoint": self.name, "calib_mode": self.calib_mode,
                "n_branches": self.n_branches, "device": str(self.device),
                "exit_positions": list(self.model.exit_positions)}


def transform_image(image_bytes, mean, std):
    """PIL → tensor normalizado [1,3,32,32]."""
    from PIL import Image
    import torchvision.transforms as T
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tf = T.Compose([
        T.Resize((data.INPUT_SIZE, data.INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return tf(img).unsqueeze(0)


def pack_tensor(t):
    buf = io.BytesIO()
    np.save(buf, t.detach().cpu().numpy().astype(np.float16), allow_pickle=False)
    return buf.getvalue()


def unpack_tensor(raw):
    return torch.from_numpy(np.load(io.BytesIO(raw), allow_pickle=False)).float()


def _confidence(served, logits, exit_idx):
    probs = torch.softmax(logits / served.temperatures[exit_idx], dim=1)
    conf, pred = probs.max(dim=1)
    return conf.item(), int(pred.item())


@torch.no_grad()
def forward_edge(served, x, p_tar, nr_branch_edge):
    """Roda os primeiros `nr_branch_edge` ramos na borda.

    Devolve (terminou_na_borda, classe, confiança, indice_da_saida, feature_map).
    O feature_map só é relevante quando não terminou: é o que vai para a nuvem.
    """
    x = x.to(served.device)
    for i in range(min(nr_branch_edge, served.n_branches)):
        x = served.model.stages[i](x)
        conf, pred = _confidence(served, served.model.exits[i](x), i)
        if conf >= p_tar:
            return True, pred, conf, i, None
    return False, None, None, None, x


@torch.no_grad()
def forward_cloud(served, x, p_tar, nr_branch_edge):
    """Continua a inferência a partir do ponto de partição."""
    x = x.to(served.device)
    for i in range(nr_branch_edge, served.n_branches):
        x = served.model.stages[i](x)
        conf, pred = _confidence(served, served.model.exits[i](x), i)
        if conf >= p_tar:
            return pred, conf, i
    logits = served.model._final_logits(x)
    conf, pred = _confidence(served, logits, served.n_exits - 1)
    return pred, conf, served.n_exits - 1


@torch.no_grad()
def forward_full(served, x):
    """Baseline "só nuvem": backbone inteiro, sem avaliar as saídas antecipadas."""
    x = x.to(served.device)
    for i in range(served.n_branches):
        x = served.model.stages[i](x)
    # _final_logits ja' aplica a ultima stage, o pooling e o classificador.
    conf, pred = _confidence(served, served.model._final_logits(x),
                             served.n_exits - 1)
    return pred, conf
