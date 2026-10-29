import csv
import io
import os

import numpy as np
import torch
import torchvision

from src.model import MobileNetV2EarlyExit

ARCHS = [
    ("b0", 0, ()), ("b2", 2, ()), ("b3", 3, ()), ("b4", 4, ()), ("b5", 5, ()),
    ("v3b_b3", 3, (7, 14)),
]
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
OUT = os.path.join(RESULTS_DIR, "model_profile.csv")
PAYLOAD_OUT = os.path.join(RESULTS_DIR, "payload.csv")


def count(module):
    return sum(p.numel() for p in module.parameters())


def largest_activation(model):
    sizes = []
    hooks = [m.register_forward_hook(lambda _m, _i, o: sizes.append(o.numel()))
             for m in model.modules() if not list(m.children())]
    with torch.no_grad():
        model(torch.zeros(1, 3, model.input_size, model.input_size))
    for h in hooks:
        h.remove()
    return max(sizes)


def profile(name, n, extra):
    model = MobileNetV2EarlyExit(n_branches=n, pretrained=False,
                                 extra_stride1_blocks=extra).eval()
    macs = [m / 1e6 for m in model.profile_exit_macs()]
    params, acc = [], 0
    for i in range(n):
        acc += count(model.stages[i]) + count(model.exits[i])
        params.append(acc)
    params.append(count(model))
    return {
        "arquitetura": name,
        "posicoes": " ".join(str(p) for p in model.exit_positions),
        "macs_acum_M": " / ".join(f"{m:.1f}" for m in macs),
        "params_acum_M": " / ".join(f"{p / 1e6:.2f}" for p in params),
        "params_total_M": round(count(model) / 1e6, 3),
        "params_cabecas_k": round(sum(count(e) for e in model.exits) / 1e3, 1),
        "pesos_fp32_MB": round(count(model) * 4 / 2**20, 2),
        "maior_ativacao_KB": round(largest_activation(model) * 4 / 1024, 1),
    }


@torch.no_grad()
def payload(name, n, extra):
    model = MobileNetV2EarlyExit(n_branches=n, pretrained=False,
                                 extra_stride1_blocks=extra).eval()
    x = torch.zeros(1, 3, model.input_size, model.input_size)
    rows = []
    for i in range(n):
        x = model.stages[i](x)
        rows.append({"arquitetura": name, "saida": i + 1,
                     "bloco": model.exit_positions[i],
                     "formato": "x".join(str(d) for d in x.shape[1:]),
                     "fp32_KB": x.numel() * 4 / 1024, "fp16_KB": x.numel() * 2 / 1024,
                     "int8_KB": x.numel() / 1024})
    return rows


def image_sizes(root="./data", n=1000):
    ds = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    sizes = []
    for i in range(n):
        buf = io.BytesIO()
        ds[i][0].save(buf, format="PNG")
        sizes.append(len(buf.getvalue()))
    return 32 * 32 * 3 / 1024, float(np.mean(sizes)) / 1024


def main():
    raw_kb, png_kb = image_sizes()
    pay = [r for a in ARCHS for r in payload(*a)]
    for r in pay:
        r["imagem_crua_KB"], r["imagem_png_KB"] = raw_kb, png_kb
    with open(PAYLOAD_OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pay[0].keys()))
        w.writeheader()
        w.writerows({k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()}
                    for r in pay)
    print(f"{len(pay)} linhas em {PAYLOAD_OUT} (imagem crua {raw_kb:.1f} KB, PNG {png_kb:.2f} KB)")
    rows = [profile(*a) for a in ARCHS]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
