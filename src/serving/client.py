"""Nó final: envia imagens à borda e mede latência real fim-a-fim.

Varre `p_tar` × ponto de partição (`nr_branch_edge`) e grava um CSV por amostra
com a decomposição de tempo e o tamanho do payload. Roda também o baseline
"só nuvem" (imagem inteira, rede completa, sem saídas antecipadas).

Uso (com os servidores já no ar):
    uv run python -m src.serving.client
"""

import io
import os
import time
from datetime import datetime

import pandas as pd
import requests
import torchvision

from src import data

EDGE_URL = os.environ.get("EDGE_URL", "http://127.0.0.1:5001")
CLOUD_URL = os.environ.get("CLOUD_URL", "http://127.0.0.1:3001")
N_IMAGES = int(os.environ.get("N_IMAGES", "200"))
WARMUP = int(os.environ.get("WARMUP", "20"))
P_TAR_LIST = [float(p) for p in
              os.environ.get("P_TAR_LIST", "0.5,0.7,0.8,0.9,0.95,0.99").split(",")]
NR_BRANCH_EDGE_LIST = [int(v) for v in
                       os.environ.get("NR_BRANCH_EDGE_LIST", "").split(",") if v]
DATA_ROOT = "./data"


def load_images(n):
    """Imagens do test set como PNG em memória (é o que um cliente enviaria)."""
    ds = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True)
    out = []
    for i in range(n):
        img, label = ds[i]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append((buf.getvalue(), label))
    return out


def post(url, payload, headers=None):
    r = requests.post(url, data=payload, timeout=60,
                      headers={"Content-Type": "application/octet-stream",
                               **(headers or {})})
    r.raise_for_status()
    return r.json()


def main():
    info = requests.get(f"{EDGE_URL}/edge/info", timeout=30).json()
    print(f"Borda:  {info['checkpoint']} | calib={info['calib_mode']} | "
          f"{info['n_branches']} ramos | device={info['device']}")
    n_branches = info["n_branches"]
    partitions = NR_BRANCH_EDGE_LIST or list(range(1, n_branches + 1))

    images = load_images(N_IMAGES + WARMUP)
    warm, test = images[:WARMUP], images[WARMUP:]

    print(f"Aquecendo ({WARMUP} imagens)...")
    for img, _ in warm:
        post(f"{EDGE_URL}/edge/infer", img,
             {"X-P-Tar": "0.9", "X-Nr-Branch-Edge": str(partitions[-1])})
        post(f"{CLOUD_URL}/cloud/full", img)

    rows = []
    for nr_edge in partitions:
        for p_tar in P_TAR_LIST:
            t0 = time.perf_counter()
            for img, label in test:
                r = post(f"{EDGE_URL}/edge/infer", img,
                         {"X-P-Tar": str(p_tar), "X-Nr-Branch-Edge": str(nr_edge)})
                rows.append({"mode": "early_exit", "p_tar": p_tar,
                             "nr_branch_edge": nr_edge, "label": label,
                             "correct": int(r["pred"] == label), **{
                                 k: r[k] for k in
                                 ("terminated_at_edge", "exit_idx", "conf",
                                  "preprocess_ms", "edge_compute_ms",
                                  "cloud_roundtrip_ms", "cloud_compute_ms",
                                  "transmission_ms", "link_ms", "payload_bytes",
                                  "total_ms")}})
            dt = time.perf_counter() - t0
            sub = [r for r in rows if r["mode"] == "early_exit"
                   and r["p_tar"] == p_tar and r["nr_branch_edge"] == nr_edge]
            edge_frac = 100 * sum(r["terminated_at_edge"] for r in sub) / len(sub)
            lat = sum(r["total_ms"] for r in sub) / len(sub)
            acc = 100 * sum(r["correct"] for r in sub) / len(sub)
            print(f"  nr_edge={nr_edge} p_tar={p_tar:.2f}: {lat:6.2f} ms/img | "
                  f"borda {edge_frac:5.1f}% | acc {acc:5.1f}% | {dt:.1f}s")

    print("Baseline só-nuvem...")
    # O baseline paga o mesmo enlace, transmitindo a imagem inteira.
    mbps, rtt = info.get("link_mbps", 0), info.get("link_rtt_ms", 0)
    for img, label in test:
        t0 = time.perf_counter()
        if mbps or rtt:
            time.sleep((rtt + (len(img) * 8 / (mbps * 1e6) * 1000 if mbps else 0)) / 1000)
        r = post(f"{CLOUD_URL}/cloud/full", img)
        total = (time.perf_counter() - t0) * 1000
        rows.append({"mode": "only_cloud", "p_tar": float("nan"),
                     "nr_branch_edge": 0, "label": label,
                     "correct": int(r["pred"] == label),
                     "terminated_at_edge": False, "exit_idx": -1,
                     "conf": r["conf"], "preprocess_ms": 0.0,
                     "edge_compute_ms": 0.0, "cloud_roundtrip_ms": total,
                     "cloud_compute_ms": r["cloud_compute_ms"],
                     "transmission_ms": total - r["cloud_compute_ms"],
                     "link_ms": 0.0,
                     "payload_bytes": r["payload_bytes"], "total_ms": total})

    df = pd.DataFrame(rows)
    os.makedirs("results/latency", exist_ok=True)
    tag = os.path.splitext(info["checkpoint"])[0]
    link = ""
    if info.get("link_mbps"):
        link = f"_link{info['link_mbps']:g}mbps-{info['link_rtt_ms']:g}ms"
    out = (f"results/latency/latency_{tag}{link}_"
           f"{datetime.now():%Y-%m-%d_%H-%M-%S}.csv")
    df.to_csv(out, index=False)

    base = df[df["mode"] == "only_cloud"]["total_ms"].mean()
    print(f"\nsó-nuvem: {base:.2f} ms/img")
    print(f"Salvo em: {out}  ({len(df)} medições)")


if __name__ == "__main__":
    main()
