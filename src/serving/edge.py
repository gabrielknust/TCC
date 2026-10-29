"""Servidor da borda.

Recebe a imagem, roda os `nr_branch_edge` primeiros ramos e decide: classifica
localmente ou envia o mapa de features para a nuvem. Devolve ao cliente a
decomposição do tempo — pré-processamento, computação na borda, ida e volta à
nuvem — e o tamanho do payload transmitido.

Uso:
    CHECKPOINT=saved_model/mobilenet_acc_b3_equal_best.pth \
    uv run python -m src.serving.edge
"""

import os
import time

import requests
import torch
from flask import Flask, jsonify, request

from src.serving.common import Served, forward_edge, pack_tensor, transform_image

app = Flask(__name__)
served = Served()
CLOUD_URL = os.environ.get("CLOUD_URL", "http://127.0.0.1:3001")
TIMEOUT = float(os.environ.get("SERVING_TIMEOUT", "30"))
# Com borda e nuvem no mesmo host, a rede e' irrealisticamente rapida e o custo
# de transmitir o mapa de features desaparece. LINK_MBPS/LINK_RTT_MS injetam um
# atraso equivalente a um enlace real (ex.: 4G ~10 Mbps, RTT 50 ms).
LINK_MBPS = float(os.environ.get("LINK_MBPS", "0") or 0)
LINK_RTT_MS = float(os.environ.get("LINK_RTT_MS", "0") or 0)


def simulate_link(n_bytes):
    """Atraso do enlace para transmitir n_bytes (ida) mais o RTT."""
    if LINK_MBPS <= 0 and LINK_RTT_MS <= 0:
        return 0.0
    delay_ms = LINK_RTT_MS
    if LINK_MBPS > 0:
        delay_ms += n_bytes * 8 / (LINK_MBPS * 1e6) * 1000
    time.sleep(delay_ms / 1000)
    return delay_ms


def _sync():
    if served.device.type == "cuda":
        torch.cuda.synchronize()


@app.route("/edge/infer", methods=["POST"])
def infer():
    p_tar = float(request.headers["X-P-Tar"])
    nr_branch_edge = int(request.headers["X-Nr-Branch-Edge"])
    raw = request.get_data()

    t_start = time.perf_counter()
    x = transform_image(raw, served.mean, served.std)
    _sync()
    t_pre = time.perf_counter()

    terminated, pred, conf, exit_idx, feat = forward_edge(
        served, x, p_tar, nr_branch_edge)
    _sync()
    t_edge = time.perf_counter()

    result = {
        "status": "ok",
        "terminated_at_edge": terminated,
        "preprocess_ms": (t_pre - t_start) * 1000,
        "edge_compute_ms": (t_edge - t_pre) * 1000,
        "cloud_roundtrip_ms": 0.0,
        "cloud_compute_ms": 0.0,
        "transmission_ms": 0.0,
        "link_ms": 0.0,
        "payload_bytes": 0,
    }

    if terminated:
        result.update(pred=pred, conf=conf, exit_idx=exit_idx)
    else:
        payload = pack_tensor(feat)
        t_send = time.perf_counter()
        link_ms = simulate_link(len(payload))
        r = requests.post(
            f"{CLOUD_URL}/cloud/infer", data=payload, timeout=TIMEOUT,
            headers={"Content-Type": "application/octet-stream",
                     "X-P-Tar": str(p_tar),
                     "X-Nr-Branch-Edge": str(nr_branch_edge),
                     "X-Best-Conf": str(conf),
                     "X-Best-Pred": str(pred)},
        )
        r.raise_for_status()
        cloud = r.json()
        roundtrip_ms = (time.perf_counter() - t_send) * 1000
        result.update(
            pred=cloud["pred"], conf=cloud["conf"], exit_idx=cloud["exit_idx"],
            cloud_roundtrip_ms=roundtrip_ms,
            cloud_compute_ms=cloud["cloud_compute_ms"],
            # O que sobra do ida-e-volta depois de descontar a computação da
            # nuvem: serialização, rede e desserialização.
            transmission_ms=roundtrip_ms - cloud["cloud_compute_ms"],
            link_ms=link_ms,
            payload_bytes=len(payload),
        )

    result["total_ms"] = (time.perf_counter() - t_start) * 1000
    return jsonify(result)


@app.route("/edge/info", methods=["GET"])
def info():
    return jsonify({**served.describe(), "cloud_url": CLOUD_URL,
                    "link_mbps": LINK_MBPS, "link_rtt_ms": LINK_RTT_MS})


if __name__ == "__main__":
    app.run(host=os.environ.get("EDGE_HOST", "127.0.0.1"),
            port=int(os.environ.get("EDGE_PORT", "5001")),
            threaded=False)
