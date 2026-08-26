"""Servidor da nuvem.

Recebe o mapa de features intermediário da borda e termina a inferência, ou —
no baseline "só nuvem" — recebe a imagem crua e roda a rede inteira.

Uso:
    CHECKPOINT=saved_model/mobilenet_acc_b3_equal_best.pth \
    uv run python -m src.serving.cloud
"""

import os
import time

import torch
from flask import Flask, jsonify, request

from src.serving.common import (
    Served,
    forward_cloud,
    forward_full,
    transform_image,
    unpack_tensor,
)

app = Flask(__name__)
served = Served()


def _sync():
    if served.device.type == "cuda":
        torch.cuda.synchronize()


@app.route("/cloud/infer", methods=["POST"])
def infer():
    p_tar = float(request.headers["X-P-Tar"])
    nr_branch_edge = int(request.headers["X-Nr-Branch-Edge"])
    raw = request.get_data()

    t0 = time.perf_counter()
    feat = unpack_tensor(raw)
    pred, conf, exit_idx = forward_cloud(served, feat, p_tar, nr_branch_edge)
    _sync()
    compute_ms = (time.perf_counter() - t0) * 1000

    return jsonify({"status": "ok", "pred": pred, "conf": conf,
                    "exit_idx": exit_idx, "cloud_compute_ms": compute_ms,
                    "payload_bytes": len(raw)})


@app.route("/cloud/full", methods=["POST"])
def full():
    """Baseline: a imagem inteira vai para a nuvem, sem saídas antecipadas."""
    raw = request.get_data()
    t0 = time.perf_counter()
    x = transform_image(raw, served.mean, served.std)
    pred, conf = forward_full(served, x)
    _sync()
    compute_ms = (time.perf_counter() - t0) * 1000
    return jsonify({"status": "ok", "pred": pred, "conf": conf,
                    "cloud_compute_ms": compute_ms, "payload_bytes": len(raw)})


@app.route("/cloud/info", methods=["GET"])
def info():
    return jsonify(served.describe())


if __name__ == "__main__":
    app.run(host=os.environ.get("CLOUD_HOST", "127.0.0.1"),
            port=int(os.environ.get("CLOUD_PORT", "3001")),
            threaded=False)
