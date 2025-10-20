#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
from typing import Dict, Tuple, List
import numpy as np

import flwr as fl
from ultralytics import YOLO


# -------------------------
# Utilidades
# -------------------------
def normpath(p: str) -> str:
    return os.path.normpath(p.replace("\\", "/"))

def robust_count_images(data_yaml: str, split: str) -> int:
    from ultralytics.data.utils import check_det_dataset
    data_yaml = normpath(data_yaml)
    cfg = check_det_dataset(data_yaml)
    key = "train" if split == "train" else "val" if split == "val" else "test"
    if key not in cfg:
        return 0
    base = cfg.get("path", "")
    imgdir = cfg[key]
    if base and not os.path.isabs(imgdir):
        imgdir = os.path.join(base, imgdir)
    imgdir = normpath(imgdir)
    if not os.path.isdir(imgdir):
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    try:
        return sum(1 for f in os.listdir(imgdir) if os.path.splitext(f)[1].lower() in exts)
    except Exception:
        return 0

def canonical_names(model_path: str) -> List[str]:
    y = YOLO(model_path)
    return [k for k, v in y.model.state_dict().items() if v.requires_grad]

def state_from_model(model: YOLO) -> Dict[str, np.ndarray]:
    return {k: p.detach().cpu().numpy().copy() for k, p in model.model.named_parameters()}

def set_state_by_name_(model: YOLO, key_order: List[str], ndarrays: List[np.ndarray]) -> None:
    import torch
    name_to_param = dict(model.model.named_parameters())
    with torch.inference_mode():
        for k, arr in zip(key_order, ndarrays):
            p = name_to_param.get(k, None)
            if p is None:
                print(f"[WARN][CLIENT] Parâmetro não encontrado: {k} (ignorado)")
                continue
            if tuple(p.shape) != tuple(arr.shape):
                print(f"[WARN][CLIENT] Shape mismatch {tuple(arr.shape)} != {tuple(p.shape)} em '{k}' (ignorado)")
                continue
            t = torch.from_numpy(arr).to(p.device).to(p.dtype)
            p.copy_(t)

def count_bn_layers(model) -> int:
    import torch.nn as nn
    n = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            n += 1
    return n


# -------------------------
# Cliente Flower
# -------------------------
class YoloFlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, model_path: str, data_yaml: str, device: str = "cpu"):
        self.cid = cid
        self.device = device
        self.data_yaml = normpath(data_yaml)

        # Carrega o .pt base uma única vez
        self.model_path = normpath(model_path)
        self.yolo = YOLO(self.model_path)
        self.yolo.to(self.device)

        # Ordem canônica estável
        self.key_order = canonical_names(self.model_path)

        # Impede simplificações e re-loads (mantemos o objeto do modelo)
        self.yolo.overrides["model"] = self.yolo.model
        self.yolo.overrides["pretrained"] = False
        self.yolo.overrides["resume"] = False
        self.yolo.overrides["simplify"] = False

        # Bloqueia FUSÃO de camadas (Conv+BN) na própria instância do modelo
        self._disable_layer_fusion()

        # Log útil: quantidade de BN (se zerar em algum round, sabemos que fundiu)
        print(f"[INFO][CLIENT] CID={self.cid} BN_layers={count_bn_layers(self.yolo.model)}")

    # Monkey-patch para impedir model.fuse() de alterar a arquitetura
    def _disable_layer_fusion(self):
        m = self.yolo.model
        if hasattr(m, "fuse"):
            def _noop_fuse(*args, **kwargs):
                # Não faz nada; retorna o próprio modelo
                return m
            m.fuse = _noop_fuse

    def _reinforce_overrides(self):
        # Reforça antes de train/val
        self.yolo.overrides["model"] = self.yolo.model
        self.yolo.overrides["pretrained"] = False
        self.yolo.overrides["resume"] = False
        self.yolo.overrides["simplify"] = False
        # Reaplica o monkey-patch (por segurança)
        self._disable_layer_fusion()

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        state = state_from_model(self.yolo)
        return [state[k] for k in self.key_order]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        set_state_by_name_(self.yolo, self.key_order, parameters)
        # Checagem rápida após aplicar pesos
        print(f"[INFO][CLIENT] CID={self.cid} BN_layers={count_bn_layers(self.yolo.model)} (após set_parameters)")

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        # aplica pesos globais
        self.set_parameters(parameters)

        # hyperparams
        epochs   = int(config.get("local_epochs", config.get("epochs", 1)))
        batch    = int(config.get("batch", 2))
        imgsz    = int(config.get("imgsz", 416))
        workers  = int(config.get("workers", 0))
        lr       = float(config.get("learning_rate", config.get("lr", 5e-4)))
        fraction = float(config.get("data_fraction", config.get("fraction", 0.25)))
        warmup   = float(config.get("warmup_epochs", config.get("warmup", 0.0)))
        do_val   = bool(config.get("val", False))

        self._reinforce_overrides()

        n_train = robust_count_images(self.data_yaml, "train")
        if n_train <= 0:
            n_train = 1

        t0 = time.time()
        _ = self.yolo.train(
            model=self.yolo.model,      # usa objeto em memória (evita reload/fuse)
            data=self.data_yaml,
            device=self.device,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            workers=workers,
            lr0=lr,
            cache=False,
            amp=False,
            val=do_val,
            plots=False,
            verbose=False,
            fraction=fraction,
            warmup_epochs=warmup,
            name=f"fl_c{self.cid}_r{int(config.get('server_round', 0))}",
            pretrained=False,
            resume=False,
            simplify=False,
        )
        t1 = time.time()

        params = self.get_parameters({})
        metrics = {"time_train_s": float(t1 - t0)}
        return params, n_train, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        # aplica pesos
        self.set_parameters(parameters)

        round_id     = int(config.get("server_round", -1))
        eval_imgsz   = int(config.get("imgsz", 640))
        eval_workers = int(config.get("workers", 0))
        split        = str(config.get("split", "val")).lower()

        self._reinforce_overrides()

        results = self.yolo.val(
            model=self.yolo.model,      # usa objeto em memória (evita reload/fuse)
            data=self.data_yaml,
            device=self.device,
            workers=eval_workers,
            imgsz=eval_imgsz,
            split=split,
            plots=False,
            verbose=False,
        )

        import numpy as _np
        P = R = F1 = 0.0
        map50 = map5095 = 0.0

        box = getattr(results, "box", None) or getattr(results, "boxes", None)
        try:
            if box is not None:
                p_arr  = getattr(box, "p", None)
                r_arr  = getattr(box, "r", None)
                f1_arr = getattr(box, "f1", None)
                if p_arr is not None: P = float(_np.asarray(p_arr, dtype=float).mean())
                if r_arr is not None: R = float(_np.asarray(r_arr, dtype=float).mean())
                if f1_arr is not None: F1 = float(_np.asarray(f1_arr, dtype=float).mean())
                if hasattr(box, "map50"): map50 = float(getattr(box, "map50"))
                if hasattr(box, "map"):   map5095 = float(getattr(box, "map"))
        except Exception:
            pass

        rd = getattr(results, "results_dict", None)
        if isinstance(rd, dict):
            map50    = float(rd.get("metrics/mAP50(B)", rd.get("metrics/mAP50", map50)))
            map5095  = float(rd.get("metrics/mAP50-95(B)", rd.get("metrics/mAP50-95", map5095)))
            if "metrics/precision(B)" in rd or "metrics/precision" in rd:
                P = float(rd.get("metrics/precision(B)", rd.get("metrics/precision", P)))
            if "metrics/recall(B)" in rd or "metrics/recall" in rd:
                R = float(rd.get("metrics/recall(B)", rd.get("metrics/recall", R)))
            if "metrics/f1(B)" in rd or "metrics/f1" in rd:
                F1 = float(rd.get("metrics/f1(B)", rd.get("metrics/f1", F1)))

        loss = 1.0 - float(map50)

        n_eval = robust_count_images(self.data_yaml, split if split in ("val", "test") else "val")
        if n_eval <= 0:
            n_eval = 1

        print(
            f"[INFO][CLIENT][EVAL][ROUND {round_id}][CID {self.cid}] "
            f"split={split} P={P:.4f} R={R:.4f} F1={F1:.4f} mAP50={map50:.4f} mAP50-95={map5095:.4f} loss={loss:.4f}"
        )
        return float(loss), int(n_eval), {
            "map50": float(map50),
            "map50_95": float(map5095),
            "precision": float(P),
            "recall": float(R),
            "f1": float(F1),
        }


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", required=True, type=str)
    ap.add_argument("--server", required=True, type=str)
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--data", required=True, type=str)
    args = ap.parse_args()

    device = "cpu"
    client = YoloFlowerClient(cid=args.cid, model_path=args.model, data_yaml=args.data, device=device)
    print(f"🤝 Iniciando cliente CID={args.cid} | servidor={args.server} | modelo={args.model}")

    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()
