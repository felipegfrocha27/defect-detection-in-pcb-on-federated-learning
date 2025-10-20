#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from ultralytics import YOLO

# Plot headless (para gerar gráficos sem display)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Utils de serialização
# -------------------------

def params_to_ndarrays_from_yolo(model_path: str) -> List[np.ndarray]:
    """Carrega um YOLO do disco e serializa seus parâmetros por POSIÇÃO (ordem de .parameters())."""
    y = YOLO(model_path)
    return [p.detach().cpu().numpy() for p in y.model.parameters()]


def load_yolo_from_ndarrays(model_path: str, nds: List[np.ndarray]) -> YOLO:
    """Reconstrói um YOLO a partir do .pt base aplicando arrays por ORDEM POSICIONAL."""
    y = YOLO(model_path)
    with y.model.no_grad():
        for p, arr in zip(y.model.parameters(), nds):
            t = y.model._get_resolved_tensor(torch_like=p, np_array=arr)
            p.copy_(t)
    return y


def _named_param_keys(model: YOLO) -> List[str]:
    """Lista os nomes (na mesma ordem de .parameters()) para log/depuração."""
    names = []
    for (name, _), _p in zip(model.model.named_parameters(), model.model.parameters()):
        names.append(name)
    return names


# Pequena ajudinha para converter numpy -> tensor mantendo dtype/device do destino
# (evita importar torch no topo do arquivo à toa)
def _np_to_like(np_array, torch_like_param):
    import torch
    t = torch.from_numpy(np_array).to(device=torch_like_param.device, dtype=torch_like_param.dtype)
    return t


# Monkey helper dentro do objeto YOLO (só para este script)
def _patch_model_np_copy():
    import torch
    def _get_resolved_tensor(self, torch_like, np_array):
        return torch.from_numpy(np_array).to(device=torch_like.device, dtype=torch_like.dtype)
    YOLO._get_resolved_tensor = _get_resolved_tensor

_patch_model_np_copy()


# -------------------------
# Helpers: CSV e gráfico de histórico do servidor
# -------------------------

def _append_dict_row(csv_path: str, header: list, row: dict) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k) for k in header})


def _plot_server_history(csv_path: str, png_path: str) -> None:
    """Plota curvas agregadas por round: mAP50, mAP50-95, Precision, Recall, F1."""
    if not os.path.exists(csv_path):
        return
    rounds, m50, m5095, P, R, F1 = [], [], [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rounds.append(int(row["round"]))
            m50.append(float(row.get("map50_mean", 0.0)))
            m5095.append(float(row.get("map50_95_mean", 0.0)))
            P.append(float(row.get("precision_mean", 0.0)))
            R.append(float(row.get("recall_mean", 0.0)))
            F1.append(float(row.get("f1_mean", 0.0)))

    if not rounds:
        return

    plt.figure()
    if any(m50):   plt.plot(rounds, m50,   label="mAP@0.5 (mean)")
    if any(m5095): plt.plot(rounds, m5095, label="mAP@0.5:0.95 (mean)")
    if any(P):     plt.plot(rounds, P,     label="Precision (mean)")
    if any(R):     plt.plot(rounds, R,     label="Recall (mean)")
    if any(F1):    plt.plot(rounds, F1,    label="F1 (mean)")
    plt.xlabel("Round")
    plt.ylabel("Métrica agregada")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


# -------------------------
# Estratégia com logging, ponderação e eval central
# -------------------------

class SimpleFedAvg(fl.server.strategy.FedAvg):
    """FedAvg com:
       - salvamento dos pesos agregados por round (npz + nomes)
       - logging por-cliente
       - agregação ponderada por num_examples das métricas de avaliação
       - avaliação central opcional (server-side) em holdout fixo
    """

    def __init__(
        self,
        *,
        init_params: Optional[List[np.ndarray]],
        model_path: str,
        save_prefix: str = "global_round",
        server_eval_data: Optional[str] = None,
        server_eval_imgsz: int = 640,
        server_eval_workers: int = 2,
        server_eval_every: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            initial_parameters=(
                fl.common.ndarrays_to_parameters(init_params) if init_params is not None else None
            ),
            **kwargs,
        )
        self.model_path = model_path
        self.save_prefix = save_prefix

        # avaliação central opcional
        self.server_eval_data = server_eval_data
        self.server_eval_imgsz = int(server_eval_imgsz)
        self.server_eval_workers = int(server_eval_workers)
        self.server_eval_every = int(server_eval_every)

    # ----------------- Fit -----------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[fl.common.Parameters, Dict[str, float]]:
        # Primeiro, deixe o FedAvg agregar os pesos
        aggregated_parameters, _ = super().aggregate_fit(server_round, results, failures)

        # Logging de métricas de fit (tempo/num_examples) manualmente, sem usar fit_metrics_aggregation_fn
        try:
            num_total = int(sum(fr.num_examples for _, fr in results))
            times = [float((fr.metrics or {}).get("time_train_s", 0.0)) for _, fr in results]
            if times:
                print(
                    f"[INFO][SERVER][ROUND {server_round}] "
                    f"num_examples_total={num_total} | time_train_s_mean={np.mean(times):.2f} | time_train_s_sum={np.sum(times):.2f}"
                )
        except Exception:
            pass

        # Salvar pesos agregados por round (npz + nomes)
        if aggregated_parameters is not None:
            nd_arrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # salva npz
            np.savez(f"{self.save_prefix}{server_round}.npz", *nd_arrays)

            # salva _keys.txt com nomes na mesma ordem de parameters()
            ytmp = YOLO(self.model_path)
            keys = _named_param_keys(ytmp)
            with open(f"{self.save_prefix}{server_round}_keys.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(keys))

            print(f"[INFO][SERVER] Pesos globais salvos em {self.save_prefix}{server_round}.npz (+ _keys.txt)")

            # Avaliação central opcional neste ponto (após o fit)
            if (
                self.server_eval_data
                and self.server_eval_every > 0
                and (server_round % self.server_eval_every == 0)
            ):
                try:
                    # Reconstrói modelo com os pesos agregados deste round
                    y = YOLO(self.model_path)
                    with y.model.no_grad():
                        for p, arr in zip(y.model.parameters(), nd_arrays):
                            t = y.model._get_resolved_tensor(torch_like=p, np_array=arr)
                            p.copy_(t)

                    # roda y.val() central
                    res = y.val(
                        data=self.server_eval_data,
                        imgsz=self.server_eval_imgsz,
                        workers=self.server_eval_workers,
                        device="cpu",   # ajuste se quiser usar GPU no servidor
                        plots=False,
                        verbose=False,
                    )
                    map50   = float(getattr(res.box, "map50", 0.0))
                    map5095 = float(getattr(res.box, "map", 0.0))
                    precision = float(getattr(res.box, "p", 0.0))
                    recall    = float(getattr(res.box, "r", 0.0))
                    f1        = float(getattr(res.box, "f1", 0.0))

                    print(
                        f"[INFO][SERVER][CENTRAL_EVAL][ROUND {server_round}] "
                        f"imgsz={self.server_eval_imgsz} workers={self.server_eval_workers} "
                        f"map50={map50:.4f} map50_95={map5095:.4f} "
                        f"P={precision:.4f} R={recall:.4f} F1={f1:.4f}"
                    )
                except Exception as e:
                    print(f"[WARN][SERVER][CENTRAL_EVAL] Falhou a avaliação central: {e}")

        return aggregated_parameters, {}

    # ----------------- Evaluate -----------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[float, Dict[str, float]]:
        """Agrupa avaliação **ponderada por num_examples** e faz logging por cliente."""
        if not results:
            return 0.0, {}

        # Logging por cliente (agora com P/R/F1)
        for client, ev in results:
            cid = getattr(client, "cid", str(client))
            m = ev.metrics or {}
            print(
                f"[INFO][SERVER][ROUND {server_round}] from CID={cid} "
                f"loss={float(ev.loss):.4f} map50={float(m.get('map50', 0.0)):.4f} "
                f"map50_95={float(m.get('map50_95', 0.0)):.4f} "
                f"P={float(m.get('precision', 0.0)):.4f} R={float(m.get('recall', 0.0)):.4f} F1={float(m.get('f1', 0.0)):.4f}"
            )

        # Agregação ponderada
        totals = [max(1, int(ev.num_examples)) for _, ev in results]
        total_ex = float(sum(totals))
        w = [t / total_ex for t in totals]

        # Médias ponderadas
        loss_agg = float(sum(wi * float(ev.loss) for wi, (_, ev) in zip(w, results)))

        def _wmean(key: str) -> float:
            return float(sum(wi * float((ev.metrics or {}).get(key, 0.0)) for wi, (_, ev) in zip(w, results)))

        metrics = {
            "map50_mean": _wmean("map50"),
            "map50_95_mean": _wmean("map50_95"),
            "precision_mean": _wmean("precision"),
            "recall_mean": _wmean("recall"),
            "f1_mean": _wmean("f1"),
        }

        print(f"[INFO][SERVER][ROUND {server_round}] eval_loss: {loss_agg:.4f} | metrics: {metrics}")

        # Salvar histórico global por round + plot
        hist_csv = "server_history.csv"
        _append_dict_row(
            hist_csv,
            header=["round", "map50_mean", "map50_95_mean", "precision_mean", "recall_mean", "f1_mean", "loss_mean"],
            row={"round": server_round, **metrics, "loss_mean": loss_agg},
        )
        try:
            _plot_server_history(hist_csv, "server_history.png")
        except Exception as e:
            print(f"[WARN][SERVER] Falhou ao plotar histórico do servidor: {e}")

        return loss_agg, metrics


# -------------------------
# Config repassada aos clientes
# -------------------------

def fit_config(
    server_round: int,
    *,
    local_epochs: int,
    learning_rate: float,
    warmup_epochs: float,
    data_fraction: float,
    subset_seed: int,
    imgsz: int,
    batch: int,
    workers: int,
) -> Dict:
    return {
        "server_round": server_round,
        "local_epochs": local_epochs,
        "learning_rate": learning_rate,
        "warmup_epochs": warmup_epochs,
        "data_fraction": data_fraction,
        "subset_seed": subset_seed,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
    }


def eval_config(server_round: int, *, imgsz: int, workers: int, split: str) -> Dict:
    return {
        "server_round": server_round,
        "imgsz": imgsz,
        "workers": workers,
        "split": split,  # "val" ou "test" será lido pelo cliente
    }


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Servidor FL para YOLOv8")
    parser.add_argument("--model", required=True, type=str, help="Modelo inicial (.pt) com nc correto")
    parser.add_argument("--address", default="0.0.0.0:8080", type=str)
    parser.add_argument("--rounds", default=3, type=int)

    parser.add_argument("--fit_clients", default=2, type=int)
    parser.add_argument("--eval_clients", default=2, type=int)
    parser.add_argument("--min_available", default=2, type=int)

    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--warmup", default=0.0, type=float)

    parser.add_argument("--data_fraction", default=1.0, type=float)
    parser.add_argument("--subset_seed", default=0, type=int)

    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--workers", default=4, type=int)

    parser.add_argument("--eval_every", default=1, type=int)
    parser.add_argument("--eval_imgsz", default=640, type=int)
    parser.add_argument("--eval_workers", default=2, type=int)
    parser.add_argument("--eval_split", default="val", choices=["val", "test"], type=str,
                        help="Conjunto a ser usado no evaluate() dos clientes")

    # Avaliação central opcional
    parser.add_argument("--server_eval_data", default=None, type=str,
                        help="YAML de holdout para avaliação central (opcional)")
    parser.add_argument("--server_eval_imgsz", default=None, type=int,
                        help="imgsz da avaliação central; se None, usa --eval_imgsz")
    parser.add_argument("--server_eval_workers", default=None, type=int,
                        help="workers da avaliação central; se None, usa --eval_workers")
    parser.add_argument("--server_eval_every", default=None, type=int,
                        help="frequência de avaliação central; se None, usa --eval_every")

    args = parser.parse_args()

    print(
        f"🚀 Servidor FL em {args.address} | rounds={args.rounds} | "
        f"data_fraction={args.data_fraction} | eval_every={args.eval_every} | eval_split={args.eval_split}"
    )

    init_params = params_to_ndarrays_from_yolo(args.model)

    server_eval_imgsz = args.server_eval_imgsz if args.server_eval_imgsz is not None else args.eval_imgsz
    server_eval_workers = args.server_eval_workers if args.server_eval_workers is not None else args.eval_workers
    server_eval_every = args.server_eval_every if args.server_eval_every is not None else args.eval_every

    strategy = SimpleFedAvg(
        init_params=init_params,
        model_path=args.model,
        save_prefix="global_round",
        # FL amostragens/limiares
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.fit_clients,
        min_evaluate_clients=args.eval_clients,
        min_available_clients=args.min_available,
        # configs que os clientes recebem a cada round
        on_fit_config_fn=lambda sr: fit_config(
            sr,
            local_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_epochs=args.warmup,
            data_fraction=args.data_fraction,
            subset_seed=args.subset_seed if args.subset_seed != 0 else sr,  # muda por round se seed=0
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
        ),
        on_evaluate_config_fn=lambda sr: eval_config(
            sr,
            imgsz=args.eval_imgsz,
            workers=args.eval_workers,
            split=args.eval_split,
        ),
        # avaliação central opcional
        server_eval_data=args.server_eval_data,
        server_eval_imgsz=server_eval_imgsz,
        server_eval_workers=server_eval_workers,
        server_eval_every=server_eval_every,
        # comportamento estrito
        accept_failures=False,
    )

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
