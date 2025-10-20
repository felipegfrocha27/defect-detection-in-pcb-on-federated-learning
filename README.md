# Defect Detection in PCBs — Federated Learning

This repository accompanies the work **“Defect Detection in Printed Circuit Boards Based on Federated Learning.”**  
It provides minimal, reproducible code for **object detection on PCBs** using YOLO-family models, MobileNetV2/FOMO, and a **local dashboard**.  
Both **centralized training** and **federated training** (Flower) are supported, with Raspberry Pi edge nodes for on-device tests.

---

## Repository Structure
├── FL/                       # Treinamento federado (server + clients)
│   ├── server/               # Servidor Flower (FedAvg)
│   └── clients/              # Clientes (Raspberry Pi 4/5)
│
├── Centralized/              # Linha de base centralizada (notebooks/artefatos)
│   ├── train_yolov8_object_detection_on_custom_dataset.ipynb
│   ├── train_yolov12_object_detection_model.ipynb
│   ├── Felipe_Mobilenetssdfpnlite_320x320.ipynb
│   └── Fomo_Model.txt        # Export do modelo/impulse do Edge Impulse
│
├── Dashboard/                # IHM (stream + detecções + métricas)
│   └── app/                  # Código da interface
│
└── README.md