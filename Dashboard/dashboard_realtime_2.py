import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
from collections import Counter, deque
import time
import tkinter as tk
from datetime import datetime

# === Inicializa variáveis globais ===
model = YOLO(r"C:\Users\new_d\Downloads\best_YoloV8.pt")
contagem_acumulada = {}
cores_por_classe = {}
historico_ultimas = deque(maxlen=5)
ultima_classe = "---"
start_time = time.time()
cap = None
canvas_video = None
reset_time = time.time()  # nova variável para cálculos por minuto

defeitos_criticos = {0, 1, 2}

labels_estatisticas = {}
labels_contagem_classe = {}
labels_historico = []

# Tradução das classes para inglês
traducao_classes = {
    "Sem_CI": "No_IC",
    "Sem_resistor": "No_Resistor",
    "Sem_transistor": "No_Transistor",
    "Sem_diodo": "No_Diode",
    "Sem_capacitor": "No_Capacitor",
    "Sem_oscilador": "No_Oscillator",
    "Sem_chave": "No_Switch",
    "Sem_antena": "No_Antenna",
    "Sem_transformador": "No_Transformer"
}

def gerar_cor_hex(idx):
    np.random.seed(idx)
    cor = tuple(np.random.randint(100, 256, size=3))
    return "#%02x%02x%02x" % cor

def resetar_estatisticas():
    global contagem_acumulada, ultima_classe, historico_ultimas, reset_time
    contagem_acumulada = {}
    ultima_classe = "---"
    historico_ultimas.clear()
    reset_time = time.time()
    atualizar_labels_estatisticas()

def atualizar_labels_estatisticas():
    tempo_decorrido = time.time() - reset_time
    tempo_formatado = time.strftime('%H:%M:%S', time.gmtime(tempo_decorrido))
    total_detectado = sum(contagem_acumulada.values())
    taxa_por_minuto = total_detectado / (tempo_decorrido / 60) if tempo_decorrido > 0 else 0

    labels_estatisticas["tempo"].config(text=tempo_formatado)
    labels_estatisticas["total"].config(text=total_detectado)
    labels_estatisticas["taxa"].config(text=f"{taxa_por_minuto:.1f}")
    labels_estatisticas["ultima"].config(text=ultima_classe)

    for cls_id, label in labels_contagem_classe.items():
        valor = contagem_acumulada.get(cls_id, 0)
        label.config(text=str(valor))

    for i, lbl in enumerate(labels_historico):
        if i < len(historico_ultimas):
            lbl.config(text=f"- {historico_ultimas[-(i+1)]}")
        else:
            lbl.config(text="")

    painel_info.after(500, atualizar_labels_estatisticas)

def atualizar_video():
    global cap, ultima_classe
    ret, frame = cap.read()
    if not ret:
        return

    results = model.predict(source=frame, conf=0.5, save=False)[0]
    classes_detectadas = []

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < 0.5:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        nome_original = model.names[cls]
        nome_traduzido = traducao_classes.get(nome_original, nome_original)
        label = f"{nome_traduzido} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        classes_detectadas.append(cls)
        ultima_classe = nome_traduzido
        historico_ultimas.append(ultima_classe)

    contagem_atual = Counter(classes_detectadas)
    for cls, count in contagem_atual.items():
        contagem_acumulada[cls] = contagem_acumulada.get(cls, 0) + count

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    canvas_video.imgtk = img_tk
    canvas_video.create_image(0, 0, anchor=tk.NW, image=img_tk)

    janela.after(30, atualizar_video)

def iniciar_camera():
    global cap, canvas_video

    if cap:
        cap.release()

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, frame = cap.read()
    if not ret:
        print("Error starting camera.")
        return

    h, w, _ = frame.shape

    for widget in frame_video.winfo_children():
        widget.destroy()

    canvas_video = tk.Canvas(frame_video, width=w, height=h)
    canvas_video.pack()

    atualizar_video()
    atualizar_labels_estatisticas()

def construir_painel():
    global labels_estatisticas, labels_contagem_classe, labels_historico

    datahora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    tk.Label(painel_info, text=f"📅 {datahora}", bg="#1e1e1e", fg="#00ff88",
             font=("Consolas", 11, "bold")).pack(pady=(0, 10))

    tk.Label(painel_info, text="🔡 Statistics", bg="#1e1e1e", fg="white", font=("Consolas", 13, "bold")).pack(pady=(0, 10))

    stats_keys = [
        ("⏱ Runtime", "tempo"),
        ("📦 Total Detected", "total"),
        ("📈 Defects per Minute", "taxa"),
        ("📌 Last Detected Class", "ultima")
    ]

    for label_text, key in stats_keys:
        box = tk.Frame(painel_info, bg="#333", bd=1, relief="groove")
        box.pack(fill="x", padx=5, pady=3)
        tk.Label(box, text=label_text, fg="#00ffe1", bg="#333", font=("Consolas", 11, "bold")).pack(side="left", padx=10)
        lbl = tk.Label(box, text="---", fg="white", bg="#333", font=("Consolas", 11))
        lbl.pack(side="right", padx=10)
        labels_estatisticas[key] = lbl

    tk.Label(painel_info, text="\n📚 Class Count", bg="#1e1e1e", fg="white", font=("Consolas", 13, "bold")).pack(pady=(10, 5))

    for cls_id, nome in model.names.items():
        nome_traduzido = traducao_classes.get(nome, nome)
        cor = gerar_cor_hex(cls_id)
        container = tk.Frame(painel_info, bg="#2c2c2c", bd=1, relief="ridge")
        container.pack(fill="x", padx=5, pady=2)
        tk.Label(container, text=f"{nome_traduzido}:", fg=cor, bg="#2c2c2c", font=("Consolas", 11, "bold")).pack(side="left", padx=10)
        lbl = tk.Label(container, text="0", fg="white", bg="#2c2c2c", font=("Consolas", 11))
        lbl.pack(side="right", padx=10)
        labels_contagem_classe[cls_id] = lbl

    tk.Label(painel_info, text="\n🧾 Recent Detections", bg="#1e1e1e", fg="white", font=("Consolas", 13, "bold")).pack(pady=(10, 5))

    for _ in range(5):
        lbl = tk.Label(painel_info, text="", bg="#1e1e1e", fg="white", font=("Consolas", 10))
        lbl.pack(anchor="w", padx=10)
        labels_historico.append(lbl)

# === GUI Principal ===
janela = tk.Tk()
janela.title("HMI - PCB Defect Detection")
janela.geometry("1100x650")
janela.configure(bg="#1e1e1e")

frame_principal = tk.Frame(janela, bg="#1e1e1e")
frame_principal.pack(fill="both", expand=True)

frame_video = tk.Frame(frame_principal, bg="#000000")
frame_video.pack(side="left", padx=10, pady=10)

painel_info = tk.Frame(frame_principal, bg="#1e1e1e", width=280)
painel_info.pack(side="right", fill="y", padx=10, pady=10)

botao_reset = tk.Button(janela, text="🔄 Reset Statistics", command=resetar_estatisticas,
                        bg="#444", fg="white", font=("Consolas", 11, "bold"))
botao_reset.pack(pady=(0, 10))

construir_painel()
iniciar_camera()
janela.mainloop()
cap.release()
cv2.destroyAllWindows()
