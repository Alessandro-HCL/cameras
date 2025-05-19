import streamlit as st
import cv2
import torch
import time
from datetime import datetime
import os
import yagmail
from PIL import Image
import numpy as np

# ⚙️ Configurações iniciais
st.set_page_config(page_title="Detecção de Mochilas", layout="wide")
st.title("🎥 Monitoramento com Detecção de Mochilas")

# 🔑 Configuração de e-mail
EMAIL_REMETENTE = "ale.moreira@gmail.com"
SENHA_APP = "gncuqrzzkstgeamn"
EMAIL_DESTINO = "ale.moreira@gmail.com"
yag = yagmail.SMTP(EMAIL_REMETENTE, SENHA_APP)

# 🎨 Carrega modelo YOLOv5
with st.spinner("Carregando modelo YOLOv5..."):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.25
    model.iou = 0.4
    model.classes = [24, 26]  # mochila e mala
    model.augment = True

# Diretório para salvar imagens
os.makedirs("imagens_mochila", exist_ok=True)

# RTSP da câmera
usuario = 'admin'
senha = 'EAQL9C67'
ip_camera = '192.168.1.11'
porta = '554'
rtsp_url = f"rtsp://{usuario}:{senha}@{ip_camera}:{porta}/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"

# Inicia captura
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    st.error("Erro ao abrir o stream da câmera RTSP.")
    st.stop()

# Variáveis de controle
ultimo_registro = 0
intervalo_segundos = 30
frame_placeholder = st.empty()

# Processa os frames em tempo real
while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("Erro ao capturar frame.")
        break

    height, width, _ = frame.shape
    sub_frame = frame[int(height / 2):, :]

    results = model(sub_frame)
    mochila_detectada = False

    for *xyxy, conf, cls in results.xyxy[0]:
        classe = int(cls)
        label = f'{model.names[classe]} {conf:.2f}'
        cor = (0, 0, 255)

        x1, y1, x2, y2 = map(int, xyxy)
        y1 += int(height / 2)
        y2 += int(height / 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

        if classe == 24:
            mochila_detectada = True

    # Exibe frame na interface
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")

    if mochila_detectada and time.time() - ultimo_registro > intervalo_segundos:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("imagens_mochila", f"mochila_{timestamp}.jpg")
        cv2.imwrite(filename, frame)

        try:
            yag.send(
                to=EMAIL_DESTINO,
                subject="🎒 Mochila Detectada pela Câmera",
                contents=f"Uma mochila foi detectada na câmera às {timestamp}.",
                attachments=filename
            )
            st.success(f"Mochila detectada e e-mail enviado: {filename}")
        except Exception as e:
            st.error(f"Erro ao enviar e-mail: {e}")

        ultimo_registro = time.time()

