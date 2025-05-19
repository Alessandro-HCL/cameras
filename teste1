import streamlit as st
import cv2
import torch
import time
from datetime import datetime
import os
import yagmail
import tempfile
from PIL import Image
import numpy as np

# ⚙️ Configurações de e-mail
EMAIL_REMETENTE = "ale.moreira@gmail.com"
SENHA_APP = "gncuqrzzkstgeamn"
EMAIL_DESTINO = "ale.moreira@gmail.com"
yag = yagmail.SMTP(EMAIL_REMETENTE, SENHA_APP)

# 🗂️ Cria diretório para salvar imagens
diretorio_img = "imagens_mochila"
os.makedirs(diretorio_img, exist_ok=True)

# 🧠 Carrega modelo YOLOv5
st.info("Carregando modelo YOLOv5...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.25
model.iou = 0.4
model.classes = [24, 26]  # mochila e mala
model.augment = True

# Função de detecção com RTSP
@st.cache_resource(show_spinner=False)
def get_video_capture():
    usuario = 'admin'
    senha = 'EAQL9C67'
    ip_camera = '192.168.1.11'
    porta = '554'
    rtsp_url = f"rtsp://{usuario}:{senha}@{ip_camera}:{porta}/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
    cap = cv2.VideoCapture(rtsp_url)
    return cap

# Interface Streamlit
st.title("📹 Monitoramento de Mochilas com YOLOv5")
iniciar = st.button("🔄 Iniciar Monitoramento")

frame_spot = st.empty()
log_spot = st.empty()

if iniciar:
    cap = get_video_capture()
    if not cap.isOpened():
        st.error("Erro ao abrir o stream RTSP.")
    else:
        st.success("Monitoramento iniciado. Aguarde...")
        ultimo_registro = 0
        intervalo_segundos = 30

        while True:
            ret, frame = cap.read()
            if not ret:
                log_spot.error("Erro ao capturar frame.")
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

            if mochila_detectada and time.time() - ultimo_registro > intervalo_segundos:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(diretorio_img, f"mochila_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                log_spot.success(f"[SALVO] Mochila detectada: {filename}")
                ultimo_registro = time.time()

                try:
                    yag.send(
                        to=EMAIL_DESTINO,
                        subject="🎒 Mochila Detectada pela Câmera",
                        contents=f"Uma mochila foi detectada na câmera às {timestamp}.",
                        attachments=filename
                    )
                    log_spot.success("[ENVIADO] E-mail enviado com sucesso!")
                except Exception as e:
                    log_spot.error(f"[ERRO AO ENVIAR EMAIL] {e}")

            # Exibe imagem no Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_spot.image(frame_rgb, channels="RGB")

            if not st.button("❌ Parar", key="stop"):
                time.sleep(1)
            else:
                break

        cap.release()
        st.warning("Monitoramento finalizado.")
