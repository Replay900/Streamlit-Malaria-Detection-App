import streamlit as st
import cv2
import numpy as np
import uuid
from collections import Counter
from ultralytics import YOLO

CLASSES = ["trophozoite", "ring", "schizont", "gametocyte"]

@st.cache_resource
def load_model(name):
    path_map = {
        "YOLOv8": "model_v8.pt",
        "YOLOv9": "model_v9.pt",
        "YOLOv10": "model_v10.pt",
        "YOLOv11": "model_v11.pt",
    }
    return YOLO(path_map[name])

st.set_page_config(page_title="Detección de Malaria con YOLO", layout="centered")
st.title("🧫 Detección de Células de Malaria")

col1, col2 = st.columns([1, 1])
with col1:
    uploaded = st.file_uploader("Arrastra la imagen o haz clic para seleccionar", type=["jpg", "jpeg", "png"])
with col2:
    modelo_sel = st.radio("Selecciona modelo YOLO", ("YOLOv8", "YOLOv9", "YOLOv10", "YOLOv11"))

if uploaded:
    bytes_data = uploaded.read()
    np_img = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    model = load_model(modelo_sel)

    conf = st.slider("Seleccionar umbral de confianza", 0.0, 1.0, 0.25)

    with st.spinner("🔍 Procesando imagen..."):
        results = model.predict(img, conf=conf)[0]

    counts = Counter()
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf_score = float(box.conf[0])
        label = f"{CLASSES[cls]}: {conf_score:.2f}"

        thickness = 4
        font_scale = 1.0
        font_thickness = 3

        cv2.rectangle(img, (x1, y1), (x2, y2), (65, 65, 222), thickness)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (65, 65, 222), font_thickness)
        counts[CLASSES[cls]] += 1

    st.image(img[:, :, ::-1], caption=f"Resultado ({modelo_sel})", use_container_width=True)

    total = sum(counts.values())
    st.subheader("Resumen de detecciones:")
    st.write(f"- Células infectadas: {total}")
    for clase in CLASSES:
        st.write(f"- {clase.capitalize()}: {counts.get(clase, 0)}")
