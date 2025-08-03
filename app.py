import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# Cargar modelos disponibles
MODEL_PATHS = {
    "YOLOv8": "models/yolov8.pt",
    "YOLOv9": "models/yolov9.pt",
    "YOLOv10": "models/yolov10.pt",
    "YOLOv11": "models/yolov11.pt"
}

# Configuración de la página
st.set_page_config(page_title="Detección de Malaria", layout="centered")
st.title("🦟 Detección de fases de malaria en imágenes")

# Selección del modelo
model_name = st.selectbox("Selecciona el modelo YOLO", list(MODEL_PATHS.keys()))
model = YOLO(MODEL_PATHS[model_name])

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

# Mostrar y procesar imagen
if uploaded_file:
    # Leer la imagen
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    st.image(img_array, caption="Imagen Original", use_column_width=True)

    if st.button("📍 Detectar Fases"):
        # Guardar temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image.save(temp.name)
            temp_path = temp.name

        # Realizar predicción
        results = model(temp_path)[0]

        # Dibujar resultados
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls_id]} ({conf:.2f})"

            cv2.rectangle(img_array, (x1, y1), (x2, y2), (65, 65, 222), 4)
            cv2.putText(img_array, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (65, 65, 222), 3)

        st.image(img_array, caption="Resultados de Detección", use_column_width=True)
