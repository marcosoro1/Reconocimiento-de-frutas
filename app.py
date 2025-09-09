import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Reconocimiento de Frutas", page_icon="🍎")
st.title("🍎 Reconocimiento de Frutas - Cámara en Vivo")

# Cargar el modelo una vez
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor

def zero_shot_classification(image, model, processor):
    try:
        categories = [
            "manzana", "banana", "naranja", "uva", "fresa", 
            "sandía", "melón", "piña", "mango", "kiwi",
            "cereza", "limón", "pera", "durazno", "ciruela",
            "frambuesa", "arándano", "granada", "papaya", "coco",
            "aguacate", "palta", "maracuyá", "guayaba", "lichi"
        ]
        
        inputs = processor(
            text=categories,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        results = {categories[i]: float(probs[0][i]) for i in range(len(categories))}
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_results
        
    except Exception as e:
        st.error(f"Error procesando la imagen: {str(e)}")
        return None

# Cargar modelo
model, processor = load_model()

# Selección de modo
modo = st.radio("Selecciona el modo:", ["📷 Usar Cámara", "📂 Subir Imagen"])

if modo == "📷 Usar Cámara":
    st.header("📷 Modo Cámara en Vivo")
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("No se pudo acceder a la cámara. Verifica que esté conectada.")
    else:
        # Botones de control
        col1, col2 = st.columns(2)
        with col1:
            tomar_foto = st.button("📸 Tomar Foto", type="primary")
        with col2:
            detener_camara = st.button("⏹️ Detener Cámara")
        
        # Mostrar video en vivo
        frame_placeholder = st.empty()
        
        while cap.isOpened() and not detener_camara:
            ret, frame = cap.read()
            
            if not ret:
                st.error("Error al capturar frame")
                break
            
            # Convertir BGR a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            if tomar_foto:
                # Procesar la foto tomada
                pil_image = Image.fromarray(frame_rgb)
                
                with st.spinner("🔍 Analizando..."):
                    results = zero_shot_classification(pil_image, model, processor)
                
                if results:
                    st.success("✅ Análisis completado!")
                    
                    top_fruit = list(results.keys())[0]
                    top_prob = results[top_fruit] * 100
                    
                    st.metric(
                        label="**Fruta detectada:**",
                        value=top_fruit.capitalize(),
                        delta=f"{top_prob:.1f}% de confianza"
                    )
                    
                    # Mostrar imagen capturada
                    st.image(pil_image, caption="Foto tomada", use_column_width=True)
                    
                    # Mostrar top 3 resultados
                    st.subheader("📊 Resultados:")
                    for i, (fruit, prob) in enumerate(list(results.items())[:3]):
                        st.write(f"**{i+1}. {fruit.capitalize()}:** {prob*100:.2f}%")
                
                tomar_foto = False
                break
            
        cap.release()

else:  # Modo subir imagen
    st.header("📂 Modo Subir Imagen")
    image = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png", "webp"])
    
    if image is not None:
        pil_image = Image.open(image)
        st.image(pil_image, caption="Imagen subida", use_column_width=True)
        
        with st.spinner("🔍 Analizando imagen..."):
            results = zero_shot_classification(pil_image, model, processor)
        
        if results:
            st.success("✅ Análisis completado!")
            
            top_fruit = list(results.keys())[0]
            top_prob = results[top_fruit] * 100
            
            st.metric(
                label="**Fruta detectada:**",
                value=top_fruit.capitalize(),
                delta=f"{top_prob:.1f}% de confianza"
            )
            
            # Gráfico de barras para top 5
            top_5 = dict(list(results.items())[:5])
            st.bar_chart({k: v*100 for k, v in top_5.items()})

# Información adicional
st.markdown("---")
st.info("""
💡 **Consejos para usar la cámara:**
- Asegura buena iluminación
- Enfoca bien la fruta
- Acerca la cámara lo suficiente
- Mantén la fruta sobre un fondo contrastante
""")