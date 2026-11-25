import streamlit as st
import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew
from fpdf import FPDF
import fitz
import pydicom
import tempfile
import os
from datetime import datetime

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTÉTICA DASHBOARD EDITORIAL + MÁQUINA DE ESCRIBIR
# ==============================================================================
st.set_page_config(
    page_title="PINKTHERMO - SISTEMA DE DIAGNÓSTICO TERMOGRÁFICO",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&family=Special+Elite&display=swap');

    /* Todo en MAYÚSCULAS y tipografía de máquina de escribir */
    html, body, [class*="css"], .stApp, p, span, div, label, li, button, input, select {
        font-family: 'Courier Prime', 'Special Elite', 'Courier New', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 0.4px;
    }

    /* Fondo exterior lila suave */
    .stApp {
        background-color: #C7B3CA;
        color: #3D2B42;
        font-size: 13px;
    }

    /* Barra lateral izquierda */
    [data-testid="stSidebar"] {
        background-color: #EAE3EB;
        border-right: 1px solid #D3C4D6;
    }

    /* Contenedor principal crema suave */
    .main-banner {
        background-color: #F7F3EE;
        border-radius: 18px;
        padding: 20px 26px;
        margin-bottom: 18px;
        box-shadow: 0 6px 18px rgba(61, 43, 66, 0.08);
        border: 1px solid #E5DDD5;
    }

    /* Tarjetas internas color malva claro */
    .dash-card {
        background-color: #E5DCE6;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 16px;
        border: 1px solid #D5C8D7;
    }

    /* Tarjetas oscuras morado ciruela para porcentajes */
    .purple-stat-card {
        background-color: #8D7492;
        color: #FFFFFF !important;
        border-radius: 16px;
        padding: 15px 12px;
        text-align: center;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(61, 43, 66, 0.15);
    }
    .purple-stat-card * {
        color: #FFFFFF !important;
    }

    /* Títulos principales en negrita resaltada */
    .bold-header {
        font-family: 'Special Elite', 'Courier Prime', monospace !important;
        font-weight: 900 !important;
        font-size: 20px !important;
        color: #2E1D33 !important;
        margin-bottom: 6px;
        display: block;
    }
    .section-title {
        font-family: 'Special Elite', 'Courier Prime', monospace !important;
        font-weight: 900 !important;
        font-size: 15px !important;
        color: #3A243F !important;
        margin-bottom: 10px;
        display: block;
    }

    /* Filas internas blancas redondeadas */
    .pill-item {
        background-color: #F7F3EE;
        border-radius: 12px;
        padding: 11px 14px;
        margin-bottom: 9px;
        border: 1px solid #DDD3DE;
        color: #3D2B42;
        font-size: 12px;
    }

    /* Botón de descarga morado elegante */
    .stDownloadButton > button {
        background-color: #7D6283 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        width: 100%;
        box-shadow: 0 4px 10px rgba(61, 43, 66, 0.2);
    }
    .stDownloadButton > button:hover {
        background-color: #634B68 !important;
    }

    [data-testid="stMetric"] {
        background-color: #F7F3EE;
        border: 1px solid #D5C8D7;
        border-radius: 14px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CARGA DE MODELO SVM OPTIMIZADO Y ESCALADOR
# ==============================================================================
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelo_termografia_v2.pkl")
    escalador = joblib.load("escalador_termografia_v2.pkl")
    return modelo, escalador

modelo, escalador = cargar_modelo()

# ==============================================================================
# 3. LECTOR UNIVERSAL DE ESTUDIOS MÉDICOS
# ==============================================================================
def leer_estudio_medico(archivo):
    nombre = archivo.name.lower()
    bytes_data = archivo.read()
    
    if nombre.endswith('.pdf'):
        doc = fitz.open(stream=bytes_data, filetype="pdf")
        pagina = doc.load_page(0)
        pix = pagina.get_pixmap(dpi=200)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        img_gris = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY if pix.n == 4 else cv2.COLOR_RGB2GRAY)
        return img_gris, "DOCUMENTO PDF CLÍNICO (.PDF)"
        
    elif nombre.endswith('.dcm'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
            tmp.write(bytes_data)
            tmp_path = tmp.name
        ds = pydicom.dcmread(tmp_path)
        os.remove(tmp_path)
        arr = ds.pixel_array.astype(float)
        arr_norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        img_gris = cv2.cvtColor(arr_norm, cv2.COLOR_RGB2GRAY) if len(arr_norm.shape) == 3 else arr_norm
        return img_gris, "IMAGEN MÉDICA DICOM (.DCM)"
        
    else:
        arr = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        img_gris = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        return img_gris, f"IMAGEN DIGITAL (.{nombre.split('.')[-1].upper()})"

# ==============================================================================
# 4. EXTRACCIÓN DE BIOMARCADORES Y MOTOR DE ALTA SENSIBILIDAD + RIESGO LATENTE
# ==============================================================================
def limpiar_recorte_termografia(img_gris):
    mask_contenido = ((img_gris > 15) & (img_gris < 245)).astype(np.uint8)
    coords = cv2.findNonZero(mask_contenido)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        if w > 50 and h > 50:
            img_gris = img_gris[y:y+h, x:x+w]

    alto = img_gris.shape[0]
    img_limpia = img_gris[:int(alto * 0.87), :]
    return img_limpia

def calcular_biomarcadores_y_riesgo(img_limpia):
    candidatas = [img_limpia, 255 - img_limpia]
    mejor_score = -1.0
    mejor_df = None
    mejor_img_calor = img_limpia

    columnas = [
        'Temp_Media', 'Temp_Max', 'Temp_Min', 'Desv_Estandar',
        'Percentil_75', 'Percentil_90', 'Percentil_95', 'Ratio_Puntos_Calientes',
        'Asimetria_Bilateral_Media', 'Asimetria_Bilateral_Max', 'Gradiente_Vascular', 'Skewness_Termico']
    
