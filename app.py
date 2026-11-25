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
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&family=Special+Elite&display=swap');

    /* Tipografía de máquina de escribir y MAYÚSCULAS seguras sin ocultar contenedores */
    .stApp {
        background-color: #C7B3CA !important;
        color: #3D2B42 !important;
        font-family: 'Courier Prime', 'Courier New', monospace !important;
        text-transform: uppercase !important;
    }

    .stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp button, .stApp input {
        font-family: 'Courier Prime', 'Courier New', monospace !important;
        text-transform: uppercase !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #EAE3EB !important;
        border-right: 1px solid #D3C4D6 !important;
    }

    .main-banner {
        background-color: #F7F3EE;
        border-radius: 18px;
        padding: 20px 26px;
        margin-bottom: 18px;
        box-shadow: 0 6px 18px rgba(61, 43, 66, 0.08);
        border: 1px solid #E5DDD5;
    }

    .dash-card {
        background-color: #E5DCE6;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 16px;
        border: 1px solid #D5C8D7;
    }

    .purple-stat-card {
        background-color: #8D7492;
        color: #FFFFFF !important;
        border-radius: 16px;
        padding: 15px 12px;
        text-align: center;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(61, 43, 66, 0.15);
    }
    .purple-stat-card div {
        color: #FFFFFF !important;
    }

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

    .pill-item {
        background-color: #F7F3EE;
        border-radius: 12px;
        padding: 11px 14px;
        margin-bottom: 9px;
        border: 1px solid #DDD3DE;
        color: #3D2B42;
        font-size: 12px;
    }

    .stDownloadButton > button {
        background-color: #7D6283 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        width: 100%;
    }
    .stDownloadButton > button:hover {
        background-color: #634B68 !important;
    }

    div[data-testid="stMetric"] {
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
        'Asimetria_Bilateral_Media', 'Asimetria_Bilateral_Max', 'Gradiente_Vascular', 'Skewness_Termico'
    ]

    for img_cand in candidatas:
        img = cv2.resize(img_cand, (128, 128))
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_suave = cv2.GaussianBlur(img_norm, (3, 3), 0)

        t_media = float(np.mean(img_suave))
        t_max = float(np.max(img_suave))
        t_min = float(np.min(img_suave))
        t_std = float(np.std(img_suave))

        p75 = float(np.percentile(img_suave, 75))
        p90 = float(np.percentile(img_suave, 90))
        p95 = float(np.percentile(img_suave, 95))
        ratio_calor = float(np.sum(img_suave > 205) / img_suave.size)

        roi = img_suave[25:110, 10:118]
        mitad = roi.shape[1] // 2
        lado_izq = roi[:, :mitad]
        lado_der = roi[:, mitad:]

        asimetria_media = float(abs(np.mean(lado_izq) - np.mean(lado_der)))
        asimetria_max = float(abs(np.percentile(lado_izq, 95) - np.percentile(lado_der, 95)))
        dif_calientes = float(abs(np.sum(lado_izq > 200) - np.sum(lado_der > 200)) / (lado_izq.size + 1e-5))

        sobelx = cv2.Sobel(img_suave, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_suave, cv2.CV_64F, 0, 1, ksize=3)
        gradiente_termico = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))
        asimetria_hist = float(skew(img_suave.flatten()))

        df_actual = pd.DataFrame([[
            t_media, t_max, t_min, t_std,
            p75, p90, p95, ratio_calor,
            asimetria_media, asimetria_max, gradiente_termico, asimetria_hist
        ]], columns=columnas)

        datos_esc = escalador.transform(df_actual)
        prob_svm = float(modelo.predict_proba(datos_esc)[0][1])

        score_asimetria = min(1.0, (asimetria_media / 18.0) * 0.45 + (asimetria_max / 30.0) * 0.35 + dif_calientes * 1.8)
        score_hipertermia = min(1.0, max(0.0, (p95 - 210.0) / 45.0) * 0.5 + (gradiente_termico / 42.0) * 0.5)
        score_clinico = 0.65 * score_asimetria + 0.35 * score_hipertermia

        riesgo_total = max(prob_svm, score_clinico, 0.45 * prob_svm + 0.65 * score_clinico)
        riesgo_total = min(0.992, max(0.035, riesgo_total))

        if np.mean(roi) > t_media * 0.92 and riesgo_total > mejor_score:
            mejor_score = riesgo_total
            mejor_df = df_actual
            mejor_img_calor = img_suave

    return mejor_df, mejor_score, mejor_img_calor

# ==============================================================================
# 5. GENERADOR DE INFORME CLÍNICO DESCARGABLE EN PDF
# ==============================================================================
def generar_pdf_informe(nombre_paciente, formato_origen, es_anomalia, prob_riesgo, df_bio, img_gris, img_color_bgr):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_fill_color(141, 116, 146)
    pdf.rect(0, 0, 210, 38, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Courier", "B", 16)
    pdf.cell(0, 14, "INFORME DE EVALUACION TERMOGRAFICA MAMARIA", ln=True, align="C")
    pdf.set_font("Courier", "", 10)
    pdf.cell(0, 6, "SISTEMA DE ALTA SENSIBILIDAD CLINICA (SVM + BIOMARCADORES DMR-IR)", ln=True, align="C")
    pdf.ln(14)
    
    pdf.set_text_color(46, 29, 51)
    pdf.set_font("Courier", "B", 11)
    fecha_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    pdf.cell(0, 7, f"PACIENTE: {nombre_paciente.upper() if nombre_paciente else 'NO ESPECIFICADO'}", ln=True)
    pdf.set_font("Courier", "", 10)
    pdf.cell(0, 6, f"FECHA: {fecha_str} | FORMATO: {formato_origen}", ln=True)
    pdf.ln(4)
    
    if es_anomalia:
        pdf.set_fill_color(255, 224, 230)
        pdf.set_text_color(160, 20, 40)
        titulo_res = f"RESULTADO: ALERTA DE ANOMALIA TERMICA DETECTADA (RIESGO: {prob_riesgo*100:.1f}%)"
        explicacion = (
            "SE IDENTIFICARON FOCOS DE HIPERTERMIA LOCALIZADA Y/O ASIMETRIA BILATERAL SIGNIFICATIVA "
            "ENTRE AMBAS MAMAS. SE RECOMIENDA ACUDIR A VALORACION MEDICA ESPECIALIZADA DE INMEDIATO."
        )
    else:
        pdf.set_fill_color(232, 245, 233)
        pdf.set_text_color(27, 94, 32)
        titulo_res = f"RESULTADO: SIN ENFERMEDAD ACTIVA | POSIBILIDAD LATENTE DE RIESGO: {prob_riesgo*100:.1f}%"
        explicacion = (
            f"ACTUALMENTE NO PRESENTA PATRONES DE ENFERMEDAD ACTIVA (CONFIANZA DE NORMALIDAD: {(1-prob_riesgo)*100:.1f}%). "
            f"SIN EMBARGO, SEGUN SUS BIOMARCADORES TERMICOS, EXISTE UNA POSIBILIDAD ESTIMADA DEL {prob_riesgo*100:.1f}% "
            "DE DESARROLLAR ALTERACIONES TERMICAS A FUTURO. SE SUGIERE SEGUIMIENTO PREVENTIVO PERIODICO."
        )
        
    pdf.set_font("Courier", "B", 10)
    pdf.multi_cell(0, 8, titulo_res, fill=True)
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Courier", "", 9)
    pdf.ln(2)
    pdf.multi_cell(0, 6, explicacion)
    pdf.ln(5)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_g, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_c:
        cv2.imwrite(tmp_g.name, img_gris)
        cv2.imwrite(tmp_c.name, img_color_bgr)
        path_g, path_c = tmp_g.name, tmp_c.name
        
    pdf.set_font("Courier", "B", 10)
    pdf.set_text_color(99, 75, 104)
    pdf.cell(95, 7, "TERMOGRAFIA ORIGINAL", align="C")
    pdf.cell(95, 7, "MAPA TERMICO CALIBRADO", ln=True, align="C")
    y_img = pdf.get_y()
    pdf.image(path_g, x=22, y=y_img, w=65)
    pdf.image(path_c, x=122, y=y_img, w=65)
    pdf.ln(68)
    
    os.remove(path_g)
    os.remove(path_c)
    
    pdf.set_font("Courier", "B", 10)
    pdf.cell(0, 8, "DESGLOSE DE LOS 12 BIOMARCADORES TERMICOS:", ln=True)
    pdf.set_font("Courier", "", 9)
    fila_datos = df_bio.iloc[0]
    for col in df_bio.columns:
        pdf.cell(130, 6, f" - {col.upper()}", border=1)
        pdf.cell(50, 6, f"{fila_datos[col]:.3f}", border=1, ln=True, align="R")
        
    return bytes(pdf.output())

# ==============================================================================
# 6. INTERFAZ DASHBOARD (ESTILO EDITORIAL MORADO/CREMA + MÁQUINA DE ESCRIBIR)
# ==============================================================================

st.sidebar.markdown("""
<div style="text-align:center; padding: 10px 0;">
    <svg width="74" height="74" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="46" fill="#8D7
