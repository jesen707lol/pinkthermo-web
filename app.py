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

st.set_page_config(page_title="PINKTHERMO - DIAGNÓSTICO TERMOGRÁFICO", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&family=Special+Elite&display=swap');

.stApp { background-color: #C7B3CA !important; color: #3D2B42 !important; }
.stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp input, .stApp div[data-testid="stMarkdownContainer"] {
    font-family: 'Courier Prime', monospace !important;
    text-transform: uppercase !important;
}
[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
    text-transform: none !important;
}
section[data-testid="stSidebar"] { background-color: #EAE3EB !important; border-right: 1px solid #D3C4D6 !important; }
.main-banner { background-color: #F7F3EE; border-radius: 18px; padding: 18px 24px; margin-bottom: 16px; border: 1px solid #E5DDD5; }
.dash-card { background-color: #E5DCE6; border-radius: 16px; padding: 16px; margin-bottom: 14px; border: 1px solid #D5C8D7; }
.purple-card { background-color: #8D7492; color: #FFFFFF !important; border-radius: 14px; padding: 14px; text-align: center; margin-bottom: 10px; }
.purple-card div { color: #FFFFFF !important; }
.bold-title { font-family: 'Special Elite', monospace !important; font-weight: 900 !important; font-size: 19px !important; color: #2E1D33 !important; display: block; }
.sec-title { font-family: 'Special Elite', monospace !important; font-weight: 900 !important; font-size: 15px !important; color: #3A243F !important; margin-bottom: 8px; display: block; }
.pill { background-color: #F7F3EE; border-radius: 12px; padding: 10px 12px; margin-bottom: 8px; border: 1px solid #DDD3DE; font-size: 12px; }
.stDownloadButton > button {
    background-color: #7D6283 !important;
    color: #FFFFFF !important;
    border-radius: 12px !important;
    font-family: 'Courier Prime', monospace !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    width: 100%;
    padding: 10px 16px !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_termografia_v2.pkl"), joblib.load("escalador_termografia_v2.pkl")

modelo, escalador = cargar_modelo()

def leer_estudio(archivo):
    nombre = archivo.name.lower()
    data = archivo.read()
    if nombre.endswith('.pdf'):
        doc = fitz.open(stream=data, filetype="pdf")
        pix = doc.load_page(0).get_pixmap(dpi=200)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
        return bgr, "PDF CLÍNICO"
    elif nombre.endswith('.dcm'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
            tmp.write(data)
            p = tmp.name
        ds = pydicom.dcmread(p)
        os.remove(p)
        arr = ds.pixel_array.astype(float)
        norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        bgr = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR) if len(norm.shape) == 2 else cv2.cvtColor(norm, cv2.COLOR_RGB2BGR)
        return bgr, "DICOM (.DCM)"
    else:
        arr = np.asarray(bytearray(data), dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return bgr, f"IMAGEN (.{nombre.split('.')[-1].upper()})"

def extraer_mapa_temperatura_flir(img_bgr):
    gris_temp = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gris_temp > 20).astype(np.uint8)
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        if w > 50 and h > 50:
            img_bgr = img_bgr[y:y+h, x:x+w]

    alto, ancho = img_bgr.shape[:2]
    # Recortar textos superiores FLIR (14%), barra inferior FLIR (14%) y escala lateral derecha (14%)
    recorte_bgr = img_bgr[int(alto * 0.14):int(alto * 0.86), int(ancho * 0.14):int(ancho * 0.84)]

    b, g, r = cv2.split(recorte_bgr)
    diferencia_color = float(np.mean(np.abs(r.astype(float) - b.astype(float))))

    if diferencia_color > 18.0:
        hsv = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        hue_norm = h_ch.astype(float)
        hue_norm[hue_norm > 145] = 0.0
        mapa_calor = np.clip(255.0 - (hue_norm * 2.15), 0, 255)
        mascara_blanco = (s_ch < 80) & (v_ch > 185)
        mapa_calor[mascara_blanco] = 255.0
        mapa_calor[v_ch < 45] = 0.0
        img_termica = mapa_calor.astype(np.uint8)
    else:
        gris = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2GRAY)
        img_termica = (255 - gris) if np.mean(gris) > 125 else gris

    return img_termica

def analizar_termografia(img_bgr):
    img_calor = extraer_mapa_temperatura_flir(img_bgr)
    img_128 = cv2.resize(img_calor, (128, 128))
    suave = cv2.GaussianBlur(img_128, (3, 3), 0)

    t_med = float(np.mean(suave))
    t_max = float(np.max(suave))
    t_min = float(np.min(suave))
    t_std = float(np.std(suave))
    p75 = float(np.percentile(suave, 75))
    p90 = float(np.percentile(suave, 90))
    p95 = float(np.percentile(suave, 95))
    ratio = float(np.sum(suave > 215) / suave.size)

    # Región mamaria glandular (excluyendo pliegue inframamario inferior y cuello)
    glandula = suave[28:92, 16:112]
    mitad = glandula.shape[1] // 2
    izq, der = glandula[:, :mitad], np.fliplr(glandula[:, mitad:])

    # 1. Asimetría píxel a píxel estructural entre mama izquierda y derecha
    dif_estructural = float(np.mean(np.abs(izq.astype(float) - der.astype(float))))
    asim_med = float(abs(np.mean(izq) - np.mean(der)))
    asim_max = float(abs(np.percentile(izq, 95) - np.percentile(der, 95)))

    # 2. Actividad vascular / focos hipertermicos intraglandulares (redes rojas/blancas)
    hiper_glandula = float(np.sum(glandula > 218) / glandula.size)
    laplaciano = float(np.var(cv2.Laplacian(glandula, cv2.CV_64F)))

    sx = cv2.Sobel(suave, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(suave, cv2.CV_64F, 0, 1, ksize=3)
    grad = float(np.mean(np.sqrt(sx**2 + sy**2)))
    skw = float(skew(suave.flatten()))

    cols = ['Temp_Media','Temp_Max','Temp_Min','Desv_Estandar','Percentil_75','Percentil_90','Percentil_95','Ratio_Puntos_Calientes','Asimetria_Bilateral_Media','Asimetria_Bilateral_Max','Gradiente_Vascular','Skewness_Termico']
    df = pd.DataFrame([[t_med, t_max, t_min, t_std, p75, p90, p95, ratio, asim_med, asim_max, grad, skw]], columns=cols)

    prob_svm = float(modelo.predict_proba(escalador.transform(df))[0][1])

    # Combinación calibrada con patrones FLIR reales (asimetría estructural + vascularización intraglandular)
    score_asim = min(1.0, (dif_estructural / 42.0) * 0.55 + (asim_med / 22.0) * 0.45)
    score_vasc = min(1.0, (hiper_glandula / 0.38) * 0.60 + (laplaciano / 950.0) * 0.40)
    score_anatomico = 0.55 * score_asim + 0.45 * score_vasc

    riesgo_final = min(0.985, max(0.04, 0.30 * prob_svm + 0.70 * score_anatomico))

    return df, riesgo_final, suave

def crear_pdf(paciente, fmt, anomalia, riesgo, df_bio, img_orig_bgr, img_c):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(141, 116, 146)
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Courier", "B", 15)
    pdf.cell(0, 14, "INFORME DE EVALUACION TERMOGRAFICA MAMARIA", ln=True, align="C")
    pdf.set_font("Courier", "", 9)
    pdf.cell(0, 6, "SISTEMA DE APOYO PREVENTIVO (MODELO SVM + BIOMARCADORES DMR-IR)", ln=True, align="C")
    pdf.ln(12)
    pdf.set_text_color(46, 29, 51)
    pdf.set_font("Courier", "B", 10)
    pdf.cell(0, 7, f"PACIENTE: {paciente.upper() if paciente else 'NO ESPECIFICADO'} | FECHA: {datetime.now().strftime('%d/%m/%Y')}", ln=True)
    if anomalia:
        pdf.set_fill_color(255, 224, 230)
        pdf.set_text_color(160, 20, 40)
        txt = f"RESULTADO: ALERTA DE ANOMALIA TERMICA DETECTADA (PROBABILIDAD: {riesgo*100:.1f}%)"
    else:
        pdf.set_fill_color(232, 245, 233)
        pdf.set_text_color(27, 94, 32)
        txt = f"RESULTADO: NO ENFERMA | POSIBILIDAD LATENTE DE RIESGO: {riesgo*100:.1f}%"
    pdf.multi_cell(0, 8, txt, fill=True)
    pdf.ln(4)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tg, tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tc:
        cv2.imwrite(tg.name, img_orig_bgr)
        cv2.imwrite(tc.name, img_c)
        pg, pc = tg.name, tc.name
    y = pdf.get_y()
    pdf.image(pg, x=22, y=y, w=65)
    pdf.image(pc, x=122, y=y, w=65)
    pdf.ln(68)
    os.remove(pg)
    os.remove(pc)
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Courier", "", 9)
    for col in df_bio.columns:
        pdf.cell(130, 6, f" - {col.upper()}", border=1)
        pdf.cell(50, 6, f"{df_bio.iloc[0][col]:.3f}", border=1, ln=True, align="R")
    return bytes(pdf.output())

st.sidebar.markdown("<div style='text-align:center; padding:8px;'><b style='font-size:18px;'>PINKTHERMO</b><br><span style='font-size:11px;'>PANEL DE DIAGNÓSTICO</span></div>", unsafe_allow_html=True)
nombre_pac = st.sidebar.text_input("NOMBRE DE LA PACIENTE:", placeholder="EJ. SOFIA THOMPSON")
archivo = st.sidebar.file_uploader("SUBIR ESTUDIO TERMOGRÁFICO:", type=["jpg","jpeg","png","bmp","tif","tiff","pdf","dcm"])
paleta = st.sidebar.selectbox("PALETA TÉRMICA:", ["JET (MÉDICO CLÁSICO)", "INFERNO (ALTO CONTRASTE)", "HOT (CÁLIDO)", "VIRIDIS (SUAVE)"])
umbral = st.sidebar.slider("UMBRAL DE ALERTA CLÍNICA (%):", 35, 65, 45, 1) / 100.0
st.sidebar.markdown("<div class='pill'><b>AUTOEXPLORACIÓN:</b> REALÍZALA 7 DÍAS DESPUÉS DE TU PERIODO.</div>", unsafe_allow_html=True)

st.markdown(f"<div class='main-banner'><span class='bold-title'>HOLA, {nombre_pac if nombre_pac else 'BIENVENIDA'} — EVALUACIÓN PREVENTIVA MAMARIA</span><span>SISTEMA DE DETECCIÓN TEMPRANA Y ESTRATIFICACIÓN DE RIESGO (SVM + DMR-IR)</span></div>", unsafe_allow_html=True)

if archivo is not None:
    img_bgr, fmt = leer_estudio(archivo)
    if img_bgr is not None:
        df_bio, riesgo, img_cal = analizar_termografia(img_bgr)
        cmap = cv2.COLORMAP_JET if "JET" in paleta else (cv2.COLORMAP_INFERNO if "INFERNO" in paleta else (cv2.COLORMAP_HOT if "HOT" in paleta else cv2.COLORMAP_VIRIDIS))
        img_color = cv2.applyColorMap(img_cal, cmap)
        anomalia = riesgo >= umbral

        c1, c2, c3 = st.columns([1.2, 1.2, 0.9])
        with c1:
            st.markdown("<div class='dash-card'><span class='sec-title'>1. ESTUDIO ORIGINAL</span>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown(f"<div class='pill'><b>FORMATO:</b> {fmt}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='dash-card'><span class='sec-title'>2. MAPA TÉRMICO DECODIFICADO</span>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown("<div class='pill'><b>ESCALA:</b> ROJO = MAYOR TEMPERATURA</div></div>", unsafe_allow_html=True)
        with c3:
            sim = max(5, int(round(100 - df_bio['Asimetria_Bilateral_Media'].values[0] * 2.0)))
            st.markdown(f"""
            <div class='dash-card'><span class='sec-title'>PROBABILIDADES</span>
                <div class='purple-card'><div>POSIBILIDAD DE ENFERMEDAD</div><div style='font-size:26px; font-weight:900;'>{riesgo*100:.1f}%</div></div>
                <div class='purple-card'><div>ESTADO SIN ANOMALÍA</div><div style='font-size:26px; font-weight:900;'>{(1-riesgo)*100:.1f}%</div></div>
                <div class='purple-card'><div>SIMETRÍA BILATERAL</div><div style='font-size:22px; font-weight:900;'>{sim}%</div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='dash-card'><span class='sec-title'>DICTAMEN DEL ANÁLISIS COMPUTACIONAL</span>", unsafe_allow_html=True)
        cr, cb = st.columns([1.6, 1.0])
        with cr:
            if anomalia:
                st.error(f"**ALERTA TÉRMICA: ANOMALÍA DETECTADA (PROBABILIDAD DE ENFERMEDAD: {riesgo*100:.1f}%)**\n\nSE DETECTARON REDES VASCULARES HIPERTÉRMICAS Y/O ASIMETRÍA TÉRMICA SIGNIFICATIVA ENTRE AMBAS MAMAS. ACUDE A VALORACIÓN MÉDICA.")
            else:
                nivel = "RIESGO PREVENTIVO MODERADO-BAJO (VIGILANCIA)" if riesgo >= 0.24 else "RIESGO PREVENTIVO MÍNIMO (RUTINARIO)"
                st.success(f"**DIAGNÓSTICO ACTUAL: NO ENFERMA / NORMAL (CONFIANZA: {(1-riesgo)*100:.1f}%)**\n\n**POSIBILIDAD ESTIMADA DE DESARROLLAR ENFERMEDAD: {riesgo*100:.1f}% — {nivel}**")
        with cb:
            st.download_button("DESCARGAR INFORME CLÍNICO PDF", crear_pdf(nombre_pac, fmt, anomalia, riesgo, df_bio, img_bgr, img_color), f"INFORME_{datetime.now().strftime('%Y%m%d')}.pdf", "application/pdf")
        st.markdown("</div>", unsafe_allow_html=True)
        with st.expander("VER TABLA DE LOS 12 BIOMARCADORES"):
            st.dataframe(df_bio.round(3), use_container_width=True)
else:
    ca, cb, cc = st.columns([1.1, 1.1, 0.9])
    with ca:
        st.markdown("<div class='dash-card'><span class='sec-title'>SEÑALES DE ALERTA</span><div class='pill'><b>01. BULTOS:</b> ENGROSAMIENTO EN MAMA O AXILA.</div><div class='pill'><b>02. PIEL:</b> HOYUELOS, ENROJECIMIENTO O PIEL DE NARANJA.</div><div class='pill'><b>03. ASIMETRÍA:</b> CAMBIO REPENTINO DE TAMAÑO O FORMA.</div></div>", unsafe_allow_html=True)
    with cb:
        st.markdown("<div class='dash-card'><span class='sec-title'>GUÍA DE PREVENCIÓN</span><div class='pill'><b>DETECCIÓN TEMPRANA:</b> REALIZA TU AUTOEXPLORACIÓN MENSUAL Y SUBE TU ESTUDIO TERMOGRÁFICO EN EL PANEL IZQUIERDO PARA EVALUAR TU RIESGO.</div></div>", unsafe_allow_html=True)
    with cc:
        st.markdown("<div class='dash-card'><span class='sec-title'>IMPACTO CLÍNICO</span><div class='purple-card'><div>CURACIÓN TEMPRANA</div><div style='font-size:28px; font-weight:900;'>95%</div></div><div class='purple-card'><div>RADIACIÓN INVASIVA</div><div style='font-size:28px; font-weight:900;'>0%</div></div></div>", unsafe_allow_html=True)
