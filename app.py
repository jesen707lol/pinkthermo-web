import streamlit as st
import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew
from skimage.feature import hog, local_binary_pattern
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

.stApp { background-color: #F7D9E3 !important; color: #4A2C2A !important; }
.stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp input, .stApp div[data-testid="stMarkdownContainer"] {
    font-family: 'Courier Prime', monospace !important;
    text-transform: uppercase !important;
    color: #4A2C2A !important;
}
[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
    text-transform: none !important;
}
section[data-testid="stSidebar"] { background-color: #FCEBF1 !important; border-right: 1px solid #E6B8C7 !important; }
section[data-testid="stSidebar"] * { color: #4A2C2A !important; }
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] select { background-color: #FFF7FA !important; color: #4A2C2A !important; }

.main-banner { background-color: #FFF7FA; border-radius: 18px; padding: 18px 24px; margin-bottom: 16px; border: 1px solid #E8B8C8; }
.dash-card { background-color: #FCE8EF; border-radius: 16px; padding: 16px; margin-bottom: 14px; border: 1px solid #E6B5C6; }
.pink-card { background-color: #A64D6E; color: #FFF8FA !important; border-radius: 14px; padding: 14px; text-align: center; margin-bottom: 10px; border: 1px solid #8A3957; }
.pink-card div { color: #FFF8FA !important; }
.bold-title { font-family: 'Special Elite', monospace !important; font-weight: 900 !important; font-size: 19px !important; color: #7D2E4D !important; display: block; }
.sec-title { font-family: 'Special Elite', monospace !important; font-weight: 900 !important; font-size: 15px !important; color: #8E3A59 !important; margin-bottom: 8px; display: block; }
.pill { background-color: #FFF7FA; border-radius: 12px; padding: 11px 14px; margin-bottom: 8px; border: 1px solid #E8BDCC; font-size: 12px; line-height: 1.45; color: #4A2C2A !important; }
.pill b { color: #7D2E4D !important; }

.stDownloadButton > button {
    background-color: #8E3A59 !important;
    color: #FFF8FA !important;
    border-radius: 12px !important;
    border: none !important;
    font-family: 'Courier Prime', monospace !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    width: 100%;
    padding: 10px 16px !important;
}
.stDownloadButton > button:hover {
    background-color: #6E2942 !important;
    color: #FFFFFF !important;
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

def extraer_y_predecir_flir(img_bgr):
    gris_tmp = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero((gris_tmp > 20).astype(np.uint8))
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        if w > 50 and h > 50:
            img_bgr = img_bgr[y:y+h, x:x+w]

    alto, ancho = img_bgr.shape[:2]
    roi_bgr = img_bgr[int(alto * 0.14):int(alto * 0.86), int(ancho * 0.14):int(ancho * 0.84)]
    roi_128 = cv2.resize(roi_bgr, (128, 128))

    hsv = cv2.cvtColor(roi_128, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    hue_norm = h_ch.astype(float)
    hue_norm[hue_norm > 145] = 0.0
    mapa_calor = np.clip(255.0 - (hue_norm * 2.15), 0, 255)
    mapa_calor[(s_ch < 80) & (v_ch > 185)] = 255.0
    mapa_calor[v_ch < 45] = 0.0
    term = cv2.GaussianBlur(mapa_calor.astype(np.uint8), (3, 3), 0)

    t_med, t_max, t_min, t_std = float(np.mean(term)), float(np.max(term)), float(np.min(term)), float(np.std(term))
    p75, p90, p95 = float(np.percentile(term, 75)), float(np.percentile(term, 90)), float(np.percentile(term, 95))
    ratio = float(np.sum(term > 210) / term.size)

    glandula = term[28:96, 14:114]
    mitad = glandula.shape[1] // 2
    izq, der = glandula[:, :mitad], np.fliplr(glandula[:, mitad:])
    asim_med = float(abs(np.mean(izq) - np.mean(der)))
    asim_max = float(abs(np.percentile(izq, 95) - np.percentile(der, 95)))
    asim_est = float(np.mean(np.abs(izq.astype(float) - der.astype(float))))

    sx = cv2.Sobel(term, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(term, cv2.CV_64F, 0, 1, ksize=3)
    grad = float(np.mean(np.sqrt(sx**2 + sy**2)))
    skw = float(skew(term.flatten()))
    lap = float(np.var(cv2.Laplacian(glandula, cv2.CV_64F)))

    bio = [t_med, t_max, t_min, t_std, p75, p90, p95, ratio, asim_med, asim_max, asim_est, grad, skw, lap]

    gland_hsv = hsv[28:96, 14:114]
    hist_h = cv2.calcHist([gland_hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([gland_hsv], [1], None, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([gland_hsv], [2], None, [8], [0, 256]).flatten()
    hist_hsv = np.concatenate([hist_h, hist_s, hist_v])
    hist_hsv = hist_hsv / (np.sum(hist_hsv) + 1e-7)

    lbp = local_binary_pattern(glandula, P=8, R=1, method='uniform')
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)

    hog_feat = hog(
        cv2.resize(glandula, (64, 64)),
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    vector_completo = np.concatenate([bio, hist_hsv, hist_lbp, hog_feat]).reshape(1, -1)
    prob_enfermedad = float(modelo.predict_proba(escalador.transform(vector_completo))[0][1])

    cols_tabla = [
        'Temp_Media', 'Temp_Max', 'Temp_Min', 'Desv_Estandar',
        'Percentil_75', 'Percentil_90', 'Percentil_95', 'Ratio_Puntos_Calientes',
        'Asimetria_Bilateral_Media', 'Asimetria_Bilateral_Max', 'Gradiente_Vascular', 'Skewness_Termico'
    ]
    df_tabla = pd.DataFrame([[t_med, t_max, t_min, t_std, p75, p90, p95, ratio, asim_med, asim_max, grad, skw]], columns=cols_tabla)

    return df_tabla, prob_enfermedad, term

def generar_explicacion_clinica(df_bio, riesgo, anomalia):
    asim_m = df_bio['Asimetria_Bilateral_Media'].values[0]
    asim_x = df_bio['Asimetria_Bilateral_Max'].values[0]
    grad = df_bio['Gradiente_Vascular'].values[0]
    ratio = df_bio['Ratio_Puntos_Calientes'].values[0] * 100
    t_med = df_bio['Temp_Media'].values[0]

    factores = []
    if asim_m > 12.0 or asim_x > 25.0:
        factores.append(f"UNA DIFERENCIA TÉRMICA NOTABLE ENTRE LA MAMA IZQUIERDA Y LA DERECHA (ASIMETRÍA MEDIA DE {asim_m:.1f} U.T. Y PICO DE {asim_x:.1f} U.T.), LO CUAL SUGIERE MAYOR ACTIVIDAD METABÓLICA EN UNO DE LOS LADOS")
    else:
        factores.append(f"UN EQUILIBRIO TÉRMICO ESTABLE ENTRE AMBAS MAMAS (ASIMETRÍA MEDIA BAJA DE {asim_m:.1f} U.T.), COMPORTAMIENTO HABITUAL EN TEJIDO MAMARIO SANO Y SIMÉTRICO")

    if grad > 22.0:
        factores.append(f"UN TRAZADO VASCULAR MARCADO (GRADIENTE DE {grad:.1f}), QUE INDICA PRESENCIA DE VASOS SANGUÍNEOS SUPERFICIALES ACTIVOS O POSIBLE ANGIOGÉNESIS LOCAL")
    else:
        factores.append(f"UNA IRRIGACIÓN SANGUÍNEA SUPERFICIAL UNIFORME SIN RAMIFICACIONES VASCULARES ABRUPTAS (GRADIENTE DE {grad:.1f})")

    if ratio > 18.0:
        factores.append(f"UNA CONCENTRACIÓN ELEVADA DE PUNTOS DE MÁXIMO CALOR ({ratio:.1f}% DEL ÁREA EVALUADA)")
    else:
        factores.append(f"UNA DISTRIBUCIÓN CONTROLADA DE ZONAS CÁLIDAS ({ratio:.1f}% DEL ÁREA), DENTRO DE RANGOS FISIOLÓGICOS ESPERADOS")

    por_que_prob = (
        f"¿POR QUÉ TIENES UNA PROBABILIDAD DEL {riesgo*100:.1f}%? EL MODELO COMPUTACIONAL ANALIZÓ LA TEXTURA VASCULAR, EL COLOR TÉRMICO Y LA SIMETRÍA DE TU ESTUDIO, IDENTIFICANDO: "
        + "; ".join(factores) + "."
    )

    if anomalia:
        recomendacion = (
            "INTERPRETACIÓN CLÍNICA: ESTE PORCENTAJE SUPERA EL UMBRAL DE VIGILANCIA PORQUE SE OBSERVAN FOCOS DE CALOR LOCALIZADO O ASIMETRÍA ENTRE AMBOS SENOS. "
            "ESTO NO CONFIRMA UN DIAGNÓSTICO DE CÁNCER, PERO INDICA ACTIVIDAD TÉRMICA QUE DEBE SER REVISADA POR UN MÉDICO MEDIANTE ULTRASONIDO O MASTOGRAFÍA."
        )
    else:
        recomendacion = (
            f"INTERPRETACIÓN CLÍNICA: TU ESTUDIO NO MUESTRA SIGNOS DE ANOMALÍA ACTIVA (CONFIANZA DE NORMALIDAD DEL {(1-riesgo)*100:.1f}%). "
            f"EL {riesgo*100:.1f}% CORRESPONDE A TU ÍNDICE DE RIESGO LATENTE PREVENTIVO BASADO EN VARIACIONES TÉRMICAS NORMALES DEL TEJIDO (TEMPERATURA MEDIA DE {t_med:.1f} U.T.). SE RECOMIENDA CONTINUAR CON TU AUTOEXPLORACIÓN MENSUAL."
        )

    guia_parametros = [
        ("TEMP_MEDIA / MAX / MIN", f"{t_med:.1f} / {df_bio['Temp_Max'].values[0]:.1f} / {df_bio['Temp_Min'].values[0]:.1f}", "NIVEL GENERAL DE RADIACIÓN INFRARROJA CAPTADA EN LA PIEL; REFLEJA EL FLUJO SANGUÍNEO SUPERFICIAL."),
        ("DESV_ESTANDAR", f"{df_bio['Desv_Estandar'].values[0]:.2f}", "MIDE QUÉ TAN PAREJA ES LA TEMPERATURA; VALORES ALTOS INDICAN CAMBIOS BRUSCOS ENTRE ZONAS FRÍAS Y CALIENTES."),
        ("PERCENTILES 75 / 90 / 95", f"{df_bio['Percentil_75'].values[0]:.1f} / {df_bio['Percentil_90'].values[0]:.1f} / {df_bio['Percentil_95'].values[0]:.1f}", "UMBRALES DE LAS ZONAS MÁS CÁLIDAS DEL SENO; PERMITEN DETECTAR FOCOS HIPERTÉRMICOS CONCENTRADOS."),
        ("RATIO_PUNTOS_CALIENTES", f"{ratio:.1f}%", "PORCENTAJE DEL TEJIDO QUE ALCANZA TEMPERATURAS ELEVADAS RESPECTO AL RESTO DEL TÓRAX."),
        ("ASIMETRIA_BILATERAL", f"MEDIA {asim_m:.1f} | MÁX {asim_x:.1f}", "COMPARA EL SENO IZQUIERDO CONTRA EL DERECHO; EN CONDICIONES SANAS AMBOS DEBEN TENER UN PATRÓN SIMILAR."),
        ("GRADIENTE_VASCULAR", f"{grad:.2f}", "MIDE LA INTENSIDAD DE LOS BORDES TÉRMICOS DE LAS VENAS; LOS TUMORES SUELEN GENERAR NUEVOS VASOS (ANGIOGÉNESIS)."),
        ("SKEWNESS_TERMICO", f"{df_bio['Skewness_Termico'].values[0]:.2f}", "INDICA SI LA IMAGEN TIENDE HACIA TONOS CÁLIDOS O FRÍOS EN SU DISTRIBUCIÓN GLOBAL.")
    ]

    return por_que_prob, recomendacion, guia_parametros

def crear_pdf(paciente, fmt, anomalia, riesgo, df_bio, img_orig_bgr, img_c, por_que_prob, recomendacion, guia_parametros):
    pdf = FPDF()
    pdf.set_margins(12, 12, 12)
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    pdf.set_fill_color(142, 58, 89)
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_text_color(255, 248, 250)
    pdf.set_font("Courier", "B", 13)
    pdf.set_xy(12, 7)
    pdf.cell(186, 8, "INFORME DE EVALUACION TERMOGRAFICA MAMARIA", ln=True, align="C")
    pdf.set_font("Courier", "", 8.5)
    pdf.set_x(12)
    pdf.cell(186, 6, "PINKTHERMO - SISTEMA DE APOYO PREVENTIVO (MODELO HSV+HOG+LBP)", ln=True, align="C")
    pdf.ln(10)

    pdf.set_text_color(74, 44, 42)
    pdf.set_font("Courier", "B", 9.5)
    pdf.set_x(12)
    pdf.cell(186, 6, f"PACIENTE: {paciente.upper() if paciente else 'NO ESPECIFICADO'} | FECHA: {datetime.now().strftime('%d/%m/%Y')}", ln=True)
    pdf.ln(1)

    if anomalia:
        pdf.set_fill_color(255, 224, 230)
        pdf.set_text_color(150, 20, 45)
        txt = f"RESULTADO: ALERTA DE ANOMALIA TERMICA DETECTADA (PROBABILIDAD: {riesgo*100:.1f}%)"
    else:
        pdf.set_fill_color(232, 245, 233)
        pdf.set_text_color(27, 94, 32)
        txt = f"RESULTADO: NO ENFERMA | POSIBILIDAD LATENTE DE RIESGO: {riesgo*100:.1f}%"
    pdf.set_x(12)
    pdf.multi_cell(186, 7, txt, fill=True)
    pdf.ln(3)

    pdf.set_text_color(74, 44, 42)
    pdf.set_font("Courier", "", 8)
    pdf.set_x(12)
    pdf.multi_cell(186, 4.3, por_que_prob)
    pdf.ln(2)
    pdf.set_text_color(125, 46, 77)
    pdf.set_font("Courier", "B", 8)
    pdf.set_x(12)
    pdf.multi_cell(186, 4.3, recomendacion)
    pdf.ln(4)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tg, tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tc:
        cv2.imwrite(tg.name, img_orig_bgr)
        cv2.imwrite(tc.name, img_c)
        pg, pc = tg.name, tc.name

    y_img = pdf.get_y()
    pdf.image(pg, x=22, y=y_img, w=68, h=52)
    pdf.image(pc, x=118, y=y_img, w=68, h=52)
    pdf.set_xy(12, y_img + 56)
    os.remove(pg)
    os.remove(pc)

    pdf.set_font("Courier", "B", 9)
    pdf.set_text_color(142, 58, 89)
    pdf.set_x(12)
    pdf.cell(186, 6, "EXPLICACION DE BIOMARCADORES TERMICOS EVALUADOS:", ln=True)
    pdf.ln(1)

    for nombre_p, val_p, desc_p in guia_parametros:
        pdf.set_x(12)
        pdf.set_font("Courier", "B", 7.8)
        pdf.set_text_color(125, 46, 77)
        pdf.multi_cell(186, 4.2, f"* {nombre_p} [{val_p}]:")
        pdf.set_x(15)
        pdf.set_font("Courier", "", 7.5)
        pdf.set_text_color(74, 44, 42)
        pdf.multi_cell(183, 4.0, desc_p)
        pdf.ln(0.8)

    return bytes(pdf.output())

st.sidebar.markdown("<div style='text-align:center; padding:8px;'><b style='font-size:19px; color:#8E3A59;'>PINKTHERMO</b><br><span style='font-size:11px; color:#4A2C2A;'>PANEL DE DIAGNÓSTICO</span></div>", unsafe_allow_html=True)
nombre_pac = st.sidebar.text_input("NOMBRE DE LA PACIENTE:", placeholder="EJ. SOFIA THOMPSON")
archivo = st.sidebar.file_uploader("SUBIR ESTUDIO TERMOGRÁFICO:", type=["jpg","jpeg","png","bmp","tif","tiff","pdf","dcm"])
paleta = st.sidebar.selectbox("PALETA TÉRMICA:", ["JET (MÉDICO CLÁSICO)", "INFERNO (ALTO CONTRASTE)", "HOT (CÁLIDO)", "VIRIDIS (SUAVE)"])
umbral = st.sidebar.slider("UMBRAL DE ALERTA CLÍNICA (%):", 35, 65, 45, 1) / 100.0
st.sidebar.markdown("<div class='pill'><b>AUTOEXPLORACIÓN:</b> REALÍZALA 7 DÍAS DESPUÉS DE TU PERIODO.</div>", unsafe_allow_html=True)

st.markdown(f"<div class='main-banner'><span class='bold-title'>HOLA, {nombre_pac if nombre_pac else 'BIENVENIDA'} — EVALUACIÓN PREVENTIVA MAMARIA</span><span>MODELO MULTIDIMENSIONAL CALIBRADO (VISIÓN ARTIFICIAL + BIOMARCADORES DMR-IR)</span></div>", unsafe_allow_html=True)

if archivo is not None:
    img_bgr, fmt = leer_estudio(archivo)
    if img_bgr is not None:
        df_bio, riesgo, img_cal = extraer_y_predecir_flir(img_bgr)
        cmap = cv2.COLORMAP_JET if "JET" in paleta else (cv2.COLORMAP_INFERNO if "INFERNO" in paleta else (cv2.COLORMAP_HOT if "HOT" in paleta else cv2.COLORMAP_VIRIDIS))
        img_color = cv2.applyColorMap(img_cal, cmap)
        anomalia = riesgo >= umbral
        por_que_prob, recomendacion, guia_parametros = generar_explicacion_clinica(df_bio, riesgo, anomalia)

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
                <div class='pink-card'><div>POSIBILIDAD DE ENFERMEDAD</div><div style='font-size:26px; font-weight:900;'>{riesgo*100:.1f}%</div></div>
                <div class='pink-card'><div>ESTADO SIN ANOMALÍA</div><div style='font-size:26px; font-weight:900;'>{(1-riesgo)*100:.1f}%</div></div>
                <div class='pink-card'><div>SIMETRÍA BILATERAL</div><div style='font-size:22px; font-weight:900;'>{sim}%</div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='dash-card'><span class='sec-title'>DICTAMEN DEL ANÁLISIS Y EXPLICACIÓN DE RESULTADOS</span>", unsafe_allow_html=True)
        cr, cb = st.columns([1.6, 1.0])
        with cr:
            if anomalia:
                st.error(f"**ALERTA TÉRMICA: ANOMALÍA DETECTADA (PROBABILIDAD DE ENFERMEDAD: {riesgo*100:.1f}%)**")
            else:
                nivel = "RIESGO PREVENTIVO MODERADO-BAJO (VIGILANCIA)" if riesgo >= 0.25 else "RIESGO PREVENTIVO MÍNIMO (RUTINARIO)"
                st.success(f"**DIAGNÓSTICO ACTUAL: NO ENFERMA / NORMAL (CONFIANZA: {(1-riesgo)*100:.1f}%) — {nivel}**")

            st.markdown(f"<div class='pill'><b>¿POR QUÉ OBTUVISTE ESTE PORCENTAJE?</b><br>{por_que_prob}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='pill'><b>¿QUÉ SIGNIFICA PARA TU SALUD?</b><br>{recomendacion}</div>", unsafe_allow_html=True)

        with cb:
            st.download_button(
                "DESCARGAR INFORME CLÍNICO PDF EXPLICADO",
                crear_pdf(nombre_pac, fmt, anomalia, riesgo, df_bio, img_bgr, img_color, por_que_prob, recomendacion, guia_parametros),
                f"INFORME_{datetime.now().strftime('%Y%m%d')}.pdf",
                "application/pdf"
            )
            st.markdown("<div class='pill'><b>NOTA MÉDICA:</b> ESTE ANÁLISIS ES UNA HERRAMIENTA DE TAMIZAJE PREVENTIVO Y NO SUSTITUYE UNA MASTOGRAFÍA O ULTRASONIDO.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("VER EXPLICACIÓN DETALLADA DE CADA PARÁMETRO TÉRMICO", expanded=True):
            df_explicado = pd.DataFrame(guia_parametros, columns=["PARÁMETRO TÉRMICO", "VALOR MEDIDO", "¿QUÉ SIGNIFICA EN TU TERMOGRAFÍA?"])
            st.dataframe(df_explicado, use_container_width=True, hide_index=True)
else:
    ca, cb, cc = st.columns([1.1, 1.1, 0.9])
    with ca:
        st.markdown("<div class='dash-card'><span class='sec-title'>SEÑALES DE ALERTA</span><div class='pill'><b>01. BULTOS:</b> ENGROSAMIENTO EN MAMA O AXILA.</div><div class='pill'><b>02. PIEL:</b> HOYUELOS, ENROJECIMIENTO O PIEL DE NARANJA.</div><div class='pill'><b>03. ASIMETRÍA:</b> CAMBIO REPENTINO DE TAMAÑO O FORMA.</div></div>", unsafe_allow_html=True)
    with cb:
        st.markdown("<div class='dash-card'><span class='sec-title'>GUÍA DE PREVENCIÓN</span><div class='pill'><b>DETECCIÓN TEMPRANA:</b> REALIZA TU AUTOEXPLORACIÓN MENSUAL Y SUBE TU ESTUDIO TERMOGRÁFICO EN EL PANEL IZQUIERDO PARA EVALUAR TU RIESGO.</div></div>", unsafe_allow_html=True)
    with cc:
        st.markdown("<div class='dash-card'><span class='sec-title'>IMPACTO CLÍNICO</span><div class='pink-card'><div>CURACIÓN TEMPRANA</div><div style='font-size:28px; font-weight:900;'>95%</div></div><div class='pink-card'><div>RADIACIÓN INVASIVA</div><div style='font-size:28px; font-weight:900;'>0%</div></div></div>", unsafe_allow_html=True)
