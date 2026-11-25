import streamlit as st
import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew
from fpdf import FPDF
import fitz  # PyMuPDF para leer PDFs médicos
import pydicom  # Para leer estudios DICOM (.dcm)
import tempfile
import os
from datetime import datetime

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTÉTICA DOSMILERA (ROSA PASTEL + PUNTITOS BLANCOS)
# ==============================================================================
st.set_page_config(
    page_title="PinkThermo - Evaluación Termográfica Mamaria",
    layout="wide"
)

st.markdown("""
<style>
    /* Fondo rosa pastel con puntitos blancos estilo dosmilero */
    .stApp {
        background-color: #FFD6E8;
        background-image: radial-gradient(#FFFFFF 2.5px, transparent 2.5px);
        background-size: 24px 24px;
        font-family: Tahoma, Verdana, "MS Sans Serif", sans-serif !important;
        color: #4A154B;
        font-size: 13px;
    }

    /* Barra lateral */
    [data-testid="stSidebar"] {
        background-color: #FFE8F2;
        border-right: 3px solid #F48FB1;
    }

    /* Ventanas limpias estilo dosmilero */
    .y2k-window {
        background-color: #FFFFFF;
        border: 2px solid #E06D9E;
        box-shadow: 4px 4px 0px rgba(224, 109, 158, 0.35);
        margin-bottom: 16px;
    }

    /* Barra de título en degradado rosa pastel y lila */
    .y2k-titlebar {
        background: linear-gradient(90deg, #FF85B3 0%, #C896FF 100%);
        color: #FFFFFF;
        font-family: Tahoma, Verdana, sans-serif;
        font-weight: bold;
        font-size: 13px;
        padding: 7px 12px;
        border-bottom: 2px solid #E06D9E;
        letter-spacing: 0.3px;
    }

    /* Contenido dentro de la ventana */
    .y2k-body {
        padding: 16px;
        background-color: #FFFFFF;
        font-family: Tahoma, Verdana, sans-serif;
    }

    /* Cajitas de notas e información preventiva */
    .note-box {
        background-color: #FFF5F9;
        border: 1px solid #F48FB1;
        border-left: 5px solid #D84B8A;
        padding: 12px;
        margin-bottom: 10px;
        font-family: Tahoma, Verdana, sans-serif;
    }
    .note-title {
        color: #AD1457;
        font-weight: bold;
        font-size: 13px;
        margin-bottom: 5px;
    }

    /* Botón de descarga estilo dosmilero */
    .stDownloadButton > button {
        background: linear-gradient(180deg, #FF94C2 0%, #E05697 100%) !important;
        color: #FFFFFF !important;
        border: 2px outset #FFD6E8 !important;
        border-radius: 4px !important;
        padding: 10px 20px !important;
        font-family: Tahoma, Verdana, sans-serif !important;
        font-weight: bold !important;
        font-size: 13px !important;
        width: 100%;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(180deg, #E05697 0%, #C2185B 100%) !important;
    }

    /* Tarjetas de métricas */
    [data-testid="stMetric"] {
        background-color: #FFF7FB;
        border: 2px solid #CE93D8;
        border-radius: 4px;
        padding: 10px;
    }

    h1, h2, h3, h4, p, li, label {
        font-family: Tahoma, Verdana, sans-serif !important;
    }
    h1, h2, h3, h4 {
        color: #8E24AA !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CARGA DE MODELO SVM OPTIMIZADO Y ESCALADOR Z-SCORE
# ==============================================================================
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelo_termografia_v2.pkl")
    escalador = joblib.load("escalador_termografia_v2.pkl")
    return modelo, escalador

modelo, escalador = cargar_modelo()

# ==============================================================================
# 3. LECTOR UNIVERSAL DE ESTUDIOS MÉDICOS (JPG, PNG, BMP, TIFF, PDF, DICOM)
# ==============================================================================
def leer_estudio_medico(archivo):
    nombre = archivo.name.lower()
    bytes_data = archivo.read()
    
    if nombre.endswith('.pdf'):
        doc = fitz.open(stream=bytes_data, filetype="pdf")
        pagina = doc.load_page(0)
        pix = pagina.get_pixmap(dpi=200)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img_gris = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
        else:
            img_gris = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        return img_gris, "Documento PDF Clínico (.pdf)"
        
    elif nombre.endswith('.dcm'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
            tmp.write(bytes_data)
            tmp_path = tmp.name
        ds = pydicom.dcmread(tmp_path)
        os.remove(tmp_path)
        arr = ds.pixel_array.astype(float)
        arr_norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        if len(arr_norm.shape) == 3:
            img_gris = cv2.cvtColor(arr_norm, cv2.COLOR_RGB2GRAY)
        else:
            img_gris = arr_norm
        return img_gris, "Imagen Médica DICOM (.dcm)"
        
    else:
        arr = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        img_gris = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        return img_gris, f"Imagen Digital (.{nombre.split('.')[-1].upper()})"

# ==============================================================================
# 4. EXTRACCIÓN DE LOS 12 BIOMARCADORES TÉRMICOS (DMR-IR)
# ==============================================================================
def extraer_biomarcadores_termicos(img_gris):
    img = cv2.resize(img_gris, (128, 128))
    img_suave = cv2.GaussianBlur(img, (3, 3), 0)
    
    t_media = np.mean(img_suave)
    t_max = np.max(img_suave)
    t_min = np.min(img_suave)
    t_std = np.std(img_suave)
    
    p75 = np.percentile(img_suave, 75)
    p90 = np.percentile(img_suave, 90)
    p95 = np.percentile(img_suave, 95)
    ratio_calor = np.sum(img_suave > p90) / img_suave.size
    
    mitad = img_suave.shape[1] // 2
    lado_izq = img_suave[:, :mitad]
    lado_der = img_suave[:, mitad:]
    asimetria_media = abs(np.mean(lado_izq) - np.mean(lado_der))
    asimetria_max = abs(np.max(lado_izq) - np.max(lado_der))
    
    sobelx = cv2.Sobel(img_suave, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_suave, cv2.CV_64F, 0, 1, ksize=3)
    gradiente_termico = np.mean(np.sqrt(sobelx**2 + sobely**2))
    
    asimetria_hist = float(skew(img_suave.flatten()))
    
    columnas = [
        'Temp_Media', 'Temp_Max', 'Temp_Min', 'Desv_Estandar',
        'Percentil_75', 'Percentil_90', 'Percentil_95', 'Ratio_Puntos_Calientes',
        'Asimetria_Bilateral_Media', 'Asimetria_Bilateral_Max', 'Gradiente_Vascular', 'Skewness_Termico'
    ]
    
    valores = [[
        t_media, t_max, t_min, t_std,
        p75, p90, p95, ratio_calor,
        asimetria_media, asimetria_max, gradiente_termico, asimetria_hist
    ]]
    
    return pd.DataFrame(valores, columns=columnas)

# ==============================================================================
# 5. GENERADOR DE INFORME CLÍNICO DESCARGABLE EN PDF
# ==============================================================================
def generar_pdf_informe(nombre_paciente, formato_origen, prediccion, probabilidad, df_bio, img_gris, img_color_bgr):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_fill_color(248, 187, 208)
    pdf.rect(0, 0, 210, 38, 'F')
    pdf.set_text_color(136, 14, 79)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 14, "INFORME DE EVALUACION TERMOGRAFICA MAMARIA", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Sistema Inteligente de Apoyo Preventivo (Modelo SVM Optimizado - Base DMR-IR)", ln=True, align="C")
    pdf.ln(14)
    
    pdf.set_text_color(60, 20, 60)
    pdf.set_font("Helvetica", "B", 11)
    fecha_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    pdf.cell(0, 7, f"Paciente / Identificador: {nombre_paciente if nombre_paciente else 'No especificado'}", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Fecha de analisis: {fecha_str}   |   Formato recibido: {formato_origen}", ln=True)
    pdf.ln(4)
    
    if prediccion == 1:
        pdf.set_fill_color(255, 224, 230)
        pdf.set_text_color(183, 28, 28)
        titulo_res = f"RESULTADO: ALERTA TERMICA DETECTADA (Probabilidad de anomalia: {probabilidad[1]*100:.1f}%)"
        explicacion = (
            "El analisis computacional identifico patrones de hipertermia localizada, asimetria bilateral "
            "o actividad vascular irregular. Se recomienda acudir a valoracion medica especializada "
            "(mastologia / ecografia / mamografia) para confirmar este hallazgo."
        )
    else:
        pdf.set_fill_color(232, 245, 233)
        pdf.set_text_color(27, 94, 32)
        titulo_res = f"RESULTADO: PATRON TERMICO NORMAL / SIMETRICO (Probabilidad normal: {probabilidad[0]*100:.1f}%)"
        explicacion = (
            "La distribucion de calor observada en la termografia muestra simetria bilateral y ausencia "
            "de focos hipertermicos sospechosos. Continua con tus autoexploraciones mensuales y chequeos anuales."
        )
        
    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(0, 8, titulo_res, fill=True)
    pdf.set_text_color(50, 50, 50)
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
    pdf.multi_cell(0, 6, explicacion)
    pdf.ln(5)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_g, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_c:
        cv2.imwrite(tmp_g.name, img_gris)
        cv2.imwrite(tmp_c.name, img_color_bgr)
        path_g, path_c = tmp_g.name, tmp_c.name
        
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(123, 31, 162)
    pdf.cell(95, 7, "Termografia Original", align="C")
    pdf.cell(95, 7, "Mapa Termico Analizado", ln=True, align="C")
    y_img = pdf.get_y()
    pdf.image(path_g, x=22, y=y_img, w=65)
    pdf.image(path_c, x=122, y=y_img, w=65)
    pdf.ln(68)
    
    os.remove(path_g)
    os.remove(path_c)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(123, 31, 162)
    pdf.cell(0, 8, "Desglose de los 12 Biomarcadores Termicos Extraidos:", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    
    fila_datos = df_bio.iloc[0]
    nombres_legibles = {
        'Temp_Media': 'Temperatura Media Global (0-255)',
        'Temp_Max': 'Temperatura Maxima Detectada',
        'Temp_Min': 'Temperatura Minima Detectada',
        'Desv_Estandar': 'Heterogeneidad Termica (Desv. Est.)',
        'Percentil_75': 'Umbral Termico Percentil 75',
        'Percentil_90': 'Umbral Hipertermia Percentil 90',
        'Percentil_95': 'Foco Calor Extremo Percentil 95',
        'Ratio_Puntos_Calientes': 'Proporcion de Area Hipertermica',
        'Asimetria_Bilateral_Media': 'Asimetria Entre Ambas Mamas (Media)',
        'Asimetria_Bilateral_Max': 'Asimetria Entre Ambas Mamas (Maxima)',
        'Gradiente_Vascular': 'Gradiente de Vascularizacion (Sobel)',
        'Skewness_Termico': 'Asimetria Estadistica de Calor (Skewness)'
    }
    
    for col, desc in nombres_legibles.items():
        pdf.cell(130, 6, f"  - {desc}", border=1)
        pdf.cell(50, 6, f"{fila_datos[col]:.3f}", border=1, ln=True, align="R")
        
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 80, 120)
    pdf.multi_cell(0, 5, "Nota Medica Importante: Este informe fue generado mediante inteligencia artificial como apoyo preventivo y educativo. No sustituye el diagnostico de un medico especialista ni estudios de imagen clinica.")
    
    return bytes(pdf.output())

# ==============================================================================
# 6. INTERFAZ PRINCIPAL DE USUARIO
# ==============================================================================
st.markdown("""
<div class="y2k-window">
    <div class="y2k-titlebar">
        PinkThermo - Sistema de Evaluación Termográfica Mamaria
    </div>
    <div class="y2k-body">
        <h2 style="margin-top:0px; margin-bottom:4px; color:#C2185B !important;">Evaluación Preventiva por Termografía Mamaria</h2>
        <p style="margin:0px; color:#5A2A5E;">
            Plataforma de apoyo diagnóstico basada en Inteligencia Artificial (SVM Optimizado • Base de datos DMR-IR). 
            Sube tu estudio médico para visualizar sus mapas de temperatura y descargar tu informe detallado.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Barra lateral
st.sidebar.markdown("""
<div class="y2k-window">
    <div class="y2k-titlebar">Panel de Control del Estudio</div>
</div>
""", unsafe_allow_html=True)

nombre_usuario = st.sidebar.text_input("Nombre de la paciente (opcional para el informe):", placeholder="Ej. Sofía Martínez")
archivo_subido = st.sidebar.file_uploader(
    "📁 Subir estudio termográfico:",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "pdf", "dcm"],
    help="Formatos compatibles: JPG, PNG, TIFF, BMP, PDF clínico y DICOM (.dcm)."
)
paleta = st.sidebar.selectbox(
    "Mapa de temperatura:",
    ["JET (Estándar Médico)", "INFERNO (Alto Contraste)", "HOT (Térmico Cálido)", "VIRIDIS (Perceptual)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="note-box">
    <div class="note-title">Recordatorio de Salud</div>
    Realiza tu autoexploración mamaria una vez al mes, preferiblemente 7 días después de terminar tu periodo menstrual.
</div>
""", unsafe_allow_html=True)

# Cuerpo principal
if archivo_subido is not None:
    try:
        img_gris, formato_detectado = leer_estudio_medico(archivo_subido)
        
        if img_gris is None:
            st.error("No se pudo leer la imagen del archivo. Intenta subirlo en otro formato.")
        else:
            if "JET" in paleta:
                img_color = cv2.applyColorMap(img_gris, cv2.COLORMAP_JET)
            elif "INFERNO" in paleta:
                img_color = cv2.applyColorMap(img_gris, cv2.COLORMAP_INFERNO)
            elif "HOT" in paleta:
                img_color = cv2.applyColorMap(img_gris, cv2.COLORMAP_HOT)
            else:
                img_color = cv2.applyColorMap(img_gris, cv2.COLORMAP_VIRIDIS)
                
            img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

            df_biomarcadores = extraer_biomarcadores_termicos(img_gris)
            datos_escalados = escalador.transform(df_biomarcadores)
            prediccion = modelo.predict(datos_escalados)[0]
            probabilidad = modelo.predict_proba(datos_escalados)[0]

            st.info(f"Archivo cargado correctamente. Formato detectado: **{formato_detectado}**")

            # Visualización lado a lado
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="y2k-window">
                    <div class="y2k-titlebar">Termografía Original (Escala de Grises)</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(img_gris, use_container_width=True)
            with col2:
                st.markdown(f"""
                <div class="y2k-window">
                    <div class="y2k-titlebar">Mapa de Calor - {paleta}</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(img_color_rgb, use_container_width=True)

            # Sección de diagnóstico y descarga
            st.markdown("""
            <div class="y2k-window">
                <div class="y2k-titlebar">Resultados del Análisis Térmico</div>
            </div>
            """, unsafe_allow_html=True)

            col_res, col_exp = st.columns([1.1, 1.3])
            with col_res:
                if prediccion == 1:
                    st.error(
                        f"**ALERTA TÉRMICA: Patrón con Anomalía Detectada**\n\n"
                        f"Probabilidad estimada de anomalía: **{probabilidad[1]*100:.1f}%**\n\n"
                        "Se detectaron zonas de calor concentrado (hipertermia) o diferencia térmica entre ambos lados. "
                        "Se recomienda acudir a valoración médica especializada."
                    )
                else:
                    st.success(
                        f"**PATRÓN TÉRMICO NORMAL Y SIMÉTRICO**\n\n"
                        f"Nivel de confianza de normalidad: **{probabilidad[0]*100:.1f}%**\n\n"
                        "La distribución de temperatura es uniforme y simétrica, sin focos de calor inusuales."
                    )

                pdf_bytes = generar_pdf_informe(
                    nombre_usuario, formato_detectado, prediccion, probabilidad,
                    df_biomarcadores, img_gris, img_color
                )
                st.download_button(
                    label="⬇️ Descargar Informe Completo en PDF",
                    data=pdf_bytes,
                    file_name=f"Informe_Termografia_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )

            with col_exp:
                st.markdown("""
                <div class="note-box">
                    <div class="note-title">Interpretación Sencilla de los Resultados</div>
                    <ul>
                        <li><b>Medición de calor:</b> Cuando existe una lesión o crecimiento celular anormal, se forman nuevos vasos sanguíneos (angiogénesis) que elevan ligeramente la temperatura local.</li>
                        <li><b>Simetría bilateral:</b> Compara el lado izquierdo contra el derecho. En un tejido sano, ambos lados mantienen temperaturas muy similares.</li>
                        <li><b>Gradiente vascular:</b> Evalúa la intensidad de los cambios térmicos en los vasos sanguíneos superficiales.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Indicadores Térmicos Principales")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Asimetría Bilateral", f"{df_biomarcadores['Asimetria_Bilateral_Media'].values[0]:.2f}", help="Diferencia de calor promedio entre mama izquierda y derecha.")
            k2.metric("Gradiente Vascular", f"{df_biomarcadores['Gradiente_Vascular'].values[0]:.2f}", help="Intensidad de cambios térmicos en los vasos sanguíneos.")
            k3.metric("Foco Máximo (P95)", f"{df_biomarcadores['Percentil_95'].values[0]:.1f}", help="Nivel de temperatura en el 5% más cálido de la imagen.")
            k4.metric("Skewness Térmico", f"{df_biomarcadores['Skewness_Termico'].values[0]:.2f}", help="Asimetría estadística en la distribución del calor.")

            with st.expander("📄 Ver tabla completa de los 12 biomarcadores extraídos"):
                st.dataframe(df_biomarcadores.round(3), use_container_width=True)

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")

else:
    st.markdown("""
    <div class="y2k-window">
        <div class="y2k-titlebar">Instrucciones de Uso</div>
        <div class="y2k-body">
            <p style="font-size: 14px; font-weight: bold; color: #AD1457; margin-bottom: 6px;">
                Selecciona tu estudio termográfico en el menú de la izquierda para comenzar el análisis.
            </p>
            <p style="margin: 0;">
                La plataforma acepta archivos en formato <b>JPG, PNG, TIFF, BMP, PDF médico y DICOM (.dcm)</b>. 
                Al finalizar podrás descargar tu <b>Informe Completo en PDF</b> con las imágenes y métricas detalladas.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="y2k-window">
        <div class="y2k-titlebar">Notas Importantes sobre Prevención y Cuidado Mamario</div>
        <div class="y2k-body">
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="note-box">
            <div class="note-title">1. Señales de Alerta</div>
            Presta atención a bultos nuevos, hundimientos en la piel, cambios de textura (piel de naranja), enrojecimiento o secreción anormal en el pezón.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="note-box">
            <div class="note-title">2. Sobre la Termografía</div>
            Es un estudio <b>100% no invasivo, sin radiación y sin dolor</b> que detecta patrones de calor y vascularización como herramienta de apoyo preventivo.
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="note-box">
            <div class="note-title">3. Detección Temprana</div>
            Detectar cualquier alteración a tiempo eleva las probabilidades de curación a más del <b>95%</b>. Complementa siempre tus estudios con valoración médica.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)