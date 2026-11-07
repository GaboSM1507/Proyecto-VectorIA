import pandas as pd
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Configuración de la Aplicación
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Vector-IA | Control Epidemiológico",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://127.0.0.1:8000"

# -----------------------------------------------------------------------------
# 2. Barra Lateral (Sidebar)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/mosquito.png", width=70)
    st.title("Vector-IA System")
    st.caption("Sistema Inteligente de Predicción Epidemiológica")
    st.divider()

    st.subheader("🌐 Estado del Backend")
    try:
        health_res = requests.get(f"{API_BASE_URL}/", timeout=3)
        if health_res.status_code == 200:
            data_status = health_res.json()
            ann_st = "🟢 Listo" if data_status.get("ann_ready") else "🔴 No disponible"
            rnn_st = "🟢 Listo" if data_status.get("rnn_ready") else "🟠 Modo Simulación"
            st.success("API Online")
            st.write(f"**ANN (Acciones):** {ann_st}")
            st.write(f"**RNN (Casos):** {rnn_st}")
        else:
            st.error(f"API con errores ({health_res.status_code})")
    except requests.exceptions.ConnectionError:
        st.error("API Offline")
        st.caption("Ejecute: `uvicorn api.main:app --reload --port 8000`")

    st.divider()
    st.markdown("**Integrantes del Proyecto:**")
    st.markdown("- Gabriel (Arquitectura & Backend)")
    st.markdown("- Jeremy (RNN Model)")

# -----------------------------------------------------------------------------
# 3. Encabezado y Pestañas
# -----------------------------------------------------------------------------
st.title("Vector-IA: Sistema de Análisis y Control Epidemiológico")
st.markdown("Plataforma analítica basada en Deep Learning para la evaluación y proyección de brotes de Dengue.")

# Aviso ético fijo (pedido en el enunciado): visible en todas las pestañas
st.warning("⚠️ **Aviso:** Vector-IA es una herramienta de apoyo a la vigilancia epidemiológica. "
           "**No diagnostica dengue en personas.**")

tab_overview, tab_ann, tab_rnn = st.tabs([
    "📋 Resumen del Proyecto",
    "🧠 Recomendación de Acción (ANN)",
    "📈 Pronóstico Temporal (RNN)"
])

# -----------------------------------------------------------------------------
# PESTAÑA 1: Resumen
# -----------------------------------------------------------------------------
with tab_overview:
    st.header("Propósito de Vector-IA")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.write(
            "Vector-IA integra modelos de Redes Neuronales Artificiales (ANN) y Redes Neuronales Recurrentes (RNN) "
            "para transformar datos climáticos en alertas tempranas utilizables por equipos de salud pública."
        )
        st.markdown("""
        * **Modelo ANN (Clasificación Densa):** Recomienda una de 4 acciones: **alerta sanitaria**, **fumigación focalizada**, **eliminación de criaderos** o **monitoreo normal**.
        * **Modelo RNN (Series Temporales):** Estima la evolución temporal de contagios para las próximas semanas.
        * **Arquitectura Desacoplada:** Backend independiente en **FastAPI** (`api/main.py`) consumido por **Streamlit** (`app/Home.py`).
        """)
    with col_b:
        st.info("""
        **Ficha Técnica del Proyecto:**
        * **Backend:** FastAPI + Uvicorn
        * **Frontend:** Streamlit Wide Layout
        * **Modelos:** ANN (Softmax) & RNN/LSTM
        """)

# -----------------------------------------------------------------------------
# PESTAÑA 2: Recomendación de Acción (ANN)
# -----------------------------------------------------------------------------
with tab_ann:
    st.header("Recomendación de Acción Preventiva")
    st.caption("Mueva los controles y vea cómo cambia la recomendación (escenarios de «¿qué pasaría si?»).")

    # Datos de entrada: cada control es un dato que la ANN necesita
    col1, col2, col3 = st.columns(3)
    with col1:
        casos = st.slider("Casos predichos (próximas 4 semanas)", 0, 5000, 339)
        umbral = st.slider("Umbral endémico (casos)", 100, 3000, 282)
    with col2:
        lluvia = st.slider("Lluvia de los últimos 14 días (mm)", 0.0, 900.0, 17.0, step=1.0)
        temp_max = st.slider("Temperatura máxima (°C)", 20.0, 35.0, 27.1, step=0.1)
    with col3:
        temp_min = st.slider("Temperatura mínima (°C)", 15.0, 26.0, 21.0, step=0.1)
        humedad = st.slider("Humedad relativa (%)", 50.0, 100.0, 85.0, step=1.0)
    aegypti = st.checkbox("Se detectó Aedes aegypti en la foto (resultado de la CNN)", value=True)

    datos = {
        "casos_pred_4sem": casos,
        "casos_umbral_endemico": umbral,
        "lluvia_14d_mm": lluvia,
        "temp_max": temp_max,
        "temp_min": temp_min,
        "humedad": humedad,
        "aegypti_detectado": int(aegypti),
    }

    # Se consulta a la API cada vez que cambia un control
    try:
        res = requests.post(f"{API_BASE_URL}/predict/accion", json=datos, timeout=10)
        if res.status_code == 200:
            r = res.json()
            st.divider()

            # Un color según la gravedad de la acción
            colores = {
                "alerta_sanitaria": st.error,
                "fumigacion_focalizada": st.warning,
                "eliminacion_criaderos": st.warning,
                "monitoreo_normal": st.success,
            }
            nombre = r["accion"].replace("_", " ").capitalize()
            colores[r["accion"]](f"### {nombre}\n\n{r['motivo']}")

            m1, m2 = st.columns([1, 2])
            with m1:
                st.metric("Confianza de la red", f"{r['confianza_pct']}%")
            with m2:
                st.caption("Probabilidad (%) que la red asigna a cada acción")
                st.bar_chart(pd.Series(r["probabilidades"]))
        else:
            st.error(f"Error devuelto por la API ({res.status_code}): {res.text}")
    except Exception as e:
        st.error(f"Error al conectar con la API: {e}")

# -----------------------------------------------------------------------------
# PESTAÑA 3: Pronóstico Temporal (RNN)
# -----------------------------------------------------------------------------
with tab_rnn:
    st.header("Pronóstico Temporal de Casos (Modelo Recurrente RNN)")
    st.caption("Ingrese la tendencia de los casos observados en las últimas 4 semanas para proyectar la curva futura.")

    with st.form(key="form_rnn_forecast"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            w1 = st.number_input("Casos Semana t-3", min_value=0.0, value=15.0)
        with c2:
            w2 = st.number_input("Casos Semana t-2", min_value=0.0, value=22.0)
        with c3:
            w3 = st.number_input("Casos Semana t-1", min_value=0.0, value=35.0)
        with c4:
            w4 = st.number_input("Casos Semana Actual (t)", min_value=0.0, value=48.0)

        horizonte = st.slider("Semanas a proyectar hacia el futuro", min_value=1, max_value=8, value=4)
        btn_proyectar = st.form_submit_button(label="📈 Generar Pronóstico Temporal", use_container_width=True)

    if btn_proyectar:
        payload_rnn = {
            "casos_ultimas_4_semanas": [w1, w2, w3, w4],
            "semanas_a_proyectar": horizonte
        }

        try:
            with st.spinner("Procesando proyección en la API..."):
                res_rnn = requests.post(f"{API_BASE_URL}/predict/rnn", json=payload_rnn, timeout=10)

            if res_rnn.status_code == 200:
                data_rnn = res_rnn.json()

                st.divider()
                st.subheader("Resultados de la Proyección Epidemiológica")

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Casos Totales Estimados (Periodo)", f"{data_rnn['total_estimado_periodo']} casos")
                with m2:
                    st.metric("Tendencia General", data_rnn['tendencia_general'])
                with m3:
                    st.metric("Horizonte de Predicción", f"{horizonte} semanas")

                # Construir DataFrame para el gráfico
                hist_labels = ["Semana t-3", "Semana t-2", "Semana t-1", "Semana t (Actual)"]
                hist_values = data_rnn['historico_ingresado']

                proj_labels = data_rnn['semanas_proyectadas']
                proj_values = data_rnn['pronostico_casos']

                df_chart = pd.DataFrame({
                    "Periodo": hist_labels + proj_labels,
                    "Casos": hist_values + proj_values,
                    "Tipo": ["Histórico Real"] * 4 + ["Proyección RNN"] * horizonte
                }).set_index("Periodo")

                st.subheader("📉 Curva Epidemiológica (Histórico + Proyección)")
                st.line_chart(df_chart["Casos"])

                with st.expander("Ver desglose detallado por semana"):
                    df_tabla = pd.DataFrame({
                        "Semana Proyectada": proj_labels,
                        "Casos Estimados": proj_values
                    })
                    st.dataframe(df_tabla, use_container_width=True)

            else:
                st.error(f"Error devuelto por la API ({res_rnn.status_code}): {res_rnn.text}")
        except Exception as ex:
            st.error(f"Error de conexión con la API: {ex}")