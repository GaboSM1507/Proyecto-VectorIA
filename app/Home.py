
import os
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="VectorIA | Vigilancia del Dengue",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

API_BASE_URL = "http://127.0.0.1:8000"

RUTA_DENGUE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "dengue_procesado.csv"
)

RUTA_CANTONES = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "vectoria_casos_cantonal_anual_real.csv"
)


# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================

def obtener_estado_api():

    try:
        respuesta = requests.get(
            f"{API_BASE_URL}/",
            timeout=5
        )

        respuesta.raise_for_status()

        return respuesta.json()

    except requests.exceptions.RequestException:
        return None


@st.cache_data
def cargar_dengue():

    df = pd.read_csv(RUTA_DENGUE)

    df["fecha_inicio_semana"] = pd.to_datetime(
        df["fecha_inicio_semana"]
    )

    return df.sort_values(
        "fecha_inicio_semana"
    ).reset_index(drop=True)


@st.cache_data
def cargar_cantones():

    return pd.read_csv(RUTA_CANTONES)


def mostrar_error_api(respuesta):

    try:
        detalle = respuesta.json().get(
            "detail",
            respuesta.text
        )
    except Exception:
        detalle = respuesta.text

    st.error(f"Error de la API: {detalle}")


# ============================================================
# 3. ESTADO DEL BACKEND
# ============================================================

estado_api = obtener_estado_api()

with st.sidebar:

    st.image(
        "https://img.icons8.com/color/96/mosquito.png",
        width=75
    )

    st.title("VectorIA")
    st.caption("Sistema inteligente de vigilancia del dengue")

    st.divider()

    st.subheader("Estado del sistema")

    if estado_api:

        st.success("API conectada")

        st.write(
            "CNN:",
            "🟢 Disponible" if estado_api["cnn_ready"] else "🔴 No disponible"
        )

        st.write(
            "RNN:",
            "🟢 Disponible" if estado_api["rnn_ready"] else "🔴 No disponible"
        )

        st.write(
            "ANN:",
            "🟢 Disponible" if estado_api["ann_ready"] else "🔴 No disponible"
        )

    else:

        st.error("API desconectada")
        st.caption(
            "Ejecute: python -m uvicorn api.main:app --reload"
        )

    st.divider()

    st.markdown("**Integrantes del proyecto**")

    st.markdown("""
    - Gabriel: Arquitectura y backend
    - Jeremy: Modelo RNN
    - Jeriel: ANN, API y documentación
    """)


# ============================================================
# 4. ENCABEZADO
# ============================================================

st.title("🦟 VectorIA")
st.subheader(
    "Asistente Inteligente para la Vigilancia del Dengue en Costa Rica"
)

st.write(
    "Plataforma de inteligencia artificial que integra "
    "redes neuronales para identificar mosquitos, "
    "pronosticar casos de dengue y recomendar acciones preventivas."
)

st.warning(
    "⚠️ VectorIA es una herramienta de apoyo a la vigilancia "
    "epidemiológica. No diagnostica dengue en personas."
)


# ============================================================
# 5. PESTAÑAS
# ============================================================

tab_inicio, tab_cnn, tab_rnn, tab_ann, tab_mapa = st.tabs([

    "📋 Resumen",

    "🦟 Identificar mosquito (CNN)",

    "📈 Pronóstico dengue (RNN)",

    "🧠 Acciones preventivas (ANN)",

    "🗺️ Mapa por cantón"

])


# ============================================================
# 6. PESTAÑA INICIO
# ============================================================

with tab_inicio:

    st.header("Bienvenido a VectorIA")

    st.write(
        "El sistema integra tres modelos de inteligencia artificial "
        "para apoyar la vigilancia y prevención del dengue."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info("""
        **CNN**

        Identificación de especies de mosquitos mediante fotografías.
        """)

    with col2:

        st.info("""
        **RNN / LSTM**

        Predicción de los casos de dengue para las próximas cuatro semanas.
        """)

    with col3:

        st.info("""
        **ANN**

        Recomendación de acciones preventivas según los datos disponibles.
        """)

    st.divider()

    st.subheader("¿Cómo funciona?")

    st.markdown("""
    1. Se carga una fotografía para identificar el mosquito.
    2. Se utilizan los datos históricos y climáticos para pronosticar casos.
    3. La ANN analiza los indicadores y recomienda una acción preventiva.
    4. El usuario puede explorar la distribución histórica de casos por cantón.
    """)

    st.caption(
        "Fuentes: OPS/PAHO mediante OpenDengue y NASA POWER."
    )


# ============================================================
# 7. PESTAÑA CNN: IDENTIFICACIÓN DE MOSQUITOS
# ============================================================

with tab_cnn:

    st.header("Identificación de mosquitos")

    st.write(
        "Suba una fotografía para que MobileNetV2 "
        "identifique la especie del mosquito."
    )

    if not estado_api or not estado_api.get("cnn_ready"):

        st.error("El modelo CNN no está disponible.")

    else:

        archivo = st.file_uploader(
            "Seleccione una fotografía",
            type=["jpg", "jpeg", "png"],
            key="foto_mosquito"
        )

        if archivo is not None:

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    archivo,
                    caption="Fotografía seleccionada",
                    use_container_width=True
                )

            with col2:

                st.write("**Información de la imagen**")

                st.write("Nombre:", archivo.name)

                st.write(
                    "Formato:",
                    archivo.type
                )

                if st.button(
                    "🔍 Identificar mosquito",
                    use_container_width=True
                ):

                    try:

                        with st.spinner(
                            "Analizando fotografía con la CNN..."
                        ):

                            respuesta = requests.post(
                                f"{API_BASE_URL}/predict/mosquito",
                                files={
                                    "file": (
                                        archivo.name,
                                        archivo.getvalue(),
                                        archivo.type
                                    )
                                },
                                timeout=60
                            )

                        if respuesta.status_code == 200:

                            resultado = respuesta.json()

                            especie = resultado["especie"]

                            confianza = resultado["confianza_pct"]

                            st.session_state["especie_detectada"] = especie

                            st.session_state["cnn_aegypti"] = (
                                "aedes aegypti" in especie.lower()
                            )

                            st.session_state["cnn_resultado"] = resultado

                        else:

                            mostrar_error_api(respuesta)

                    except Exception as error:

                        st.error(
                            f"No se pudo conectar con la CNN: {error}"
                        )

        if "cnn_resultado" in st.session_state:

            resultado = st.session_state["cnn_resultado"]

            st.divider()

            st.subheader("Resultado de identificación")

            st.success(
                f"Especie identificada: {resultado['especie']}"
            )

            st.metric(
                "Confianza del modelo",
                f"{resultado['confianza_pct']}%"
            )

            st.write("Probabilidades por especie")

            probabilidades = pd.Series(
                resultado["probabilidades"]
            )

            st.bar_chart(probabilidades)

            st.caption(
                "La confianza representa la salida del clasificador, "
                "no una certeza científica de identificación."
            )


# ============================================================
# 8. PESTAÑA RNN: PRONÓSTICO DE DENGUE
# ============================================================

with tab_rnn:

    st.header("Pronóstico de casos de dengue")

    st.write(
        "El modelo SimpleRNN utiliza las últimas 26 semanas "
        "de casos y variables climáticas para pronosticar "
        "las próximas cuatro semanas."
    )

    if not estado_api or not estado_api.get("rnn_ready"):

        st.error("El modelo RNN no está disponible.")

    elif not os.path.exists(RUTA_DENGUE):

        st.error(
            "No se encontró el archivo dengue_procesado.csv."
        )

    else:

        try:

            df_dengue = cargar_dengue()

            if len(df_dengue) < 26:

                st.error(
                    "No existen 26 semanas disponibles."
                )

            else:

                ultimas = df_dengue.tail(26).copy()

                st.subheader("Datos históricos utilizados")

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Semanas históricas",
                        len(ultimas)
                    )

                with c2:

                    st.metric(
                        "Fecha inicial",
                        ultimas["fecha_inicio_semana"].iloc[0].strftime(
                            "%d/%m/%Y"
                        )
                    )

                with c3:

                    st.metric(
                        "Fecha final",
                        ultimas["fecha_inicio_semana"].iloc[-1].strftime(
                            "%d/%m/%Y"
                        )
                    )

                with st.expander(
                    "Consultar las 26 semanas utilizadas"
                ):

                    st.dataframe(
                        ultimas,
                        use_container_width=True
                    )

                if st.button(
                    "📈 Generar pronóstico de dengue",
                    use_container_width=True
                ):

                    semanas = []

                    for _, fila in ultimas.iterrows():

                        semanas.append({

                            "casos_dengue": float(
                                fila["casos_dengue"]
                            ),

                            "lluvia_mm": float(
                                fila["lluvia_mm"]
                            ),

                            "temp_max_c": float(
                                fila["temp_max_c"]
                            ),

                            "temp_min_c": float(
                                fila["temp_min_c"]
                            ),

                            "humedad_rel_pct": float(
                                fila["humedad_rel_pct"]
                            ),

                            "semana_epi": int(
                                fila["semana_epi"]
                            )

                        })

                    payload = {
                        "semanas": semanas
                    }

                    try:

                        with st.spinner(
                            "La red neuronal está generando el pronóstico..."
                        ):

                            respuesta = requests.post(
                                f"{API_BASE_URL}/predict/rnn",
                                json=payload,
                                timeout=60
                            )

                        if respuesta.status_code == 200:

                            resultado = respuesta.json()

                            st.session_state["rnn_resultado"] = resultado

                        else:

                            mostrar_error_api(respuesta)

                    except Exception as error:

                        st.error(
                            f"Error al generar el pronóstico: {error}"
                        )

                if "rnn_resultado" in st.session_state:

                    resultado = st.session_state["rnn_resultado"]

                    st.divider()

                    st.subheader("Resultados del pronóstico")

                    m1, m2, m3 = st.columns(3)

                    with m1:

                        st.metric(
                            "Total estimado",
                            f"{resultado['total_estimado_periodo']} casos"
                        )

                    with m2:

                        st.metric(
                            "Tendencia",
                            resultado["tendencia_general"]
                        )

                    with m3:

                        st.metric(
                            "Horizonte",
                            "4 semanas"
                        )

                    historicos = resultado["historico_ingresado"]

                    predicciones = resultado["pronostico_casos"]

                    fechas_historicas = ultimas[
                        "fecha_inicio_semana"
                    ].dt.strftime("%d/%m/%Y").tolist()

                    etiquetas_futuras = resultado[
                        "semanas_proyectadas"
                    ]

                    grafico = pd.DataFrame({

                        "Periodo": (
                            fechas_historicas + etiquetas_futuras
                        ),

                        "Casos": (
                            historicos + predicciones
                        )

                    })

                    st.subheader(
                        "Evolución histórica y pronóstico"
                    )

                    st.line_chart(
                        grafico.set_index("Periodo"),
                        use_container_width=True
                    )

                    st.subheader("Predicción por semana")

                    tabla = pd.DataFrame({

                        "Semana": etiquetas_futuras,

                        "Casos estimados": predicciones

                    })

                    st.dataframe(
                        tabla,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "Modelo utilizado: " +
                        resultado["modelo_utilizado"]
                    )

        except Exception as error:

            st.error(
                f"Error al cargar los datos: {error}"
            )


# ============================================================
# 9. PESTAÑA ANN: RECOMENDACIÓN PREVENTIVA
# ============================================================

with tab_ann:

    st.header("Recomendación de acción preventiva")

    st.write(
        "La ANN analiza los indicadores epidemiológicos y climáticos "
        "para recomendar una de cuatro acciones preventivas."
    )

    if not estado_api or not estado_api.get("ann_ready"):

        st.error("El modelo ANN no está disponible.")

    else:

        try:

            df_dengue = cargar_dengue()

            ultima = df_dengue.iloc[-1]

            ultimas_dos = df_dengue.tail(2)

            lluvia_14d = float(
                ultimas_dos["lluvia_mm"].sum()
            )

            temp_max = float(
                ultima["temp_max_c"]
            )

            temp_min = float(
                ultima["temp_min_c"]
            )

            humedad = float(
                ultima["humedad_rel_pct"]
            )

            st.subheader("Indicadores de entrada")

            if "rnn_resultado" in st.session_state:

                pronostico = st.session_state["rnn_resultado"]

                casos_predichos = float(
                    pronostico["total_estimado_periodo"]
                )

                st.success(
                    "Se utilizará el pronóstico generado por la RNN."
                )

            else:

                casos_predichos = 339.0

                st.info(
                    "Primero puedes ejecutar el pronóstico RNN. "
                    "Mientras tanto se utilizará un escenario de ejemplo."
                )

            st.metric(
                "Casos estimados para las próximas 4 semanas",
                f"{casos_predichos:.1f}"
            )

            st.caption(
                "Los indicadores climáticos se obtienen de las "
                "últimas semanas disponibles en el CSV procesado."
            )

            st.write(
                f"Lluvia de las últimas 2 semanas: {lluvia_14d:.1f} mm"
            )

            st.write(
                f"Temperatura máxima: {temp_max:.1f} °C"
            )

            st.write(
                f"Temperatura mínima: {temp_min:.1f} °C"
            )

            st.write(
                f"Humedad relativa: {humedad:.1f} %"
            )

            st.divider()

            umbral = st.number_input(
                "Umbral endémico de referencia",
                min_value=1.0,
                value=282.0,
                step=1.0
            )

            aegypti_detectado = st.checkbox(
                "Se detectó Aedes aegypti",
                value=st.session_state.get(
                    "cnn_aegypti",
                    False
                )
            )

            if "especie_detectada" in st.session_state:

                st.caption(
                    "Última identificación CNN: " +
                    st.session_state["especie_detectada"]
                )

            if st.button(
                "🧠 Generar recomendación preventiva",
                use_container_width=True
            ):

                datos = {

                    "casos_pred_4sem": casos_predichos,

                    "casos_umbral_endemico": float(umbral),

                    "lluvia_14d_mm": lluvia_14d,

                    "temp_max": temp_max,

                    "temp_min": temp_min,

                    "humedad": humedad,

                    "aegypti_detectado": int(
                        aegypti_detectado
                    )

                }

                try:

                    with st.spinner(
                        "La ANN está evaluando los indicadores..."
                    ):

                        respuesta = requests.post(
                            f"{API_BASE_URL}/predict/accion",
                            json=datos,
                            timeout=30
                        )

                    if respuesta.status_code == 200:

                        st.session_state["ann_resultado"] = (
                            respuesta.json()
                        )

                    else:

                        mostrar_error_api(respuesta)

                except Exception as error:

                    st.error(
                        f"Error al solicitar la recomendación: {error}"
                    )

            if "ann_resultado" in st.session_state:

                resultado = st.session_state["ann_resultado"]

                st.divider()

                st.subheader("Resultado de la ANN")

                colores = {

                    "alerta_sanitaria": st.error,

                    "fumigacion_focalizada": st.warning,

                    "eliminacion_criaderos": st.warning,

                    "monitoreo_normal": st.success

                }

                accion = resultado["accion"]

                nombre = accion.replace(
                    "_",
                    " "
                ).capitalize()

                colores[accion](
                    f"### {nombre}\n\n{resultado['motivo']}"
                )

                st.metric(
                    "Confianza del modelo",
                    f"{resultado['confianza_pct']}%"
                )

                st.subheader(
                    "Probabilidades de cada recomendación"
                )

                st.bar_chart(
                    pd.Series(
                        resultado["probabilidades"]
                    )
                )

        except Exception as error:

            st.error(
                f"Error al preparar los indicadores: {error}"
            )


# ============================================================
# 10. PESTAÑA MAPA DE CALOR
# ============================================================

with tab_mapa:

    st.header("Casos de dengue por cantón")

    st.caption(
        "Datos cantonales históricos de 2013 a 2018. "
        "Se utilizan para comparar cantones, no para calcular "
        "el total nacional."
    )

    try:

        cantones = cargar_cantones()

        col1, col2 = st.columns(2)

        with col1:

            provincias = [
                "Todas"
            ] + sorted(
                cantones["provincia"].dropna().unique()
            )

            provincia = st.selectbox(
                "Seleccionar provincia",
                provincias
            )

        with col2:

            top = st.slider(
                "Cantidad de cantones",
                min_value=5,
                max_value=30,
                value=15
            )

        if provincia != "Todas":

            cantones = cantones[
                cantones["provincia"] == provincia
            ]

        if cantones.empty:

            st.warning(
                "No hay datos para esta selección."
            )

        else:

            mas_casos = (
                cantones.groupby("canton")["casos_dengue"]
                .sum()
                .nlargest(top)
                .index
            )

            tabla = (

                cantones[
                    cantones["canton"].isin(mas_casos)
                ]

                .pivot_table(
                    index="canton",
                    columns="anio",
                    values="casos_dengue",
                    aggfunc="sum"
                )

                .reindex(mas_casos)

            )

            tabla.columns = tabla.columns.astype(str)

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Cantón con más casos",
                    mas_casos[0]
                )

            with col2:

                st.metric(
                    "Casos acumulados",
                    f"{int(tabla.iloc[0].sum()):,}"
                )

            figura = px.imshow(

                tabla,

                text_auto=True,

                aspect="auto",

                color_continuous_scale="YlOrRd",

                labels={
                    "x": "Año",
                    "y": "Cantón",
                    "color": "Casos"
                }

            )

            st.plotly_chart(
                figura,
                use_container_width=True
            )

            st.caption(
                "Las casillas vacías corresponden a años "
                "sin registros para determinados cantones."
            )

    except Exception as error:

        st.error(
            f"No se pudo cargar el mapa: {error}"
        )
