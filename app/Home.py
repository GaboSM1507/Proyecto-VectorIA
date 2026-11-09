
import os
import json
import textwrap
import unicodedata
import requests
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="VectorIA | Inteligencia Epidemiológica",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

API = "http://127.0.0.1:8000"
TIMEOUT = 60

RUTA_DENGUE = os.path.join(
    ROOT_DIR, "data", "processed", "dengue_procesado.csv"
)

RUTA_CANTONES = os.path.join(
    ROOT_DIR, "data", "raw",
    "vectoria_casos_cantonal_anual_real.csv"
)

C = {
    "navy": "#102A43",
    "navy2": "#183B56",
    "teal": "#20A39E",
    "green": "#2BB673",
    "blue": "#247BA0",
    "red": "#DC6262",
    "orange": "#E9A23B",
    "background": "#F3F7FA",
    "white": "#FFFFFF",
    "text": "#243B53",
    "muted": "#718096",
    "border": "#E2EAF0"
}


# ============================================================
# 2. RENDERIZADO HTML
# ============================================================

def html(contenido):
    """
    Renderiza HTML directamente con Streamlit.
    No utiliza Markdown para interpretar los bloques HTML.
    """
    contenido = textwrap.dedent(contenido).strip()

    if contenido:
        st.html(contenido)


# ============================================================
# 3. ESTILOS
# ============================================================

html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #F3F7FA;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1550px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #102A43;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102A43, #183B56, #087F70);
}

section[data-testid="stSidebar"] * {
    color: #EDF6FA;
}

section[data-testid="stSidebar"] .stButton button {
    background: #20A39E;
    color: white;
    border: none;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border: 1px solid #E2EAF0;
    border-radius: 15px;
    box-shadow: 0 5px 18px rgba(16,42,67,.045);
}

div[data-testid="stMetricLabel"] {
    color: #718096;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #102A43;
    font-weight: 800;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 7px;
    background: #E8EFF4;
    padding: 7px;
    border-radius: 14px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px;
    color: #486581;
    font-weight: 700;
    padding: 12px 16px;
}

.stTabs [aria-selected="true"] {
    background: white !important;
    color: #087F70 !important;
}

.stButton > button {
    background: linear-gradient(110deg, #087F70, #20A39E);
    color: white;
    font-weight: 700;
    border: none;
    border-radius: 11px;
    min-height: 44px;
    transition: .2s;
}

.stButton > button:hover {
    background: linear-gradient(110deg, #066B60, #168B87);
    color: white;
    transform: translateY(-1px);
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border: 1px solid #E2EAF0;
    border-radius: 15px;
}

.stTextInput input,
.stNumberInput input {
    border-radius: 10px;
}

hr {
    border-color: #E2EAF0;
}

footer {
    visibility: hidden;
}

[data-testid="stFileUploader"] {
    background: #F8FBFC;
    border: 1px dashed #20A39E;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
}

</style>
""")


# ============================================================
# 4. COMPONENTES VISUALES
# ============================================================

def encabezado(titulo, descripcion):
    html(f"""
    <div style="margin: 8px 0 24px 0;">
        <div style="
            font-size:28px;
            font-weight:800;
            color:#102A43;
            letter-spacing:-.8px;
        ">{titulo}</div>
        <div style="
            font-size:14px;
            color:#718096;
            margin-top:6px;
        ">{descripcion}</div>
    </div>
    """)


def tarjeta(titulo, valor, descripcion, color="#20A39E"):
    html(f"""
    <div style="
        background:white;
        border:1px solid #E2EAF0;
        border-left:4px solid {color};
        border-radius:14px;
        padding:19px;
        box-shadow:0 5px 18px rgba(16,42,67,.045);
        min-height:120px;
    ">
        <div style="
            font-size:11px;
            color:#718096;
            font-weight:700;
            text-transform:uppercase;
            letter-spacing:.7px;
        ">{titulo}</div>

        <div style="
            font-size:27px;
            color:#102A43;
            font-weight:800;
            margin-top:9px;
        ">{valor}</div>

        <div style="
            font-size:12px;
            color:#8293A5;
            margin-top:5px;
        ">{descripcion}</div>
    </div>
    """)


def estilo_grafico(fig, altura=370):

    fig.update_layout(
        height=altura,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="DM Sans",
            color="#486581",
            size=12
        ),
        margin=dict(l=15, r=15, t=35, b=15),
        hoverlabel=dict(
            bgcolor="#102A43",
            font_color="white"
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0)"
        )
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#E2EAF0"
    )

    fig.update_yaxes(
        gridcolor="#EDF2F7",
        zeroline=False
    )

    return fig


def error_api(respuesta):

    try:
        detalle = respuesta.json().get(
            "detail",
            respuesta.text
        )
    except Exception:
        detalle = respuesta.text

    st.error(f"Error de la API: {detalle}")


def mostrar_probabilidades(probabilidades, titulo="Distribución de probabilidades"):

    if not probabilidades:
        return

    datos = pd.DataFrame({
        "Clase": list(probabilidades.keys()),
        "Probabilidad": [
            float(x) for x in probabilidades.values()
        ]
    })

    # La API ya entrega porcentajes.
    datos["Probabilidad"] = datos["Probabilidad"].clip(0, 100)

    datos = datos.sort_values(
        "Probabilidad",
        ascending=True
    )

    st.markdown(f"### {titulo}")

    fig = px.bar(
        datos,
        x="Probabilidad",
        y="Clase",
        orientation="h",
        color="Probabilidad",
        color_continuous_scale=[
            "#CDEBE5",
            "#20A39E",
            "#087F70"
        ],
        text_auto=".2f"
    )

    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Probabilidad (%)",
        yaxis_title="",
        xaxis_range=[0, 100]
    )

    fig.update_traces(
        texttemplate="%{x:.2f}%",
        textposition="outside",
        cliponaxis=False
    )

    st.plotly_chart(
        estilo_grafico(fig, 300),
        use_container_width=True
    )


# ============================================================
# 5. CARGA DE DATOS
# ============================================================

@st.cache_data(ttl=20)
def estado_api():

    try:
        respuesta = requests.get(
            f"{API}/",
            timeout=5
        )

        respuesta.raise_for_status()

        return respuesta.json()

    except requests.exceptions.RequestException:
        return None


@st.cache_data
def cargar_dengue():

    if not os.path.exists(RUTA_DENGUE):
        raise FileNotFoundError(
            f"No se encontró: {RUTA_DENGUE}"
        )

    df = pd.read_csv(RUTA_DENGUE)

    requeridas = [
        "fecha_inicio_semana",
        "casos_dengue",
        "lluvia_mm",
        "temp_max_c",
        "temp_min_c",
        "humedad_rel_pct",
        "semana_epi"
    ]

    faltantes = [
        c for c in requeridas
        if c not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Columnas faltantes: {faltantes}"
        )

    df["fecha_inicio_semana"] = pd.to_datetime(
        df["fecha_inicio_semana"],
        errors="coerce"
    )

    for col in requeridas[1:]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=requeridas
    )

    return df.sort_values(
        "fecha_inicio_semana"
    ).reset_index(drop=True)


@st.cache_data
def cargar_cantones():

    if not os.path.exists(RUTA_CANTONES):
        raise FileNotFoundError(
            f"No se encontró: {RUTA_CANTONES}"
        )

    df = pd.read_csv(RUTA_CANTONES)

    requeridas = [
        "provincia",
        "canton",
        "anio",
        "casos_dengue"
    ]

    faltantes = [
        c for c in requeridas
        if c not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Columnas faltantes: {faltantes}"
        )

    df["casos_dengue"] = pd.to_numeric(
        df["casos_dengue"],
        errors="coerce"
    )

    df["anio"] = pd.to_numeric(
        df["anio"],
        errors="coerce"
    )

    return df.dropna(
        subset=requeridas
    )


def umbral_endemico(df):
    """
    Umbral endémico para las 4 semanas que siguen al último registro:
    percentil 75 de los casos de esas mismas 4 semanas en los 5 años
    anteriores. Es la misma fórmula de vectoria_acciones.csv, con la
    que se entrenó la ANN.
    """
    ultima = df.iloc[-1]
    sumas = []

    for k in range(1, 6):

        misma_semana = df.index[
            (df["anio"] == ultima["anio"] - k)
            & (df["semana_epi"] == ultima["semana_epi"])
        ]

        if len(misma_semana) > 0:

            i = misma_semana[0]
            siguientes = df["casos_dengue"].iloc[i + 1:i + 5]

            if len(siguientes) == 4:
                sumas.append(float(siguientes.sum()))

    if not sumas:
        # Sin años anteriores: 4 semanas de casos promedio
        return float(round(df["casos_dengue"].mean() * 4))

    return max(1.0, float(round(np.percentile(sumas, 75))))


# ============================================================
# 6. ESTADO DE LOS MODELOS
# ============================================================

estado = estado_api()

api_ok = estado is not None

cnn_ok = api_ok and estado.get("cnn_ready", False)
rnn_ok = api_ok and estado.get("rnn_ready", False)
ann_ok = api_ok and estado.get("ann_ready", False)


# ============================================================
# 7. BARRA LATERAL
# ============================================================

with st.sidebar:

    html("""
    <div style="padding:15px 3px 20px 3px;">

        <div style="
            font-size:33px;
            font-weight:800;
            letter-spacing:-1.5px;
            color:white;
        ">🦟 VECTORIA</div>

        <div style="
            font-size:11px;
            color:#A9E7DF;
            letter-spacing:2px;
            margin-top:3px;
        ">INTELLIGENCE & HEALTH</div>

        <div style="
            font-size:12px;
            color:#D4E7EF;
            margin-top:13px;
            line-height:1.6;
        ">
            Plataforma inteligente de vigilancia epidemiológica.
        </div>

    </div>
    """)

    st.divider()

    st.markdown("### ESTADO DEL SISTEMA")

    if api_ok:
        st.success("API conectada")
    else:
        st.error("API desconectada")

    for nombre, disponible in [
        ("CNN · Mosquitos", cnn_ok),
        ("RNN · Pronóstico", rnn_ok),
        ("ANN · Prevención", ann_ok)
    ]:

        st.markdown(
            f"{'🟢' if disponible else '🔴'} {nombre}"
        )

    st.divider()

    st.markdown("### INFORMACIÓN")

    st.markdown("""
    **Proyecto académico**

    Vigilancia inteligente del dengue en Costa Rica.

    **Tecnologías**

    Python · TensorFlow · FastAPI · Streamlit
    """)

    st.divider()

    if st.button(
        "🔄 Actualizar estado",
        use_container_width=True
    ):

        estado_api.clear()
        st.rerun()

    html("""
    <div style="
        text-align:center;
        font-size:10px;
        color:#C1D8E2;
        padding:15px 0;
    ">
        VECTORIA © 2026<br>
        Inteligencia artificial para la salud
    </div>
    """)


# ============================================================
# 8. CABECERA PRINCIPAL
# ============================================================

html("""
<div style="
    background:linear-gradient(110deg,#102A43,#183B56 55%,#087F70);
    padding:32px;
    border-radius:20px;
    margin-bottom:22px;
    box-shadow:0 10px 30px rgba(16,42,67,.13);
    position:relative;
    overflow:hidden;
">

    <div style="
        position:absolute;
        right:30px;
        top:10px;
        font-size:100px;
        opacity:.10;
    ">🦟</div>

    <div style="
        color:#A9E7DF;
        font-size:11px;
        font-weight:700;
        letter-spacing:2px;
    ">
        PLATAFORMA DE INTELIGENCIA EPIDEMIOLÓGICA
    </div>

    <div style="
        color:white;
        font-size:39px;
        font-weight:800;
        letter-spacing:-1.5px;
        margin-top:9px;
    ">VectorIA</div>

    <div style="
        color:#D8E9F0;
        font-size:15px;
        margin-top:7px;
        max-width:750px;
        line-height:1.7;
    ">
        Vigilancia inteligente, análisis predictivo y apoyo
        a la prevención del dengue en Costa Rica.
    </div>

    <div style="
        display:inline-block;
        margin-top:20px;
        padding:8px 15px;
        background:rgba(255,255,255,.12);
        border:1px solid rgba(255,255,255,.2);
        border-radius:30px;
        color:white;
        font-size:11px;
        font-weight:700;
    ">
        CNN · RNN · ANN · FASTAPI
    </div>

</div>
""")

st.info(
    "VectorIA es una herramienta académica de apoyo a la "
    "vigilancia epidemiológica. No realiza diagnósticos médicos "
    "ni sustituye las decisiones de las autoridades sanitarias."
)


# ============================================================
# 9. NAVEGACIÓN
# ============================================================

tab_inicio, tab_cnn, tab_rnn, tab_ann, tab_mapa = st.tabs([
    "◈ RESUMEN",
    "🦟 IDENTIFICACIÓN CNN",
    "📈 PRONÓSTICO RNN",
    "🧠 PREVENCIÓN ANN",
    "🗺️ MAPA CANTONAL"
])


# ============================================================
# 10. RESUMEN
# ============================================================

with tab_inicio:

    encabezado(
        "Centro de inteligencia",
        "Vista general de los componentes y capacidades de VectorIA."
    )

    try:

        df = cargar_dengue()

        ultima = df.iloc[-1]

        casos = float(ultima["casos_dengue"])

        fecha = ultima["fecha_inicio_semana"].strftime(
            "%d/%m/%Y"
        )

        promedio = float(
            df.tail(4)["casos_dengue"].mean()
        )

        semanas = len(df)

        datos_ok = True

    except Exception as e:

        casos = promedio = 0
        fecha = "No disponible"
        semanas = 0
        datos_ok = False

        st.warning(
            f"No fue posible cargar los datos: {e}"
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        tarjeta(
            "API",
            "ONLINE" if api_ok else "OFFLINE",
            "Estado de conexión",
            C["teal"] if api_ok else C["red"]
        )

    with c2:
        tarjeta(
            "Datos históricos",
            str(semanas),
            "Registros semanales válidos",
            C["blue"]
        )

    with c3:
        tarjeta(
            "Último registro",
            f"{casos:,.0f}",
            f"Semana del {fecha}",
            C["green"]
        )

    with c4:
        tarjeta(
            "Promedio reciente",
            f"{promedio:,.1f}",
            "Últimas cuatro semanas",
            C["orange"]
        )

    st.markdown(" ")

    izquierda, derecha = st.columns([1.55, 1])

    with izquierda:

        st.markdown("### Evolución epidemiológica")

        if datos_ok:

            historico = df.tail(52)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=historico["fecha_inicio_semana"],
                y=historico["casos_dengue"],
                mode="lines",
                name="Casos registrados",
                line=dict(
                    color=C["teal"],
                    width=3
                ),
                fill="tozeroy",
                fillcolor="rgba(32,163,158,.10)"
            ))

            fig.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Casos",
                hovermode="x unified"
            )

            st.plotly_chart(
                estilo_grafico(fig, 370),
                use_container_width=True
            )

        else:

            st.warning(
                "No hay datos históricos disponibles."
            )

    with derecha:

        st.markdown("### Arquitectura inteligente")

        for nombre, descripcion, disponible in [
            ("CNN", "Identificación visual de mosquitos", cnn_ok),
            ("RNN", "Pronóstico de dengue", rnn_ok),
            ("ANN", "Recomendación preventiva", ann_ok)
        ]:

            color = C["teal"] if disponible else C["red"]

            estado_texto = (
                "Disponible" if disponible
                else "No disponible"
            )

            html(f"""
            <div style="
                background:white;
                border:1px solid #E2EAF0;
                border-left:4px solid {color};
                border-radius:12px;
                padding:17px;
                margin-bottom:12px;
            ">

                <div style="
                    font-size:18px;
                    font-weight:800;
                    color:#102A43;
                ">{nombre}</div>

                <div style="
                    color:#718096;
                    font-size:12px;
                    margin-top:4px;
                ">{descripcion}</div>

                <div style="
                    color:{color};
                    font-size:11px;
                    font-weight:700;
                    margin-top:10px;
                ">● {estado_texto}</div>

            </div>
            """)

    st.markdown("### Capacidades de la plataforma")

    a, b, c = st.columns(3)

    with a:
        with st.container(border=True):
            st.markdown("### 🦟")
            st.markdown("**Visión artificial**")
            st.caption(
                "Clasificación de especies mediante imágenes."
            )

    with b:
        with st.container(border=True):
            st.markdown("### 📊")
            st.markdown("**Análisis predictivo**")
            st.caption(
                "Pronóstico de casos mediante series temporales."
            )

    with c:
        with st.container(border=True):
            st.markdown("### 🧠")
            st.markdown("**Inteligencia preventiva**")
            st.caption(
                "Clasificación de acciones de prevención."
            )


# ============================================================
# 11. CNN: IDENTIFICACIÓN DE MOSQUITOS
# ============================================================

with tab_cnn:

    encabezado(
        "Identificación de mosquitos",
        "Análisis de imágenes mediante redes neuronales convolucionales."
    )

    if not cnn_ok:

        st.error(
            "El modelo CNN no está disponible. Verifica la API."
        )

    else:

        col_img, col_resultado = st.columns([1.05, 1])

        with col_img:

            with st.container(border=True):

                st.markdown("### Cargar fotografía")

                archivo = st.file_uploader(
                    "Selecciona una imagen del mosquito",
                    type=["jpg", "jpeg", "png"],
                    key="foto_mosquito"
                )

                if archivo:

                    st.image(
                        archivo,
                        use_container_width=True
                    )

                    st.caption(
                        f"Archivo: {archivo.name}"
                    )

                ejecutar = st.button(
                    "🔍 Analizar imagen",
                    disabled=archivo is None,
                    use_container_width=True,
                    key="analizar_cnn"
                )

        with col_resultado:

            with st.container(border=True):

                st.markdown("### Resultado del análisis")

                if ejecutar and archivo is not None:

                    try:

                        with st.spinner(
                            "Analizando imagen..."
                        ):

                            respuesta = requests.post(
                                f"{API}/predict/mosquito",
                                files={
                                    "file": (
                                        archivo.name,
                                        archivo.getvalue(),
                                        archivo.type
                                    )
                                },
                                timeout=TIMEOUT
                            )

                        if respuesta.status_code == 200:

                            resultado = respuesta.json()

                            st.session_state["cnn_resultado"] = resultado

                            especie = resultado.get(
                                "especie",
                                "No disponible"
                            )

                            st.session_state["especie_detectada"] = especie

                            st.session_state["cnn_aegypti"] = (
                                "aegypti" in str(especie).lower()
                            )

                        else:

                            error_api(respuesta)

                    except requests.exceptions.RequestException as e:

                        st.error(
                            f"Error de conexión: {e}"
                        )

                    except Exception as e:

                        st.error(
                            f"Error al procesar el resultado: {e}"
                        )

                if "cnn_resultado" in st.session_state:

                    resultado = st.session_state["cnn_resultado"]

                    especie = resultado.get(
                        "especie",
                        "No disponible"
                    )

                    confianza = float(
                        resultado.get("confianza_pct", 0)
                    )

                    st.success(
                        f"Clasificación: {especie}"
                    )

                    st.metric(
                        "Confianza del modelo",
                        f"{confianza:.2f}%"
                    )

                    st.progress(
                        max(0, min(100, int(confianza))) / 100
                    )

                    mostrar_probabilidades(
                        resultado.get("probabilidades", {}),
                        "Probabilidad por especie"
                    )

                    st.caption(
                        "La confianza es una salida del clasificador, "
                        "no una certeza científica."
                    )

                else:

                    st.info(
                        "Carga una fotografía para visualizar "
                        "el resultado del análisis."
                    )


# ============================================================
# 12. RNN: PRONÓSTICO
# ============================================================

with tab_rnn:

    encabezado(
        "Pronóstico de dengue",
        "Estimación de casos mediante análisis de series temporales."
    )

    if not rnn_ok:

        st.error(
            "El modelo RNN no está disponible."
        )

    else:

        try:

            df = cargar_dengue()

            if len(df) < 26:

                st.error(
                    "Se requieren al menos 26 semanas válidas."
                )

            else:

                ultimas = df.tail(26).copy()

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Ventana histórica",
                        "26 semanas"
                    )

                with c2:
                    st.metric(
                        "Inicio",
                        ultimas.iloc[0]["fecha_inicio_semana"].strftime(
                            "%d/%m/%Y"
                        )
                    )

                with c3:
                    st.metric(
                        "Último registro",
                        ultimas.iloc[-1]["fecha_inicio_semana"].strftime(
                            "%d/%m/%Y"
                        )
                    )

                with st.expander(
                    "Consultar datos históricos utilizados"
                ):

                    st.dataframe(
                        ultimas,
                        use_container_width=True,
                        hide_index=True
                    )

                generar = st.button(
                    "📈 Generar pronóstico de cuatro semanas",
                    use_container_width=True
                )

                if generar:

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

                    try:

                        with st.spinner(
                            "Procesando información con SimpleRNN..."
                        ):

                            respuesta = requests.post(
                                f"{API}/predict/rnn",
                                json={"semanas": semanas},
                                timeout=TIMEOUT
                            )

                        if respuesta.status_code == 200:

                            st.session_state["rnn_resultado"] = (
                                respuesta.json()
                            )

                            st.session_state.pop(
                                "ann_resultado",
                                None
                            )

                        else:

                            error_api(respuesta)

                    except requests.exceptions.RequestException as e:

                        st.error(
                            f"Error de comunicación: {e}"
                        )

                if "rnn_resultado" in st.session_state:

                    resultado = st.session_state["rnn_resultado"]

                    predicciones = resultado.get(
                        "pronostico_casos",
                        []
                    )

                    historicos = resultado.get(
                        "historico_ingresado",
                        []
                    )

                    etiquetas = resultado.get(
                        "semanas_proyectadas",
                        []
                    )

                    total = resultado.get(
                        "total_estimado_periodo",
                        sum(predicciones)
                    )

                    tendencia = resultado.get(
                        "tendencia_general",
                        "No disponible"
                    )

                    st.divider()

                    st.markdown(
                        "### Resultado del pronóstico"
                    )

                    a, b, c = st.columns(3)

                    with a:
                        st.metric(
                            "Total estimado",
                            f"{float(total):,.1f} casos"
                        )

                    with b:
                        # Texto corto para que quepa en la tarjeta
                        st.metric(
                            "Tendencia general",
                            {
                                "Ascendente": "↗ Al alza",
                                "Descendente / Estable": "↘ A la baja"
                            }.get(str(tendencia), str(tendencia)),
                            help=(
                                "Compara el pronóstico de la semana +4 "
                                "con la última semana registrada."
                            )
                        )

                    with c:
                        st.metric(
                            "Horizonte",
                            "4 semanas"
                        )

                    if (
                        len(predicciones) == 4
                        and len(etiquetas) == 4
                        and len(historicos) > 0
                    ):

                        st.markdown(
                            "### Evolución histórica y proyección"
                        )

                        fechas = (
                            ultimas["fecha_inicio_semana"]
                            .dt.strftime("%d/%m/%Y")
                            .tolist()
                        )

                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=fechas[-len(historicos):],
                            y=historicos,
                            mode="lines",
                            name="Histórico",
                            line=dict(
                                color=C["blue"],
                                width=3
                            )
                        ))

                        # La línea del pronóstico arranca en la última semana real,
                        # para que no quede un hueco entre las dos
                        fig.add_trace(go.Scatter(
                            x=[fechas[-1]] + list(etiquetas),
                            y=[historicos[-1]] + list(predicciones),
                            mode="lines+markers",
                            name="Pronóstico",
                            line=dict(
                                color=C["teal"],
                                width=3,
                                dash="dash"
                            ),
                            marker=dict(size=[0] + [9] * len(predicciones))
                        ))

                        fig.update_layout(
                            xaxis_title="Semana",
                            yaxis_title="Casos estimados"
                        )

                        st.plotly_chart(
                            estilo_grafico(fig, 400),
                            use_container_width=True
                        )

                        tabla = pd.DataFrame({
                            "Semana": etiquetas,
                            "Casos estimados": predicciones
                        })

                        st.markdown(
                            "### Detalle de predicciones"
                        )

                        st.dataframe(
                            tabla,
                            use_container_width=True,
                            hide_index=True
                        )

                    else:

                        st.warning(
                            "La respuesta de la API no contiene "
                            "las cuatro predicciones esperadas."
                        )

                    st.caption(
                        "Modelo: " + str(
                            resultado.get(
                                "modelo_utilizado",
                                "No especificado"
                            )
                        )
                    )

                    st.caption(
                        "Las cifras son estimaciones del modelo, "
                        "no datos oficiales."
                    )

        except Exception as e:

            st.error(
                f"No fue posible cargar los datos: {e}"
            )


# ============================================================
# 13. ANN: PREVENCIÓN
# ============================================================

with tab_ann:

    encabezado(
        "Centro de prevención inteligente",
        "Evaluación de indicadores para sugerir acciones preventivas."
    )

    if not ann_ok:

        st.error(
            "El modelo ANN no está disponible."
        )

    elif "rnn_resultado" not in st.session_state:

        st.warning(
            "Primero ejecuta el pronóstico en la pestaña RNN. "
            "La ANN utilizará ese resultado para su evaluación."
        )

    else:

        try:

            df = cargar_dengue()

            ultima = df.iloc[-1]

            ultimas_dos = df.tail(2)

            resultado_rnn = st.session_state["rnn_resultado"]

            casos_predichos = float(
                resultado_rnn.get(
                    "total_estimado_periodo",
                    sum(
                        resultado_rnn.get(
                            "pronostico_casos",
                            []
                        )
                    )
                )
            )

            lluvia = float(
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

            st.markdown(
                "### Indicadores de entrada"
            )

            a, b, c = st.columns(3)

            with a:

                tarjeta(
                    "Casos proyectados",
                    f"{casos_predichos:.1f}",
                    "Próximas cuatro semanas",
                    C["blue"]
                )

            with b:

                tarjeta(
                    "Lluvia acumulada",
                    f"{lluvia:.1f} mm",
                    "Últimas dos semanas",
                    C["teal"]
                )

            with c:

                tarjeta(
                    "Humedad relativa",
                    f"{humedad:.1f}%",
                    "Último registro disponible",
                    C["green"]
                )

            st.divider()

            col_a, col_b = st.columns(2)

            with col_a:

                st.markdown(
                    "### Condiciones climáticas"
                )

                st.write(
                    f"Temperatura máxima: **{temp_max:.1f} °C**"
                )

                st.write(
                    f"Temperatura mínima: **{temp_min:.1f} °C**"
                )

                st.write(
                    f"Humedad relativa: **{humedad:.1f}%**"
                )

                st.write(
                    f"Precipitación 14 días: **{lluvia:.1f} mm**"
                )

            with col_b:

                st.markdown(
                    "### Indicadores epidemiológicos"
                )

                umbral_calculado = umbral_endemico(df)

                umbral = st.number_input(
                    "Umbral endémico de referencia",
                    min_value=1.0,
                    value=umbral_calculado,
                    step=1.0
                )

                st.caption(
                    "Calculado con los datos: percentil 75 de los casos "
                    "de esas mismas 4 semanas en los 5 años anteriores, "
                    "igual que en los datos de entrenamiento de la ANN."
                )

                aegypti = st.checkbox(
                    "Se detectó Aedes aegypti",
                    value=st.session_state.get(
                        "cnn_aegypti",
                        False
                    )
                )

                if "especie_detectada" in st.session_state:

                    st.caption(
                        "Última identificación: "
                        + st.session_state["especie_detectada"]
                    )

            st.divider()

            ejecutar_ann = st.button(
                "🧠 Generar recomendación preventiva",
                use_container_width=True
            )

            if ejecutar_ann:

                datos = {

                    "casos_pred_4sem": casos_predichos,

                    "casos_umbral_endemico": float(umbral),

                    "lluvia_14d_mm": lluvia,

                    "temp_max": temp_max,

                    "temp_min": temp_min,

                    "humedad": humedad,

                    "aegypti_detectado": int(aegypti)

                }

                try:

                    with st.spinner(
                        "La ANN está evaluando los indicadores..."
                    ):

                        respuesta = requests.post(
                            f"{API}/predict/accion",
                            json=datos,
                            timeout=TIMEOUT
                        )

                    if respuesta.status_code == 200:

                        st.session_state["ann_resultado"] = (
                            respuesta.json()
                        )

                    else:

                        error_api(respuesta)

                except requests.exceptions.RequestException as e:

                    st.error(
                        f"Error de conexión: {e}"
                    )

            if "ann_resultado" in st.session_state:

                resultado = st.session_state["ann_resultado"]

                accion = resultado.get(
                    "accion",
                    "No disponible"
                )

                motivo = resultado.get(
                    "motivo",
                    "No se proporcionó una explicación."
                )

                confianza = float(
                    resultado.get(
                        "confianza_pct",
                        0
                    )
                )

                colores = {
                    "alerta_sanitaria": C["red"],
                    "fumigacion_focalizada": C["orange"],
                    "eliminacion_criaderos": C["teal"],
                    "monitoreo_normal": C["green"]
                }

                color = colores.get(
                    accion,
                    C["blue"]
                )

                nombre = accion.replace(
                    "_",
                    " "
                ).capitalize()

                st.divider()

                html(f"""
                <div style="
                    background:white;
                    border-radius:16px;
                    border:1px solid #E2EAF0;
                    border-top:5px solid {color};
                    padding:25px;
                    box-shadow:0 5px 20px rgba(16,42,67,.05);
                ">

                    <div style="
                        color:#718096;
                        font-size:11px;
                        font-weight:700;
                        letter-spacing:1.5px;
                    ">
                        RECOMENDACIÓN DEL MODELO ANN
                    </div>

                    <div style="
                        color:#102A43;
                        font-size:28px;
                        font-weight:800;
                        margin-top:12px;
                    ">{nombre}</div>

                    <div style="
                        color:#486581;
                        font-size:14px;
                        line-height:1.7;
                        margin-top:13px;
                    ">{motivo}</div>

                    <div style="
                        display:inline-block;
                        background:#E8F5F2;
                        color:#087F70;
                        padding:8px 13px;
                        border-radius:20px;
                        font-size:12px;
                        font-weight:700;
                        margin-top:17px;
                    ">
                        Confianza: {confianza:.2f}%
                    </div>

                </div>
                """)

                mostrar_probabilidades(
                    resultado.get("probabilidades", {}),
                    "Distribución de probabilidades"
                )

                st.caption(
                    "La recomendación es orientativa y debe ser "
                    "validada por personal responsable."
                )

        except Exception as e:

            st.error(
                f"No fue posible preparar los indicadores: {e}"
            )


# ============================================================
# 14. MAPA CANTONAL
# ============================================================

RUTA_MAPA = os.path.join(
    ROOT_DIR, "data", "raw", "cantones_costa_rica.geojson"
)

# Los datos (2013-2018) usan el nombre viejo de algunos cantones
NOMBRES_ANTIGUOS = {
    "alfaro ruiz": "zarcero",
    "valverde vega": "sarchi",
    "coronado": "vazquez de coronado"
}

# Escala del mapa: pocos casos (turquesa) -> muchos casos (rojo)
ESCALA_MAPA = [
    [0.00, "#2EC4B6"],
    [0.40, "#E9C46A"],
    [0.70, "#F4A261"],
    [0.88, "#E76F51"],
    [1.00, "#E63946"]
]

FONDO_MAPA = "#0F2236"
SIN_DATOS = "#2C4258"
BORDE_MAPA = "#0A1828"

COLORES_PROVINCIA = [
    "#20A39E", "#247BA0", "#E9A23B", "#DC6262",
    "#2BB673", "#8E7CC3", "#F2C14E"
]

# Costa Rica continental (sin la Isla del Coco)
RANGO_CR = ([-86.0, -82.5], [8.0, 11.25])


def clave_canton(nombre):
    """Nombre en minúscula y sin tildes, para unir los datos con el mapa."""
    texto = unicodedata.normalize("NFD", str(nombre).lower().strip())
    texto = "".join(
        c for c in texto
        if unicodedata.category(c) != "Mn"
    )
    return NOMBRES_ANTIGUOS.get(texto, texto)


@st.cache_data
def cargar_mapa():

    if not os.path.exists(RUTA_MAPA):
        raise FileNotFoundError(
            f"No se encontró: {RUTA_MAPA}"
        )

    with open(RUTA_MAPA, encoding="utf-8") as archivo:
        return json.load(archivo)


def rango_cantones(mapa, claves, margen=0.15):
    """Longitudes y latitudes que encierran esos cantones (para hacer zoom)."""
    xs, ys = [], []

    for f in mapa["features"]:
        if f["id"] in claves:
            for poligono in f["geometry"]["coordinates"]:
                for x, y in poligono[0]:
                    if x > -86:          # deja fuera la Isla del Coco
                        xs.append(x)
                        ys.append(y)

    if not xs:
        return RANGO_CR

    return (
        [min(xs) - margen, max(xs) + margen],
        [min(ys) - margen, max(ys) + margen]
    )


def tabla_periodo(datos, nombres_mapa):
    """Una fila por cantón: casos del periodo y su parte del total."""
    tabla = (
        datos.groupby(["clave", "provincia"], as_index=False)["casos_dengue"]
        .sum()
        .rename(columns={"casos_dengue": "casos"})
    )

    total = tabla["casos"].sum()

    tabla["canton"] = tabla["clave"].map(nombres_mapa)
    tabla["pct"] = tabla["casos"] / total * 100 if total else 0.0

    # Escala logarítmica: si no, solo se verían 3 o 4 cantones
    tabla["z"] = np.log10(tabla["casos"] + 1)

    return tabla.sort_values(
        "casos",
        ascending=False
    ).reset_index(drop=True)


def rango_color(valores_z):
    """
    La escala empieza en el 10 % de cantones con menos casos, para que los
    colores se repartan mejor (los que tienen menos quedan en turquesa).
    """
    valores_z = np.asarray(list(valores_z), dtype=float)
    zmin = float(np.percentile(valores_z, 10))
    zmax = float(valores_z.max())
    if zmax - zmin < 0.5:
        zmin = max(0.0, zmax - 0.5)
    return zmin, zmax


def marcas_escala(zmin, zmax):
    valores = [
        v for v in [1, 10, 100, 1000, 10000, 100000]
        if zmin + 0.15 < np.log10(v + 1) <= zmax + 1e-9
    ]
    primera = "0" if zmin < 0.01 else f"≤ {10 ** zmin - 1:,.0f}"
    return (
        [zmin] + [float(np.log10(v + 1)) for v in valores],
        [primera] + [f"{v:,}" for v in valores]
    )


def datos_traza(tabla):
    return dict(
        locations=tabla["clave"].tolist(),
        z=tabla["z"].tolist(),
        customdata=tabla[["canton", "provincia", "casos", "pct"]].values.tolist()
    )


def figura_mapa(mapa, nombres_mapa, textos_gris, tabla, zrango, rango, cuadros=None):

    claves = list(nombres_mapa)
    zmin, zmax = zrango
    marcas = marcas_escala(zmin, zmax)

    fig = go.Figure()

    # Todos los cantones en gris: los que no tienen datos quedan así
    fig.add_trace(go.Choropleth(
        geojson=mapa,
        featureidkey="id",
        locations=claves,
        z=[0] * len(claves),
        colorscale=[[0, SIN_DATOS], [1, SIN_DATOS]],
        showscale=False,
        marker_line_color=BORDE_MAPA,
        marker_line_width=0.7,
        text=[nombres_mapa[c] for c in claves],
        customdata=[textos_gris[c] for c in claves],
        hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>"
    ))

    # Cantones con datos, coloreados según sus casos
    fig.add_trace(go.Choropleth(
        geojson=mapa,
        featureidkey="id",
        zmin=zmin,
        zmax=zmax,
        colorscale=ESCALA_MAPA,
        marker_line_color=BORDE_MAPA,
        marker_line_width=0.7,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]}<br><br>"
            "<b>%{customdata[2]:,.0f}</b> casos<br>"
            "%{customdata[3]:.1f} % de la selección"
            "<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="Casos", font=dict(color="#D8E9F0")),
            tickvals=marcas[0],
            ticktext=marcas[1],
            tickfont=dict(color="#D8E9F0"),
            thickness=14,
            len=0.7,
            x=0.97,
            outlinewidth=0
        ),
        **datos_traza(tabla)
    ))

    if cuadros:

        # Un cuadro por año: solo cambian los colores de la segunda capa
        fig.frames = [
            go.Frame(
                name=str(anio),
                data=[go.Choropleth(**datos_traza(t))],
                traces=[1]
            )
            for anio, t in cuadros.items()
        ]

        pasos = [
            dict(
                label=str(anio),
                method="animate",
                args=[[str(anio)], dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0)
                )]
            )
            for anio in cuadros
        ]

        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                direction="left",
                x=0.02,
                y=-0.01,
                xanchor="left",
                yanchor="top",
                pad=dict(r=10, t=48),
                bgcolor="#20A39E",
                bordercolor="#20A39E",
                font=dict(color="#0F2236", size=13),
                buttons=[
                    dict(
                        label="▶  Reproducir",
                        method="animate",
                        args=[None, dict(
                            frame=dict(duration=1300, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=True,
                            mode="immediate"
                        )]
                    ),
                    dict(
                        label="⏸  Pausa",
                        method="animate",
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            transition=dict(duration=0),
                            mode="immediate"
                        )]
                    )
                ]
            )],
            sliders=[dict(
                active=0,
                steps=pasos,
                x=0.22,
                y=-0.01,
                len=0.72,
                xanchor="left",
                yanchor="top",
                pad=dict(t=10, b=10),
                bgcolor="#2C4258",
                activebgcolor="#E9C46A",
                bordercolor="#2C4258",
                tickcolor="#D8E9F0",
                font=dict(color="#D8E9F0"),
                currentvalue=dict(
                    prefix="Año ",
                    font=dict(size=26, color="white"),
                    xanchor="left"
                )
            )]
        )

    fig.update_geos(
        visible=False,
        projection_type="mercator",
        lonaxis_range=rango[0],
        lataxis_range=rango[1],
        bgcolor=FONDO_MAPA
    )

    fig.update_layout(
        height=760 if cuadros else 640,
        paper_bgcolor=FONDO_MAPA,
        margin=dict(l=0, r=0, t=0, b=120 if cuadros else 0),
        font=dict(family="DM Sans", color="#D8E9F0"),
        hoverlabel=dict(
            bgcolor="#102A43",
            bordercolor="#20A39E",
            font=dict(color="white", size=13)
        ),
        dragmode=False
    )

    return fig


def color_escala(valor):
    """Color RGB de la escala del mapa para un valor entre 0 y 1."""
    posiciones = [p for p, _ in ESCALA_MAPA]
    rgb = [
        [int(c[i:i + 2], 16) for i in (1, 3, 5)]
        for _, c in ESCALA_MAPA
    ]
    return [
        int(np.interp(valor, posiciones, [c[k] for c in rgb]))
        for k in range(3)
    ]


def vista_3d(mapa, tabla, rango):
    """Cada cantón se levanta según sus casos (pydeck)."""
    casos = dict(zip(tabla["clave"], tabla["casos"]))
    provincias = dict(zip(tabla["clave"], tabla["provincia"]))
    maximo = max(float(tabla["casos"].max()), 1.0) if len(tabla) else 1.0
    zmin, zmax = rango_color(tabla["z"])

    cantones_3d = []

    for f in mapa["features"]:

        valor = casos.get(f["id"])

        if valor is None:
            color = [44, 66, 88, 160]
            altura = 0.0
            texto = "Sin datos en la selección"
        else:
            t = (np.log10(valor + 1) - zmin) / (zmax - zmin)
            color = color_escala(float(np.clip(t, 0, 1))) + [235]
            altura = float(np.sqrt(valor / maximo) * 30000)
            texto = f"{provincias[f['id']]} · {valor:,.0f} casos"

        cantones_3d.append({
            "type": "Feature",
            "geometry": f["geometry"],
            "properties": {
                "canton": f["properties"]["canton"],
                "texto": texto,
                "color": color,
                "altura": altura
            }
        })

    capa = pdk.Layer(
        "GeoJsonLayer",
        {"type": "FeatureCollection", "features": cantones_3d},
        extruded=True,
        wireframe=False,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 90],
        get_fill_color="properties.color",
        get_elevation="properties.altura",
        get_line_color=[10, 24, 40],
        line_width_min_pixels=1
    )

    vista = pdk.ViewState(
        longitude=(rango[0][0] + rango[0][1]) / 2,
        latitude=(rango[1][0] + rango[1][1]) / 2 - 0.35,
        zoom=8.0 if rango == RANGO_CR else 9.0,
        pitch=50,
        bearing=-15
    )

    return pdk.Deck(
        layers=[capa],
        initial_view_state=vista,
        # La rueda del mouse baja la página en vez de alejar el mapa
        views=[pdk.View(
            type="MapView",
            controller={"scrollZoom": False, "dragRotate": True}
        )],
        map_style="dark",
        tooltip={
            "html": "<b>{canton}</b><br>{texto}",
            "style": {
                "backgroundColor": "#102A43",
                "color": "white",
                "fontFamily": "DM Sans"
            }
        }
    )


with tab_mapa:

    encabezado(
        "Inteligencia territorial",
        "Cómo se distribuyó el dengue entre los cantones del país, de 2013 a 2018."
    )

    try:

        cantones = cargar_cantones().copy()
        cantones["anio"] = cantones["anio"].astype(int)
        cantones["clave"] = cantones["canton"].map(clave_canton)

        mapa = cargar_mapa()

        nombres_mapa = {
            f["id"]: f["properties"]["canton"]
            for f in mapa["features"]
        }

        anios = sorted(cantones["anio"].unique())
        todo_el_periodo = f"{anios[0]}–{anios[-1]} (acumulado)"

        col_prov, col_anio, col_vista = st.columns([1.1, 1.1, 1.5])

        with col_prov:

            provincia = st.selectbox(
                "Provincia",
                ["Todas las provincias"] + sorted(
                    cantones["provincia"].astype(str).unique()
                )
            )

        with col_anio:

            periodo = st.selectbox(
                "Periodo",
                [todo_el_periodo] + [str(a) for a in anios]
            )

        with col_vista:

            vista = st.segmented_control(
                "Vista",
                ["🗺️ Mapa", "▶️ Animación por año", "🏔️ Vista 3D"],
                default="🗺️ Mapa",
                required=True
            )

        if provincia != "Todas las provincias":
            filtrado = cantones[
                cantones["provincia"].astype(str) == provincia
            ].copy()
        else:
            filtrado = cantones.copy()

        if periodo == todo_el_periodo:
            del_periodo = filtrado
        else:
            del_periodo = filtrado[filtrado["anio"] == int(periodo)]

        if del_periodo.empty:

            st.warning(
                "No hay registros para esta selección."
            )

        else:

            tabla = tabla_periodo(del_periodo, nombres_mapa)

            # Qué dice el mapa sobre los cantones en gris
            en_archivo = set(cantones["clave"])
            en_seleccion = set(filtrado["clave"])
            textos_gris = {
                c: (
                    "Sin datos en el archivo" if c not in en_archivo
                    else "Fuera de la selección" if c not in en_seleccion
                    else "Sin datos en este periodo"
                )
                for c in nombres_mapa
            }

            rango = (
                RANGO_CR if provincia == "Todas las provincias"
                else rango_cantones(mapa, set(tabla["clave"]))
            )

            # ---------------- Indicadores ----------------

            total = tabla["casos"].sum()
            primero = tabla.iloc[0]
            por_anio = filtrado.groupby("anio")["casos_dengue"].sum()
            top5 = tabla.head(5)["casos"].sum() / total * 100 if total else 0

            a, b, c, d = st.columns(4)

            with a:
                tarjeta(
                    "Casos acumulados",
                    f"{total:,.0f}",
                    f"{provincia} · {periodo}",
                    C["blue"]
                )

            with b:
                tarjeta(
                    "Cantones con datos",
                    str(len(tabla)),
                    "En la selección",
                    C["teal"]
                )

            with c:
                tarjeta(
                    "Cantón más afectado",
                    primero["canton"],
                    f"{primero['casos']:,.0f} casos ({primero['pct']:.1f} %)",
                    C["red"]
                )

            with d:
                tarjeta(
                    "Año más fuerte",
                    str(int(por_anio.idxmax())),
                    f"{por_anio.max():,.0f} casos en la selección",
                    C["orange"]
                )

            st.markdown(" ")

            # ---------------- Mapa ----------------

            if vista == "🏔️ Vista 3D":

                st.pydeck_chart(
                    vista_3d(mapa, tabla, rango),
                    height=640
                )

                st.caption(
                    "Arrastra para mover el mapa; con clic derecho (o Ctrl + arrastrar) "
                    "se gira, y con los botones + y − se acerca. "
                    "La altura de cada cantón crece con sus casos."
                )

            elif vista == "▶️ Animación por año":

                cuadros = {
                    anio: tabla_periodo(
                        filtrado[filtrado["anio"] == anio],
                        nombres_mapa
                    )
                    for anio in anios
                }

                # La misma escala en todos los años, para poder compararlos
                zrango = rango_color(
                    pd.concat([t["z"] for t in cuadros.values()])
                )

                st.plotly_chart(
                    figura_mapa(
                        mapa, nombres_mapa, textos_gris,
                        cuadros[anios[0]], zrango, rango, cuadros
                    ),
                    width="stretch",
                    theme=None,
                    config={"displayModeBar": False, "scrollZoom": False}
                )

                st.caption(
                    "Presiona «Reproducir» para ver cómo cambian los casos año a año, "
                    "o mueve la barra. Todos los años usan la misma escala de colores."
                )

            else:

                st.plotly_chart(
                    figura_mapa(
                        mapa, nombres_mapa, textos_gris,
                        tabla, rango_color(tabla["z"]), rango
                    ),
                    width="stretch",
                    theme=None,
                    config={"displayModeBar": False, "scrollZoom": False}
                )

            st.info(
                f"Los 5 cantones con más casos concentran el **{top5:.0f} %** "
                f"de los casos de la selección ({provincia.lower()}, {periodo})."
            )

            st.divider()

            # ---------------- Ranking ----------------

            col_grafico, col_tabla = st.columns([1.25, 1])

            with col_grafico:

                st.markdown(
                    "### Cantones con más casos"
                )

                top = st.slider(
                    "Cantidad de cantones",
                    min_value=5,
                    max_value=30,
                    value=15
                )

                top_datos = tabla.head(top).sort_values(
                    "casos",
                    ascending=True
                )

                fig = px.bar(
                    top_datos,
                    x="casos",
                    y="canton",
                    orientation="h",
                    color="z",
                    color_continuous_scale=ESCALA_MAPA,
                    range_color=list(rango_color(tabla["z"])),
                    custom_data=["provincia", "pct"],
                    text="casos"
                )

                fig.update_traces(
                    texttemplate="%{x:,.0f}",
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=(
                        "<b>%{y}</b><br>%{customdata[0]}<br>"
                        "%{x:,.0f} casos (%{customdata[1]:.1f} %)<extra></extra>"
                    )
                )

                fig.update_layout(
                    coloraxis_showscale=False,
                    xaxis_title="Casos acumulados",
                    yaxis_title="",
                    # espacio a la derecha para que no se corte el número
                    xaxis_range=[0, float(top_datos["casos"].max()) * 1.18]
                )

                st.plotly_chart(
                    estilo_grafico(fig, 480),
                    width="stretch"
                )

            with col_tabla:

                st.markdown(
                    "### Ranking cantonal"
                )

                ranking = tabla.head(top)[
                    ["canton", "provincia", "casos", "pct"]
                ].copy()

                ranking.insert(
                    0,
                    "Posición",
                    range(1, len(ranking) + 1)
                )

                # Mismo formato de miles que el resto de la app (12,066)
                ranking["casos"] = ranking["casos"].map(
                    lambda v: f"{v:,.0f}"
                )

                st.dataframe(
                    ranking,
                    width="stretch",
                    hide_index=True,
                    height=520,
                    column_config={
                        "Posición": st.column_config.NumberColumn(
                            "#", width=36
                        ),
                        "canton": st.column_config.TextColumn(
                            "Cantón", width=125
                        ),
                        "provincia": st.column_config.TextColumn(
                            "Provincia", width=95
                        ),
                        "casos": st.column_config.TextColumn(
                            "Casos", width=62
                        ),
                        "pct": st.column_config.ProgressColumn(
                            "% del total",
                            format="%.1f %%",
                            min_value=0,
                            max_value=float(ranking["pct"].max()),
                            width=110
                        )
                    }
                )

            st.divider()

            # ---------------- Evolución anual ----------------

            st.markdown(
                "### Evolución anual"
            )

            if provincia == "Todas las provincias":

                evolucion = (
                    filtrado.groupby(["anio", "provincia"], as_index=False)["casos_dengue"]
                    .sum()
                )

                fig = px.area(
                    evolucion,
                    x="anio",
                    y="casos_dengue",
                    color="provincia",
                    color_discrete_sequence=COLORES_PROVINCIA,
                    markers=True
                )

            else:

                evolucion = (
                    filtrado.groupby("anio", as_index=False)["casos_dengue"]
                    .sum()
                )

                fig = px.area(
                    evolucion,
                    x="anio",
                    y="casos_dengue",
                    markers=True,
                    color_discrete_sequence=[C["teal"]]
                )

            fig.update_layout(
                xaxis_title="Año",
                yaxis_title="Casos registrados",
                legend_title_text="Provincia",
                hovermode="x unified"
            )

            fig.update_xaxes(dtick=1)

            st.plotly_chart(
                estilo_grafico(fig, 380),
                width="stretch"
            )

            st.caption(
                "Datos cantonales de 2013 a 2018. La suma de los cantones no coincide con el "
                "total nacional, así que sirven para comparar cantones entre sí, no como cifra "
                "oficial. En gris, los cantones sin datos en el archivo: las cabeceras de "
                "Alajuela, Cartago, Heredia, Puntarenas y Limón, Dota, Tarrazú, León Cortés, "
                "Quepos y los cantones nuevos Río Cuarto y Puerto Jiménez. "
                "Fronteras cantonales: geoBoundaries (dominio público)."
            )

    except Exception as e:

        st.error(
            f"No se pudo generar el análisis territorial: {e}"
        )


# ============================================================
# 15. PIE DE PÁGINA
# ============================================================

html("""
<div style="
    margin-top:45px;
    padding:25px 10px;
    border-top:1px solid #DCE6ED;
    text-align:center;
">

    <div style="
        font-size:17px;
        font-weight:800;
        color:#102A43;
        letter-spacing:1.5px;
    ">VECTORIA</div>

    <div style="
        font-size:12px;
        color:#8293A5;
        margin-top:7px;
    ">
        Inteligencia artificial para la vigilancia
        y prevención del dengue en Costa Rica.
    </div>

    <div style="
        font-size:10px;
        color:#A0AEC0;
        margin-top:10px;
    ">
        Proyecto académico · 2026 · CNN / RNN / ANN
    </div>

</div>
""")