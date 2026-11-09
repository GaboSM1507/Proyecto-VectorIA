
import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import date
from enum import Enum
from functools import lru_cache
from io import BytesIO
from typing import Annotated, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Path, Query, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Los gráficos se dibujan sin ventana (en un servidor no hay pantalla)
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

DESCRIPCION = """
**VectorIA** apoya la vigilancia del dengue en Costa Rica con tres redes neuronales:

| Módulo | Red | Endpoint |
|---|---|---|
| 🦟 Identificar el mosquito de una foto | CNN (MobileNetV2) | `POST /predict/mosquito` |
| 📈 Pronosticar los casos de las próximas 4 semanas | SimpleRNN | `POST /predict/casos` |
| 🛡️ Recomendar una acción preventiva | ANN | `POST /predict/accion` |
| 🗺️ Bonus: pronóstico por provincia | SimpleRNN (datos sintéticos) | `POST /predict/casos/provincia` |

Los endpoints de **📊 Visualizaciones** devuelven gráficos PNG: presiona *Try it out* y luego *Execute*, y el gráfico aparece aquí mismo.

Cada endpoint de predicción trae ejemplos listos con datos reales del proyecto: elige uno en *Examples* y ejecútalo.

> ⚠️ Herramienta de apoyo a la vigilancia. **No diagnostica dengue en personas.**
"""

TAG_CNN = "🦟 Módulo 1 · Mosquitos (CNN)"
TAG_RNN = "📈 Módulo 2 · Casos (SimpleRNN)"
TAG_ANN = "🛡️ Módulo 3 · Acciones (ANN)"
TAG_PROVINCIAS = "🗺️ Bonus · Provincias"
TAG_VISUAL = "📊 Visualizaciones"
TAG_ESTADO = "⚙️ Estado"

ETIQUETAS = [
    {"name": TAG_CNN, "description": "Sube una foto JPG o PNG y la red identifica la especie: *Aedes aegypti*, *Aedes albopictus* o *Culex*."},
    {"name": TAG_RNN, "description": "Con las últimas 26 semanas de casos y clima, pronostica los casos de las 4 siguientes. Datos reales (OPS/PAHO y NASA POWER)."},
    {"name": TAG_ANN, "description": "Combina el pronóstico, el umbral endémico, el clima y la especie detectada, y recomienda una acción con su motivo."},
    {"name": TAG_PROVINCIAS, "description": "Bonus del enunciado: una SimpleRNN para las 7 provincias. **Usa datos sintéticos** del paquete del curso, no cifras oficiales."},
    {"name": TAG_VISUAL, "description": "Gráficos PNG del pronóstico contra el umbral endémico, generados en el momento por la API."},
    {"name": TAG_ESTADO, "description": "Qué modelos están cargados."},
]

app = FastAPI(
    title="VectorIA API",
    summary="API inteligente para vigilancia y prevención del dengue en Costa Rica.",
    description=DESCRIPCION,
    version="2.1.0",
    openapi_tags=ETIQUETAS,
    docs_url=None    # /docs se arma en la sección 11, con los colores de VectorIA
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


# ============================================================
# 2. RUTAS DE LOS MODELOS Y DATOS
# ============================================================

cnn_model_path = os.path.join(MODELS_DIR, "mobilenet_mosquitos.keras")
cnn_clases_path = os.path.join(MODELS_DIR, "mosquitos_clases.json")

ann_model_path = os.path.join(MODELS_DIR, "ann_acciones.keras")
ann_prep_path = os.path.join(MODELS_DIR, "ann_acciones_preprocesado.pkl")

rnn_model_path = os.path.join(MODELS_DIR, "rnn_simple.h5")
rnn_prep_path = os.path.join(MODELS_DIR, "rnn_simple_preprocesado.pkl")

prov_model_path = os.path.join(MODELS_DIR, "rnn_provincias.h5")
prov_prep_path = os.path.join(MODELS_DIR, "rnn_provincias_preprocesado.pkl")
prov_resultados_path = os.path.join(MODELS_DIR, "rnn_provincias_resultados.json")

nacional_path = os.path.join(DATA_DIR, "dengue_procesado.csv")
provincias_path = os.path.join(DATA_DIR, "dengue_provincias_limpio.csv")


# ============================================================
# 3. CARGA DE MODELOS
# ============================================================

cnn_ready = False
ann_ready = False
rnn_ready = False
prov_ready = False

cnn_model = None
cnn_clases = None

ann_model = None
ann_prep = None

rnn_model = None
rnn_prep = None

prov_model = None
prov_prep = None
prov_resultados = None


# ---------------- CNN ----------------

try:
    cnn_model = tf.keras.models.load_model(
        cnn_model_path,
        compile=False
    )

    with open(cnn_clases_path, "r", encoding="utf-8") as archivo:
        cnn_clases = json.load(archivo)

    cnn_ready = True
    print("CNN MobileNetV2 cargada correctamente.")

except Exception as e:
    print(f"Error cargando CNN: {e}")


# ---------------- ANN ----------------

try:
    ann_model = tf.keras.models.load_model(
        ann_model_path,
        compile=False
    )

    ann_prep = joblib.load(ann_prep_path)

    ann_ready = True
    print("ANN cargada correctamente.")

except Exception as e:
    print(f"Error cargando ANN: {e}")


# ---------------- RNN ----------------

try:
    rnn_model = tf.keras.models.load_model(
        rnn_model_path,
        compile=False
    )

    rnn_prep = joblib.load(rnn_prep_path)

    # Verificar que existan todos los elementos necesarios
    claves_necesarias = [
        "columnas",
        "escalador_X",
        "escalador_y",
        "ventana",
        "horizonte"
    ]

    for clave in claves_necesarias:
        if clave not in rnn_prep:
            raise ValueError(
                f"Falta la clave {clave} en el preprocesador RNN."
            )

    if rnn_prep["ventana"] != 26:
        raise ValueError("La ventana del modelo no es de 26 semanas.")

    if rnn_prep["horizonte"] != 4:
        raise ValueError("El horizonte del modelo no es de 4 semanas.")

    rnn_ready = True
    print("SimpleRNN cargada correctamente.")

except Exception as e:
    print(f"Error cargando RNN: {e}")


# ---------------- Bonus: RNN por provincia ----------------

try:
    prov_model = tf.keras.models.load_model(
        prov_model_path,
        compile=False
    )

    prov_prep = joblib.load(prov_prep_path)

    for clave in ["columnas", "provincias", "escalador_X", "escalador_y", "ventana", "horizonte"]:
        if clave not in prov_prep:
            raise ValueError(f"Falta la clave {clave} en el preprocesador por provincia.")

    if prov_prep["ventana"] != 26 or prov_prep["horizonte"] != 4:
        raise ValueError("El modelo por provincia debe usar 26 semanas y pronosticar 4.")

    # Métricas calculadas en el notebook 05 (opcional: solo se muestran en la página de inicio)
    if os.path.exists(prov_resultados_path):
        with open(prov_resultados_path, "r", encoding="utf-8") as archivo:
            prov_resultados = json.load(archivo)

    prov_ready = True
    print("SimpleRNN por provincia cargada correctamente.")

except Exception as e:
    print(f"Error cargando la RNN por provincia: {e}")


# ---------------- Datos para gráficos y ejemplos ----------------

COLUMNAS_SEMANA = ["casos_dengue", "lluvia_mm", "temp_max_c", "temp_min_c", "humedad_rel_pct", "semana_epi"]


def leer_semanas(ruta):
    df = pd.read_csv(ruta, parse_dates=["fecha_inicio_semana"])
    return df.sort_values("fecha_inicio_semana").reset_index(drop=True)


datos_nacionales = None
datos_provincias = {}

try:
    datos_nacionales = leer_semanas(nacional_path)
except Exception as e:
    print(f"No se pudieron leer los datos nacionales: {e}")

try:
    todas = leer_semanas(provincias_path)
    datos_provincias = {
        provincia: grupo.reset_index(drop=True)
        for provincia, grupo in todas.groupby("provincia")
    }
except Exception as e:
    print(f"No se pudieron leer los datos por provincia: {e}")


# ============================================================
# 4. ESQUEMAS DE ENTRADA
# ============================================================

class Provincia(str, Enum):

    alajuela = "Alajuela"
    cartago = "Cartago"
    guanacaste = "Guanacaste"
    heredia = "Heredia"
    limon = "Limón"
    puntarenas = "Puntarenas"
    san_jose = "San José"


class AccionInput(BaseModel):

    casos_pred_4sem: float = Field(..., ge=0, description="Casos pronosticados para las próximas 4 semanas (suma).")
    casos_umbral_endemico: float = Field(..., gt=0, description="Umbral endémico de esas 4 semanas.")
    lluvia_14d_mm: float = Field(..., ge=0, description="Lluvia acumulada de las últimas 2 semanas (mm).")
    temp_max: float = Field(..., description="Temperatura máxima promedio (°C).")
    temp_min: float = Field(..., description="Temperatura mínima promedio (°C).")
    humedad: float = Field(..., ge=0, le=100, description="Humedad relativa promedio (%).")
    aegypti_detectado: int = Field(..., ge=0, le=1, description="1 si la CNN detectó Aedes aegypti.")


class SemanaHistoricaInput(BaseModel):

    casos_dengue: float = Field(..., ge=0)
    lluvia_mm: float = Field(..., ge=0)
    temp_max_c: float
    temp_min_c: float
    humedad_rel_pct: float = Field(..., ge=0, le=100)
    semana_epi: int = Field(..., ge=1, le=53)


class RNNForecastInput(BaseModel):

    semanas: list[SemanaHistoricaInput] = Field(
        ...,
        min_length=26,
        max_length=26,
        description="Historial de exactamente 26 semanas."
    )


class ProvinciaForecastInput(BaseModel):

    provincia: Provincia = Field(..., description="Provincia que se quiere pronosticar.")

    semanas: list[SemanaHistoricaInput] = Field(
        ...,
        min_length=26,
        max_length=26,
        description="Historial de exactamente 26 semanas de esa provincia."
    )


# ---------------- Ejemplos para /docs (con datos reales del proyecto) ----------------

def fila_de_corte(df, fecha_corte):
    """Posición de la última semana que empieza antes o en fecha_corte."""
    posiciones = np.flatnonzero(df["fecha_inicio_semana"].dt.date.to_numpy() <= fecha_corte)
    return int(posiciones[-1]) if len(posiciones) else -1


def semanas_hasta(df, fecha_corte):
    """Las 26 semanas que terminan en fecha_corte, en el formato de la API."""
    i = fila_de_corte(df, fecha_corte)
    filas = df.iloc[i - 25:i + 1]
    return [
        {
            "casos_dengue": float(fila.casos_dengue),
            "lluvia_mm": float(fila.lluvia_mm),
            "temp_max_c": float(fila.temp_max_c),
            "temp_min_c": float(fila.temp_min_c),
            "humedad_rel_pct": float(fila.humedad_rel_pct),
            "semana_epi": int(fila.semana_epi)
        }
        for fila in filas.itertuples()
    ]


EJEMPLOS_CASOS = None
EJEMPLOS_PROVINCIA = None

try:
    if datos_nacionales is not None:
        EJEMPLOS_CASOS = {
            "brote_2023": {
                "summary": "Brote de 2023 (datos reales, corte 15/10/2023)",
                "description": "Las 26 semanas que terminan el 15 de octubre de 2023, en pleno brote.",
                "value": {"semanas": semanas_hasta(datos_nacionales, date(2023, 10, 15))}
            },
            "tranquila_2022": {
                "summary": "Semana tranquila de 2022 (datos reales, corte 01/05/2022)",
                "description": "Un periodo sin brote, para comparar.",
                "value": {"semanas": semanas_hasta(datos_nacionales, date(2022, 5, 1))}
            }
        }

    if datos_provincias:
        EJEMPLOS_PROVINCIA = {
            "alajuela_2024": {
                "summary": "Alajuela, antes del pico de 2024 (sintético)",
                "value": {"provincia": "Alajuela",
                          "semanas": semanas_hasta(datos_provincias["Alajuela"], date(2024, 9, 22))}
            },
            "limon_2024": {
                "summary": "Limón, julio de 2024 (sintético)",
                "value": {"provincia": "Limón",
                          "semanas": semanas_hasta(datos_provincias["Limón"], date(2024, 7, 14))}
            },
            "cartago_2025": {
                "summary": "Cartago, casi sin casos (sintético)",
                "value": {"provincia": "Cartago",
                          "semanas": semanas_hasta(datos_provincias["Cartago"], date(2025, 12, 28))}
            }
        }

except Exception as e:
    print(f"No se pudieron preparar los ejemplos de /docs: {e}")


EJEMPLOS_ACCION = {
    "alerta": {
        "summary": "Brote: el pronóstico supera el doble del umbral",
        "value": {"casos_pred_4sem": 4825, "casos_umbral_endemico": 615, "lluvia_14d_mm": 96.4,
                  "temp_max": 29.5, "temp_min": 21.7, "humedad": 86.0, "aegypti_detectado": 1}
    },
    "fumigacion": {
        "summary": "Sobre el umbral y con Aedes aegypti detectado",
        "value": {"casos_pred_4sem": 1150, "casos_umbral_endemico": 900, "lluvia_14d_mm": 40,
                  "temp_max": 30.0, "temp_min": 21.0, "humedad": 80.0, "aegypti_detectado": 1}
    },
    "criaderos": {
        "summary": "Pocos casos, pero mucha lluvia y calor",
        "value": {"casos_pred_4sem": 0, "casos_umbral_endemico": 900, "lluvia_14d_mm": 200,
                  "temp_max": 30.0, "temp_min": 22.0, "humedad": 88.0, "aegypti_detectado": 0}
    },
    "monitoreo": {
        "summary": "Semana tranquila",
        "value": {"casos_pred_4sem": 200, "casos_umbral_endemico": 900, "lluvia_14d_mm": 10,
                  "temp_max": 27.0, "temp_min": 18.0, "humedad": 70.0, "aegypti_detectado": 0}
    }
}


# ============================================================
# 5. ESTADO DE LA API
# ============================================================

def estado_servicios():

    return {
        "status": "online",
        "cnn_ready": cnn_ready,
        "ann_ready": ann_ready,
        "rnn_ready": rnn_ready,
        "rnn_provincias_ready": prov_ready,
        "models_directory": MODELS_DIR
    }


# En el navegador, "/" muestra la página de inicio (sección 11).
# La app de Streamlit y curl reciben el JSON de siempre.
@app.get("/", include_in_schema=False)
def inicio(request: Request):

    if "text/html" in request.headers.get("accept", ""):
        return HTMLResponse(pagina_inicio())

    return estado_servicios()


@app.get("/estado", summary="Estado de los servicios", tags=[TAG_ESTADO])
def health_check():

    return estado_servicios()


# ============================================================
# 6. MÓDULO ANN: RECOMENDACIÓN DE ACCIONES
# ============================================================

MOTIVOS = {

    "alerta_sanitaria":
        "Los casos predichos superan el doble del umbral endémico.",

    "fumigacion_focalizada":
        "Los casos superan el umbral endémico y se detectó Aedes aegypti.",

    "eliminacion_criaderos":
        "Las condiciones de lluvia y temperatura favorecen los criaderos.",

    "monitoreo_normal":
        "Se recomienda mantener la vigilancia y el monitoreo de rutina."
}


@app.post("/predict/accion", summary="Recomendar acción preventiva", tags=[TAG_ANN])

def predict_accion(data: Annotated[AccionInput, Body(openapi_examples=EJEMPLOS_ACCION)]):
    """
    Devuelve la acción recomendada, la confianza, la probabilidad de cada acción y el motivo.

    Las cuatro acciones posibles: **alerta sanitaria**, **fumigación focalizada**,
    **eliminación de criaderos** y **monitoreo normal**.
    """

    if not ann_ready:
        raise HTTPException(
            status_code=503,
            detail="El modelo ANN no está disponible."
        )

    try:

        fila = pd.DataFrame([data.model_dump()])

        # Ordenar las variables como durante el entrenamiento
        columnas = ann_prep["columnas"]

        fila = fila[columnas]

        # Aplicar escalado
        entrada = ann_prep["escalador"].transform(fila)

        # Predicción de la ANN
        probabilidades = ann_model.predict(
            entrada,
            verbose=0
        )[0]

        indice = int(np.argmax(probabilidades))

        accion = str(
            ann_prep["codificador"].classes_[indice]
        )

        return {
            "accion": accion,

            "confianza_pct": round(
                float(probabilidades[indice]) * 100, 2
            ),

            "probabilidades": {
                str(clase): round(float(prob) * 100, 2)
                for clase, prob in zip(
                    ann_prep["codificador"].classes_,
                    probabilidades
                )
            },

            "motivo": MOTIVOS.get(
                accion,
                "Acción preventiva recomendada por el modelo."
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error en la predicción ANN: {str(e)}"
        )


# ============================================================
# 7. MÓDULO RNN: PREDICCIÓN DE CASOS DE DENGUE
# ============================================================

# /predict/casos es el nombre que pide el enunciado. /predict/rnn se mantiene
# por compatibilidad (hace exactamente lo mismo) y no aparece en /docs.
@app.post("/predict/casos", summary="Pronóstico de casos de dengue (SimpleRNN)", tags=[TAG_RNN])
@app.post("/predict/rnn", include_in_schema=False)
def predict_rnn(data: Annotated[RNNForecastInput, Body(openapi_examples=EJEMPLOS_CASOS)]):
    """
    Recibe **26 semanas** de casos y clima a nivel nacional y pronostica los casos
    de las **4 semanas siguientes**.

    Para ver el mismo pronóstico dibujado contra el umbral endémico, usa
    `GET /visual/nacional`.
    """

    if not rnn_ready:

        raise HTTPException(
            status_code=503,
            detail="El modelo SimpleRNN no está disponible."
        )

    try:

        # ----------------------------------------------------
        # A. Convertir las 26 semanas en un DataFrame
        # ----------------------------------------------------

        df = pd.DataFrame([
            semana.model_dump()
            for semana in data.semanas
        ])

        # Comprobar que los valores sean válidos
        valores = df.select_dtypes(include=[np.number]).to_numpy()

        if not np.isfinite(valores).all():
            raise ValueError(
                "Los datos contienen valores no válidos."
            )

        # ----------------------------------------------------
        # B. Crear las variables utilizadas en entrenamiento
        # ----------------------------------------------------

        df["log_casos"] = np.log1p(
            df["casos_dengue"]
        )

        df["semana_sin"] = np.sin(
            2 * np.pi * df["semana_epi"] / 52
        )

        df["semana_cos"] = np.cos(
            2 * np.pi * df["semana_epi"] / 52
        )

        # ----------------------------------------------------
        # C. Mantener el orden exacto de las columnas
        # ----------------------------------------------------

        columnas = rnn_prep["columnas"]

        df_modelo = df[columnas]

        # ----------------------------------------------------
        # D. Escalar los datos
        # ----------------------------------------------------

        # El escalador se entrenó con números sin nombres de columna,
        # así que se le pasan igual (en el mismo orden): evita un aviso de sklearn
        entrada_escalada = rnn_prep["escalador_X"].transform(
            df_modelo.to_numpy()
        )

        # Formato que necesita la red:
        # (1 muestra, 26 semanas, 7 variables)

        entrada = entrada_escalada.reshape(
            1,
            rnn_prep["ventana"],
            len(columnas)
        )

        # ----------------------------------------------------
        # E. Ejecutar el modelo entrenado
        # ----------------------------------------------------

        predicciones_escaladas = rnn_model.predict(
            entrada,
            verbose=0
        )

        # Verificar las cuatro salidas
        if predicciones_escaladas.shape != (1, 4):
            raise ValueError(
                "El modelo no devolvió las cuatro semanas esperadas."
            )

        # ----------------------------------------------------
        # F. Recuperar la escala original
        # ----------------------------------------------------

        predicciones_log = rnn_prep["escalador_y"].inverse_transform(
            predicciones_escaladas.reshape(-1, 1)
        )

        # Deshacer log1p
        predicciones = np.expm1(
            predicciones_log
        ).flatten()

        # Evitar resultados negativos
        predicciones = np.maximum(predicciones, 0)

        predicciones_limpias = [
            round(float(x), 1)
            for x in predicciones
        ]

        # ----------------------------------------------------
        # G. Preparar la respuesta
        # ----------------------------------------------------

        casos_historicos = [
            float(semana.casos_dengue)
            for semana in data.semanas
        ]

        tendencia = (
            "Ascendente"
            if predicciones_limpias[-1] > casos_historicos[-1]
            else "Descendente / Estable"
        )

        return {

            "modelo_utilizado": "SimpleRNN entrenada",

            "semanas_historicas": 26,

            "variables_utilizadas": columnas,

            "historico_ingresado": casos_historicos,

            "semanas_proyectadas": [
                "Semana +1",
                "Semana +2",
                "Semana +3",
                "Semana +4"
            ],

            "pronostico_casos": predicciones_limpias,

            "total_estimado_periodo": round(
                float(sum(predicciones_limpias)), 1
            ),

            "tendencia_general": tendencia
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error al generar el pronóstico RNN: {str(e)}"
        )


# ============================================================
# 8. MÓDULO CNN: IDENTIFICACIÓN DE MOSQUITOS
# ============================================================

@app.post("/predict/mosquito", summary="Identificar especie de mosquito", tags=[TAG_CNN])

async def predict_mosquito(file: UploadFile = File(..., description="Foto del mosquito (JPG o PNG).")):
    """
    Sube una foto y la red (MobileNetV2 con transfer learning) devuelve la especie,
    la confianza y la probabilidad de cada clase.
    """

    if not cnn_ready:

        raise HTTPException(
            status_code=503,
            detail="El modelo CNN no está disponible."
        )

    if file.content_type not in (
        "image/jpeg",
        "image/png"
    ):

        raise HTTPException(
            status_code=400,
            detail="Solo se permiten imágenes JPG o PNG."
        )

    try:

        # Leer imagen
        contenido = await file.read()

        if not contenido:
            raise ValueError("La imagen está vacía.")

        # Procesar fotografía igual que en el entrenamiento
        # (src/train/entrenar_mosquitos.py): leerla con TensorFlow en RGB
        # y redimensionarla a 128x128 con tf.image.resize
        imagen = tf.io.decode_image(
            contenido,
            channels=3,
            expand_animations=False
        )

        imagen = tf.image.resize(imagen, (128, 128))

        imagen_array = imagen.numpy()

        # Preprocesamiento de MobileNetV2
        imagen_array = preprocess_input(
            imagen_array
        )

        # Agregar dimensión del lote
        imagen_array = np.expand_dims(
            imagen_array,
            axis=0
        )

        # Ejecutar CNN
        probabilidades = cnn_model.predict(
            imagen_array,
            verbose=0
        )[0]

        # Obtener especie
        indice = int(np.argmax(probabilidades))

        especie = cnn_clases[str(indice)]

        return {

            "especie": especie,

            "confianza_pct": round(
                float(probabilidades[indice]) * 100,
                2
            ),

            "probabilidades": {
                cnn_clases[str(i)]: round(
                    float(p) * 100,
                    2
                )
                for i, p in enumerate(probabilidades)
            }
        }

    except tf.errors.InvalidArgumentError:

        # TensorFlow no pudo leer el archivo como imagen
        raise HTTPException(
            status_code=400,
            detail="No se pudo leer la imagen: el archivo está dañado o no es un JPG o PNG válido."
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"No se pudo procesar la imagen: {str(e)}"
        )


# ============================================================
# 9. BONUS: PRONÓSTICO POR PROVINCIA (DATOS SINTÉTICOS)
# ============================================================

AVISO_SINTETICO = (
    "Datos sintéticos del paquete del curso (vectoria_dengue_semanal.csv): "
    "no son cifras oficiales."
)


def pronosticar(modelo, prep, semanas, provincia=None):
    """
    Los mismos pasos de /predict/casos (variables, escalado, red, quitar el logaritmo),
    para 26 semanas en un DataFrame. Con provincia, agrega su one-hot (prov_<nombre>).
    """
    df = semanas[COLUMNAS_SEMANA].astype(float).reset_index(drop=True)

    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("Los datos contienen valores no válidos.")

    df["log_casos"] = np.log1p(df["casos_dengue"])
    df["semana_sin"] = np.sin(2 * np.pi * df["semana_epi"] / 52)
    df["semana_cos"] = np.cos(2 * np.pi * df["semana_epi"] / 52)

    if provincia is not None:
        for nombre in prep["provincias"]:
            df[f"prov_{nombre}"] = 1.0 if nombre == provincia else 0.0

    columnas = prep["columnas"]
    entrada = prep["escalador_X"].transform(df[columnas].to_numpy())
    entrada = entrada.reshape(1, prep["ventana"], len(columnas))

    salida = modelo.predict(entrada, verbose=0)

    if salida.shape != (1, 4):
        raise ValueError("El modelo no devolvió las cuatro semanas esperadas.")

    casos = np.expm1(prep["escalador_y"].inverse_transform(salida.reshape(-1, 1))).ravel()

    return [round(float(x), 1) for x in np.maximum(casos, 0)]


@app.post(
    "/predict/casos/provincia",
    summary="Pronóstico de casos por provincia (bonus, datos sintéticos)",
    tags=[TAG_PROVINCIAS]
)
def predict_provincia(data: Annotated[ProvinciaForecastInput, Body(openapi_examples=EJEMPLOS_PROVINCIA)]):
    """
    Igual que `/predict/casos`, pero para una de las **7 provincias**. Una sola SimpleRNN
    aprendió las 7 provincias a la vez y recibe cuál es como variable *one-hot*
    (ver `notebooks/05_Bonus_Provincias.ipynb`).

    ⚠️ Se entrenó con **datos sintéticos**: sirve para mostrar el método, no para
    pronosticar el dengue real de cada provincia.
    """

    if not prov_ready:
        raise HTTPException(
            status_code=503,
            detail="El modelo por provincia no está disponible."
        )

    try:

        semanas = pd.DataFrame([semana.model_dump() for semana in data.semanas])
        provincia = data.provincia.value
        predicciones = pronosticar(prov_model, prov_prep, semanas, provincia)
        historicos = [float(x) for x in semanas["casos_dengue"]]

        return {
            "provincia": provincia,
            "modelo_utilizado": "SimpleRNN por provincia (bonus)",
            "datos": "sinteticos",
            "aviso": AVISO_SINTETICO,
            "semanas_historicas": 26,
            "variables_utilizadas": prov_prep["columnas"],
            "historico_ingresado": historicos,
            "semanas_proyectadas": ["Semana +1", "Semana +2", "Semana +3", "Semana +4"],
            "pronostico_casos": predicciones,
            "total_estimado_periodo": round(float(sum(predicciones)), 1),
            "tendencia_general": (
                "Ascendente" if predicciones[-1] > historicos[-1] else "Descendente / Estable"
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error al generar el pronóstico por provincia: {str(e)}"
        )


# ============================================================
# 10. VISUALIZACIONES (GRÁFICOS PNG)
# ============================================================

COLOR = {
    "fondo": "#0F2236",
    "panel": "#132B44",
    "rejilla": "#22405E",
    "texto": "#E6F1F5",
    "suave": "#8FA9BF",
    "historia": "#5BC0EB",
    "pronostico": "#2EE6C5",
    "umbral": "#FF6B6B",
    "reales": "#F6AE2D",
    "verde": "#2BB673"
}

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

matplotlib.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

RESPUESTA_PNG = {200: {"content": {"image/png": {}}, "description": "Gráfico PNG"}}


def semanas_siguientes(df, i, n=4):
    """Fecha, año y semana epidemiológica de las n semanas después de la fila i (igual que la app)."""
    salida = []
    fecha = df["fecha_inicio_semana"].iloc[i]
    anio = int(df["anio"].iloc[i])
    semana = int(df["semana_epi"].iloc[i])

    for k in range(1, n + 1):
        if i + k < len(df):
            fila = df.iloc[i + k]
            fecha, anio, semana = fila["fecha_inicio_semana"], int(fila["anio"]), int(fila["semana_epi"])
        else:
            fecha = fecha + pd.Timedelta(days=7)
            semana += 1
            if semana > 52:
                semana, anio = 1, anio + 1
        salida.append((fecha, anio, semana))

    return salida


def umbral_semanal(df, anio, semana):
    """Percentil 75 de esa misma semana en los 5 años anteriores (hacen falta al menos 3)."""
    valores = df.loc[df["anio"].between(anio - 5, anio - 1) & (df["semana_epi"] == semana), "casos_dengue"]
    return float(np.percentile(valores, 75)) if len(valores) >= 3 else None


def umbral_4_semanas(df, i):
    """Umbral de las 4 semanas después de la fila i: la misma fórmula de vectoria_acciones.csv y de la app."""
    ultima = df.iloc[i]
    sumas = []

    for k in range(1, 6):
        misma = np.flatnonzero(((df["anio"] == ultima["anio"] - k) & (df["semana_epi"] == ultima["semana_epi"])).to_numpy())
        if len(misma) > 0:
            siguientes = df["casos_dengue"].iloc[misma[0] + 1:misma[0] + 5]
            if len(siguientes) == 4:
                sumas.append(float(siguientes.sum()))

    if not sumas:
        return float(round(df["casos_dengue"].mean() * 4))

    return max(1.0, float(round(np.percentile(sumas, 75))))


def elegir_corte(df, fecha_corte):
    """Fila de la semana de corte: la última semana con datos si no se indica fecha."""
    if fecha_corte is None:
        return len(df) - 1

    i = fila_de_corte(df, fecha_corte)

    if i < 25:
        minima = df["fecha_inicio_semana"].iloc[25].date()
        raise HTTPException(
            status_code=400,
            detail=f"La fecha de corte debe ser desde el {minima:%d/%m/%Y}: el modelo necesita 26 semanas antes."
        )

    return i


def datos_del_grafico(df, i, modelo, prep, provincia=None):
    """Historia (52 semanas), pronóstico, umbral y casos reales con corte en la fila i."""
    prediccion = pronosticar(modelo, prep, df.iloc[i - 25:i + 1], provincia)
    futuras = semanas_siguientes(df, i)
    historia = df.iloc[max(0, i - 51):i + 1]

    return {
        "fechas_hist": list(historia["fecha_inicio_semana"]),
        "hist": [float(x) for x in historia["casos_dengue"]],
        "fechas_fut": [f for f, _, _ in futuras],
        "pred": prediccion,
        "umbral_hist": [umbral_semanal(df, int(a), int(s)) for a, s in zip(historia["anio"], historia["semana_epi"])],
        "umbral_fut": [umbral_semanal(df, a, s) for _, a, s in futuras],
        "reales": [float(df["casos_dengue"].iloc[i + k]) for k in range(1, 5) if i + k < len(df)],
        "umbral_4": umbral_4_semanas(df, i),
        "corte": df["fecha_inicio_semana"].iloc[i]
    }


def formato_fecha(valor, _=None):
    fecha = mdates.num2date(valor)
    return f"{MESES[fecha.month - 1]} {fecha.year}"


def dibujar_pronostico(ax, d, compacto=False):
    """Dibuja en ax la historia, el umbral, el pronóstico con brillo y los casos reales."""
    ax.set_facecolor(COLOR["panel"])
    fh, ff = d["fechas_hist"], d["fechas_fut"]
    fin = ff[-1] + pd.Timedelta(days=4)

    # Zona del pronóstico y ventana que ve el modelo
    ax.axvspan(fh[-1], fin, color=COLOR["pronostico"], alpha=0.08, lw=0)
    ax.axvspan(fh[max(0, len(fh) - 26)], fh[-1], color="white", alpha=0.025, lw=0)

    # Umbral endémico semanal
    umbral = [np.nan if u is None else u for u in d["umbral_hist"] + d["umbral_fut"]]
    ax.plot(fh + ff, umbral, color=COLOR["umbral"], lw=1.6, ls=(0, (2, 2)), label="Umbral endémico semanal")

    # Casos registrados con relleno
    ax.fill_between(fh, d["hist"], color=COLOR["historia"], alpha=0.16, lw=0)
    ax.plot(fh, d["hist"], color=COLOR["historia"], lw=2.2, label="Casos registrados")

    # Casos reales de las semanas pronosticadas (si ya existen)
    if d["reales"]:
        ax.plot([fh[-1]] + ff[:len(d["reales"])], [d["hist"][-1]] + d["reales"], color=COLOR["reales"],
                lw=1.8, marker="o", ms=5 if compacto else 6, label="Casos reales (para comparar)")

    # Pronóstico con un brillo alrededor
    x = [fh[-1]] + ff
    y = [d["hist"][-1]] + d["pred"]
    for ancho, alfa in [(12, 0.05), (7, 0.10), (4, 0.18)]:
        ax.plot(x, y, color=COLOR["pronostico"], lw=ancho, alpha=alfa, solid_capstyle="round")
    ax.plot(x, y, color=COLOR["pronostico"], lw=2.6, ls=(0, (4, 1.5)), label="Pronóstico SimpleRNN")
    ax.scatter(ff, d["pred"], s=34 if compacto else 60, color=COLOR["pronostico"],
               edgecolor=COLOR["fondo"], linewidth=1.6, zorder=6)

    if not compacto:
        # Cada etiqueta va arriba si el punto es un pico y abajo si es un valle, para no tapar la línea
        for k, (xf, yf) in enumerate(zip(ff, d["pred"])):
            vecinos = [y[k]] + ([y[k + 2]] if k + 2 < len(y) else [])
            arriba = yf >= np.mean(vecinos)
            ax.annotate(f"{yf:,.0f}", (xf, yf), textcoords="offset points", xytext=(0, 13 if arriba else -21),
                        ha="center", color=COLOR["pronostico"], fontsize=10, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc=COLOR["fondo"], ec="none", alpha=0.85), zorder=7)

    ax.axvline(fh[-1], color=COLOR["suave"], lw=1, ls=":")

    # Ejes
    maximo = max([v for v in d["hist"] + d["pred"] + d["reales"] + umbral if np.isfinite(v)] + [1])
    ax.set_ylim(0, maximo * (1.25 if compacto else 1.22))
    ax.set_xlim(fh[0], fin)
    ax.grid(axis="y", color=COLOR["rejilla"], lw=0.8)
    ax.set_axisbelow(True)
    for borde in ax.spines.values():
        borde.set_visible(False)
    ax.tick_params(colors=COLOR["suave"], labelsize=8 if compacto else 9.5, length=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3 if compacto else 2))
    ax.xaxis.set_major_formatter(FuncFormatter(formato_fecha))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    if not compacto:
        ax.text(fh[-1], maximo * 1.17, "  pronóstico →", color=COLOR["pronostico"], fontsize=10, va="center")
        ax.text(fh[max(0, len(fh) - 26)], maximo * 1.17, "  ventana del modelo (26 semanas)",
                color=COLOR["suave"], fontsize=9, va="center")


def estado_umbral(total, umbral):
    """Texto y color del pronóstico contra el umbral."""
    if total > 2 * umbral:
        return "▲ Más del doble del umbral", COLOR["umbral"]
    if total > umbral:
        return "▲ Sobre el umbral", "#FF9F68"
    return "▼ Bajo el umbral", COLOR["verde"]


def chip(fig, x, y, titulo, valor, color):
    fig.text(x, y + 0.032, titulo.upper(), color=COLOR["suave"], fontsize=8.5, ha="left", fontweight="bold")
    fig.text(x, y, valor, color=color, fontsize=16, ha="left", fontweight="bold")


def a_png(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=110, facecolor=fig.get_facecolor())
    return buffer.getvalue()


def grafico_simple(d, titulo, subtitulo, pie):
    """Un pronóstico grande, con chips de resumen arriba."""
    fig = Figure(figsize=(12.5, 7.4))
    fig.patch.set_facecolor(COLOR["fondo"])

    fig.text(0.05, 0.935, titulo, color=COLOR["texto"], fontsize=21, fontweight="bold")
    fig.text(0.05, 0.895, subtitulo, color=COLOR["suave"], fontsize=11)

    # Fila de indicadores debajo del título
    total = sum(d["pred"])
    texto, color = estado_umbral(total, d["umbral_4"])
    chip(fig, 0.05, 0.80, "Próximas 4 semanas", f"{total:,.0f} casos", COLOR["pronostico"])
    chip(fig, 0.25, 0.80, "Umbral endémico", f"{d['umbral_4']:,.0f} casos", COLOR["umbral"])
    chip(fig, 0.45, 0.80, "Estado", texto, color)
    if d["reales"]:
        chip(fig, 0.72, 0.80, "Casos reales", f"{sum(d['reales']):,.0f} casos", COLOR["reales"])

    ax = fig.add_axes([0.06, 0.16, 0.91, 0.57])
    dibujar_pronostico(ax, d)

    fig.legend(*ax.get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.05, 0.035), ncol=4,
               frameon=False, labelcolor=COLOR["texto"], fontsize=10)
    fig.text(0.97, 0.015, pie, color=COLOR["suave"], fontsize=8.5, ha="right")
    return a_png(fig)


@lru_cache(maxsize=256)
def png_nacional(i):
    d = datos_del_grafico(datos_nacionales, i, rnn_model, rnn_prep)
    reales = " · con los casos reales para comparar" if d["reales"] else " · pronóstico a futuro"
    return grafico_simple(
        d,
        "Pronóstico de dengue · Costa Rica",
        f"SimpleRNN · semana de corte {d['corte']:%d/%m/%Y}{reales} · datos reales (OPS/PAHO y NASA POWER)",
        "VectorIA API · GET /visual/nacional · no diagnostica dengue en personas"
    )


@lru_cache(maxsize=512)
def png_provincia(provincia, i):
    df = datos_provincias[provincia]
    d = datos_del_grafico(df, i, prov_model, prov_prep, provincia)
    return grafico_simple(
        d,
        f"Pronóstico de dengue · {provincia}",
        f"SimpleRNN por provincia · semana de corte {d['corte']:%d/%m/%Y} · ⚠ datos sintéticos (bonus)",
        f"VectorIA API · GET /visual/provincia/{provincia} · datos sintéticos, no son cifras oficiales"
    )


@lru_cache(maxsize=128)
def png_panel(i):
    """Las 7 provincias a la vez, más un ranking del pronóstico contra el umbral."""
    fig = Figure(figsize=(15, 14.5))
    fig.patch.set_facecolor(COLOR["fondo"])
    rejilla = fig.add_gridspec(4, 2, left=0.055, right=0.975, top=0.885, bottom=0.07, hspace=0.42, wspace=0.13)

    primera = next(iter(datos_provincias.values()))
    corte = primera["fecha_inicio_semana"].iloc[i]
    fig.text(0.055, 0.955, "Pronóstico de dengue por provincia", color=COLOR["texto"], fontsize=24, fontweight="bold")
    fig.text(0.055, 0.928, f"SimpleRNN por provincia · semana de corte {corte:%d/%m/%Y} · próximas 4 semanas contra el "
             "umbral endémico · ⚠ datos sintéticos (bonus)", color=COLOR["suave"], fontsize=11.5)

    resumen = []
    ax = None

    for k, provincia in enumerate(sorted(datos_provincias)):
        df = datos_provincias[provincia]
        d = datos_del_grafico(df, i, prov_model, prov_prep, provincia)
        total = sum(d["pred"])
        resumen.append((provincia, total, d["umbral_4"]))

        ax = fig.add_subplot(rejilla[k // 2, k % 2])
        dibujar_pronostico(ax, d, compacto=True)
        texto, color = estado_umbral(total, d["umbral_4"])
        ax.set_title(provincia, loc="left", color=COLOR["texto"], fontsize=14, fontweight="bold", pad=8)
        ax.set_title(f"{total:,.0f} casos en 4 semanas · {texto}", loc="right", color=color, fontsize=10.5,
                     fontweight="bold", pad=9)

    manejadores, etiquetas = ax.get_legend_handles_labels()

    # Ranking: pronóstico como % del umbral
    ranking = fig.add_subplot(rejilla[3, 1])
    ranking.set_facecolor(COLOR["panel"])
    resumen.sort(key=lambda r: r[1] / r[2])
    nombres = [r[0] for r in resumen]
    porcentajes = [100 * r[1] / r[2] for r in resumen]
    colores = [estado_umbral(r[1], r[2])[1] for r in resumen]
    barras = ranking.barh(nombres, porcentajes, color=colores, height=0.62)
    ranking.axvline(100, color=COLOR["texto"], lw=1, ls=":")
    ranking.set_ylim(-1.1, len(nombres) - 0.5)
    ranking.text(100, -0.85, " umbral = 100 %", color=COLOR["texto"], fontsize=9, va="center")
    for barra, valor in zip(barras, porcentajes):
        ranking.text(barra.get_width(), barra.get_y() + barra.get_height() / 2, f"  {valor:,.0f} %",
                     va="center", color=COLOR["texto"], fontsize=9.5, fontweight="bold")
    ranking.set_xlim(0, max(porcentajes + [100]) * 1.28)
    ranking.set_title("Pronóstico de 4 semanas como % del umbral", loc="left", color=COLOR["texto"],
                      fontsize=14, fontweight="bold", pad=8)
    ranking.grid(axis="x", color=COLOR["rejilla"], lw=0.8)
    ranking.set_axisbelow(True)
    for borde in ranking.spines.values():
        borde.set_visible(False)
    ranking.tick_params(colors=COLOR["texto"], labelsize=10, length=0)
    ranking.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f} %"))

    fig.legend(manejadores, etiquetas, loc="lower left", bbox_to_anchor=(0.05, 0.012), ncol=4, frameon=False,
               labelcolor=COLOR["texto"], fontsize=10.5)
    fig.text(0.975, 0.022, "VectorIA API · GET /visual/provincias · datos sintéticos, no son cifras oficiales",
             color=COLOR["suave"], fontsize=9, ha="right")
    return a_png(fig)


DESCRIPCION_CORTE = "Última semana que conoce el modelo (AAAA-MM-DD). Vacío = la última semana con datos."


@app.get("/visual/nacional", summary="Gráfico del pronóstico nacional (datos reales)", tags=[TAG_VISUAL],
         response_class=Response, responses=RESPUESTA_PNG)
def visual_nacional(
    fecha_corte: Annotated[Optional[date], Query(description=DESCRIPCION_CORTE, examples=["2023-10-15"])] = None
):
    """
    Dibuja el pronóstico de la SimpleRNN nacional contra el umbral endémico semanal.
    Si la fecha tiene semanas reales después, también las dibuja para comparar.

    Prueba con `2023-10-15` (brote de 2023) o déjalo vacío para el pronóstico a futuro.
    """
    if not rnn_ready or datos_nacionales is None:
        raise HTTPException(status_code=503, detail="El modelo SimpleRNN o sus datos no están disponibles.")

    i = elegir_corte(datos_nacionales, fecha_corte)
    return Response(content=png_nacional(i), media_type="image/png")


@app.get("/visual/provincia/{provincia}", summary="Gráfico del pronóstico de una provincia (bonus)",
         tags=[TAG_VISUAL], response_class=Response, responses=RESPUESTA_PNG)
def visual_provincia(
    provincia: Annotated[Provincia, Path(description="Provincia que se quiere ver.")],
    fecha_corte: Annotated[Optional[date], Query(description=DESCRIPCION_CORTE, examples=["2024-09-22"])] = None
):
    """
    Dibuja el pronóstico de la SimpleRNN por provincia contra su umbral endémico.
    ⚠️ Datos sintéticos.
    """
    if not prov_ready or provincia.value not in datos_provincias:
        raise HTTPException(status_code=503, detail="El modelo por provincia o sus datos no están disponibles.")

    i = elegir_corte(datos_provincias[provincia.value], fecha_corte)
    return Response(content=png_provincia(provincia.value, i), media_type="image/png")


@app.get("/visual/provincias", summary="Panel de las 7 provincias (bonus)",
         tags=[TAG_VISUAL], response_class=Response, responses=RESPUESTA_PNG)
def visual_provincias(
    fecha_corte: Annotated[Optional[date], Query(description=DESCRIPCION_CORTE, examples=["2024-07-14"])] = None
):
    """
    Las 7 provincias en un solo gráfico, cada una contra su umbral, más un ranking
    de qué provincias superan su umbral en las próximas 4 semanas. ⚠️ Datos sintéticos.
    """
    if not prov_ready or len(datos_provincias) != 7:
        raise HTTPException(status_code=503, detail="El modelo por provincia o sus datos no están disponibles.")

    i = elegir_corte(next(iter(datos_provincias.values())), fecha_corte)
    return Response(content=png_panel(i), media_type="image/png")


# ============================================================
# 11. PÁGINA DE INICIO Y DOCUMENTACIÓN
# ============================================================

ICONO = (
    "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
    "<text y='.9em' font-size='90'>🦟</text></svg>"
)

ESTILO_DOCS = """
body { margin: 0; background: #F3F7FA; font-family: "Segoe UI", system-ui, sans-serif; }
.vx-cabecera { background: linear-gradient(120deg, #0F2236 0%, #183B56 55%, #1B6F7A 100%); color: #fff;
  padding: 26px 0 30px; position: relative; overflow: hidden; }
.vx-cabecera::after { content: ""; position: absolute; right: -120px; top: -160px; width: 420px; height: 420px;
  border-radius: 50%; background: radial-gradient(circle, rgba(46,230,197,.35), rgba(46,230,197,0) 70%); }
.vx-interior { max-width: 1460px; margin: 0 auto; padding: 0 20px; position: relative; z-index: 1; }
.vx-marca { display: flex; align-items: center; gap: 14px; }
.vx-logo { width: 52px; height: 52px; border-radius: 14px; background: rgba(255,255,255,.12);
  display: grid; place-items: center; font-size: 30px; border: 1px solid rgba(255,255,255,.18); }
.vx-titulo { font-size: 30px; font-weight: 800; letter-spacing: -.5px; }
.vx-titulo span { color: #2EE6C5; }
.vx-sub { color: #B8CCDA; font-size: 14px; margin-top: 2px; }
.vx-pastillas { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.vx-pastilla { background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.18); border-radius: 999px;
  padding: 6px 13px; font-size: 13px; color: #E6F1F5; text-decoration: none; }
.vx-pastilla b { color: #2EE6C5; }
.vx-pastilla.ok::before { content: "●"; color: #2EE6C5; margin-right: 6px; }
.vx-pastilla.no::before { content: "●"; color: #FF6B6B; margin-right: 6px; }
a.vx-pastilla:hover { background: rgba(255,255,255,.2); }
.swagger-ui .wrapper { max-width: 1460px; }
.swagger-ui .information-container { padding-top: 6px; }
.swagger-ui .info { margin: 26px 0 10px; }
.swagger-ui .info .title { color: #102A43; font-weight: 800; }
.swagger-ui .info .title small { background: #20A39E; }
.swagger-ui .info .title small.version-stamp { background: #102A43; }
.swagger-ui .info .renderedMarkdown table { border-radius: 10px; overflow: hidden; border-collapse: collapse; }
.swagger-ui .info .renderedMarkdown th { background: #102A43; color: #fff; padding: 8px 12px; }
.swagger-ui .info .renderedMarkdown td { background: #fff; padding: 8px 12px; border-bottom: 1px solid #E2EAF0; }
.swagger-ui .info .renderedMarkdown blockquote { border-left: 4px solid #E9A23B; background: #FFF7E8; margin: 12px 0;
  padding: 8px 14px; border-radius: 0 8px 8px 0; }
.swagger-ui .opblock-tag { font-size: 21px; color: #102A43; border-bottom: 2px solid #E2EAF0; padding: 14px 6px; }
.swagger-ui .opblock-tag small { font-size: 14px; color: #52677D; }
.swagger-ui .opblock { border-radius: 12px; box-shadow: 0 2px 10px rgba(16,42,67,.06); margin-bottom: 12px; }
.swagger-ui .opblock .opblock-summary { padding: 9px 12px; }
.swagger-ui .opblock .opblock-summary-method { border-radius: 8px; min-width: 76px; font-weight: 800; }
.swagger-ui .opblock.opblock-post { border-color: #20A39E; background: rgba(32,163,158,.06); }
.swagger-ui .opblock.opblock-post .opblock-summary-method { background: #20A39E; }
.swagger-ui .opblock.opblock-post .opblock-summary { border-color: #20A39E; }
.swagger-ui .opblock.opblock-get { border-color: #247BA0; background: rgba(36,123,160,.06); }
.swagger-ui .opblock.opblock-get .opblock-summary-method { background: #247BA0; }
.swagger-ui .opblock.opblock-get .opblock-summary { border-color: #247BA0; }
.swagger-ui .opblock-summary-path { font-weight: 700; }
.swagger-ui .btn.execute { background: #20A39E; border-color: #20A39E; border-radius: 8px; }
.swagger-ui .btn.try-out__btn { border-radius: 8px; border-color: #20A39E; color: #20A39E; }
.swagger-ui .btn.try-out__btn.cancel { color: #DC6262; border-color: #DC6262; }
.swagger-ui .response-col_description img, .swagger-ui .live-responses-table img { max-width: 100%;
  border-radius: 12px; box-shadow: 0 10px 30px rgba(16,42,67,.25); }
.swagger-ui section.models { display: none; }
.swagger-ui .scheme-container { box-shadow: none; background: transparent; padding: 0; }
"""


def pastillas_estado():
    servicios = [("CNN", cnn_ready), ("SimpleRNN", rnn_ready), ("ANN", ann_ready), ("Provincias", prov_ready)]
    return "".join(
        f'<span class="vx-pastilla {"ok" if listo else "no"}">{nombre}</span>' for nombre, listo in servicios
    )


@app.get("/docs", include_in_schema=False)
def documentacion():
    """Swagger UI con los colores y la cabecera de VectorIA."""
    html = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="VectorIA API · Documentación",
        swagger_favicon_url=ICONO,
        swagger_ui_parameters={
            "docExpansion": "list",
            "defaultModelsExpandDepth": -1,
            "displayRequestDuration": True,
            "tryItOutEnabled": True,
            "syntaxHighlight": {"theme": "obsidian"}
        }
    ).body.decode("utf-8")

    cabecera = f"""
<div class="vx-cabecera"><div class="vx-interior">
  <div class="vx-marca"><div class="vx-logo">🦟</div><div>
    <div class="vx-titulo">Vector<span>IA</span> API</div>
    <div class="vx-sub">Vigilancia y prevención del dengue en Costa Rica · Inteligencia Artificial Aplicada · CUC</div>
  </div></div>
  <div class="vx-pastillas">{pastillas_estado()}
    <a class="vx-pastilla" href="/">← Página de inicio</a>
    <a class="vx-pastilla" href="/redoc">ReDoc</a>
  </div>
</div></div>
"""
    html = html.replace("</head>", f"<style>{ESTILO_DOCS}</style></head>")
    html = html.replace('<div id="swagger-ui">', cabecera + '<div id="swagger-ui">')
    return HTMLResponse(html)


def tarjeta_modulo(icono, titulo, red, metodo, ruta, texto, dato, listo):
    estado = "En línea" if listo else "No disponible"
    return f"""
<div class="tarjeta">
  <div class="tarjeta-top"><span class="icono">{icono}</span>
    <span class="estado {'ok' if listo else 'no'}">{estado}</span></div>
  <h3>{titulo}</h3>
  <div class="red">{red}</div>
  <p>{texto}</p>
  <div class="dato">{dato}</div>
  <a class="ruta" href="/docs"><span class="metodo {metodo.lower()}">{metodo}</span>{ruta}</a>
</div>"""


def pagina_inicio():
    """Página de inicio de la API: estado, módulos y gráficos en vivo."""
    # Datos de los modelos tomados de los archivos, no escritos a mano
    dato_cnn = "Modelo: mobilenet_mosquitos.keras"
    try:
        with open(os.path.join(MODELS_DIR, "mosquitos_resultados.json"), encoding="utf-8") as archivo:
            f1 = json.load(archivo)["mobilenetv2"]["f1_macro"]
        dato_cnn = f"F1 macro en test: <b>{100 * f1:.1f} %</b>"
    except Exception:
        pass

    dato_rnn = f"{rnn_model.count_params():,} pesos · 26 → 4 semanas" if rnn_ready else "Modelo no cargado"
    dato_ann = f"{ann_model.count_params():,} pesos · 4 acciones" if ann_ready else "Modelo no cargado"
    dato_prov = "Modelo no cargado"
    if prov_ready and prov_resultados:
        g = prov_resultados["global"]
        dato_prov = (f"RMSE en test: <b>{g['SimpleRNN']['RMSE']:,.1f}</b> contra "
                     f"{g['Baseline estacional']['RMSE']:,.1f} del baseline estacional")

    tarjetas = "".join([
        tarjeta_modulo("🦟", "Identificar el mosquito", "CNN · MobileNetV2", "POST", "/predict/mosquito",
                       "Sube una foto y obtén la especie: Aedes aegypti, Aedes albopictus o Culex.", dato_cnn, cnn_ready),
        tarjeta_modulo("📈", "Pronosticar casos", "SimpleRNN · datos reales", "POST", "/predict/casos",
                       "Con 26 semanas de casos y clima, pronostica las 4 siguientes.", dato_rnn, rnn_ready),
        tarjeta_modulo("🛡️", "Recomendar una acción", "ANN · clasificador", "POST", "/predict/accion",
                       "Combina pronóstico, umbral, clima y especie, y sugiere la acción con su motivo.", dato_ann, ann_ready),
        tarjeta_modulo("🗺️", "Bonus: por provincia", "SimpleRNN · datos sintéticos", "POST", "/predict/casos/provincia",
                       "Una red para las 7 provincias. Datos sintéticos del curso, no cifras oficiales.", dato_prov, prov_ready),
    ])

    rutas = [
        ("POST", "/predict/mosquito", "Especie de mosquito de una foto"),
        ("POST", "/predict/casos", "Pronóstico nacional de 4 semanas"),
        ("POST", "/predict/accion", "Acción preventiva recomendada"),
        ("POST", "/predict/casos/provincia", "Pronóstico de una provincia (bonus)"),
        ("GET", "/visual/nacional?fecha_corte=2023-10-15", "Gráfico PNG del pronóstico nacional"),
        ("GET", "/visual/provincia/Limón", "Gráfico PNG de una provincia"),
        ("GET", "/visual/provincias", "Panel PNG de las 7 provincias"),
        ("GET", "/estado", "Estado de los modelos (JSON)"),
    ]
    filas = "".join(
        f'<a class="fila" href="{ruta if metodo == "GET" else "/docs"}"><span class="metodo {metodo.lower()}">{metodo}</span>'
        f'<code>{ruta}</code><span class="desc">{desc}</span></a>'
        for metodo, ruta, desc in rutas
    )

    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VectorIA API</title><link rel="icon" href="{ICONO}">
<style>
:root {{ --navy:#0F2236; --navy2:#183B56; --teal:#20A39E; --neon:#2EE6C5; --texto:#243B53; --suave:#5B7083;
  --borde:#E2EAF0; --fondo:#F3F7FA; --rojo:#DC6262; --azul:#247BA0; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family:"Segoe UI",system-ui,sans-serif; background:var(--fondo); color:var(--texto); }}
.heroe {{ background: radial-gradient(1200px 500px at 85% -10%, rgba(46,230,197,.28), transparent 60%),
  linear-gradient(125deg, var(--navy) 0%, var(--navy2) 60%, #1B6F7A 100%); color:#fff; padding:56px 20px 120px; }}
.interior {{ max-width:1180px; margin:0 auto; }}
.sello {{ display:inline-flex; gap:8px; align-items:center; background:rgba(255,255,255,.1);
  border:1px solid rgba(255,255,255,.2); padding:6px 14px; border-radius:999px; font-size:13px; color:#CFE3EE; }}
.sello i {{ width:8px; height:8px; border-radius:50%; background:var(--neon); box-shadow:0 0 12px var(--neon);
  animation: latido 1.6s infinite; }}
@keyframes latido {{ 50% {{ opacity:.35; }} }}
h1 {{ font-size:clamp(38px,6vw,64px); margin:18px 0 6px; letter-spacing:-1.5px; }}
h1 span {{ color:var(--neon); }}
.lema {{ font-size:19px; color:#C4D6E2; max-width:720px; line-height:1.5; }}
.botones {{ display:flex; gap:12px; flex-wrap:wrap; margin-top:26px; }}
.boton {{ padding:12px 22px; border-radius:12px; font-weight:700; text-decoration:none; font-size:15px; }}
.boton.principal {{ background:var(--neon); color:var(--navy); box-shadow:0 8px 24px rgba(46,230,197,.35); }}
.boton.secundario {{ border:1px solid rgba(255,255,255,.35); color:#fff; }}
.tarjetas {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:18px; margin-top:-84px; }}
.tarjeta {{ background:#fff; border-radius:18px; padding:20px; box-shadow:0 14px 40px rgba(16,42,67,.12);
  border:1px solid var(--borde); display:flex; flex-direction:column; }}
.tarjeta-top {{ display:flex; justify-content:space-between; align-items:center; }}
.icono {{ font-size:28px; width:50px; height:50px; display:grid; place-items:center; border-radius:14px; background:#EAF7F6; }}
.estado {{ font-size:12px; font-weight:700; padding:4px 10px; border-radius:999px; }}
.estado.ok {{ background:#E3F7EE; color:#1E8E5A; }} .estado.no {{ background:#FDECEC; color:var(--rojo); }}
.tarjeta h3 {{ margin:14px 0 2px; font-size:18px; color:#102A43; }}
.red {{ font-size:12.5px; color:var(--teal); font-weight:700; text-transform:uppercase; letter-spacing:.4px; }}
.tarjeta p {{ font-size:14px; color:var(--suave); line-height:1.5; flex:1; }}
.dato {{ font-size:13px; background:var(--fondo); border-radius:10px; padding:8px 10px; margin-bottom:12px; }}
.ruta {{ display:flex; align-items:center; gap:8px; font-family:Consolas,monospace; font-size:13px; color:#102A43;
  text-decoration:none; font-weight:700; }}
.metodo {{ font-family:"Segoe UI",sans-serif; font-size:11px; font-weight:800; color:#fff; padding:4px 8px; border-radius:6px; }}
.metodo.post {{ background:var(--teal); }} .metodo.get {{ background:var(--azul); }}
section {{ margin-top:46px; }}
h2 {{ font-size:26px; color:#102A43; margin:0 0 4px; }}
.nota {{ color:var(--suave); margin:0 0 16px; }}
.grafico {{ background:var(--navy); border-radius:20px; padding:10px; box-shadow:0 18px 50px rgba(16,42,67,.25); }}
.grafico img {{ width:100%; display:block; border-radius:14px; min-height:200px; }}
.lista {{ background:#fff; border:1px solid var(--borde); border-radius:16px; overflow:hidden; }}
.fila {{ display:grid; grid-template-columns:62px minmax(0,1.3fr) minmax(0,1fr); gap:12px; align-items:center;
  padding:13px 18px; border-bottom:1px solid var(--borde); text-decoration:none; color:inherit; }}
.fila:last-child {{ border-bottom:0; }} .fila:hover {{ background:#F7FBFC; }}
.fila code {{ font-size:14px; color:#102A43; overflow-wrap:anywhere; }} .fila .desc {{ color:var(--suave); font-size:14px; }}
.aviso {{ margin:46px 0 30px; background:#FFF7E8; border:1px solid #F5D9A8; border-radius:14px; padding:14px 18px; color:#7A5418; }}
footer {{ text-align:center; color:var(--suave); font-size:13px; padding:0 0 40px; }}
@media (max-width:640px) {{ .fila {{ grid-template-columns:56px 1fr; }} .fila .desc {{ grid-column:2; }} }}
</style></head>
<body>
<div class="heroe"><div class="interior">
  <span class="sello"><i></i>API en línea · versión {app.version}</span>
  <h1>Vector<span>IA</span> API</h1>
  <div class="lema">Tres redes neuronales para la vigilancia del dengue en Costa Rica: identifican el mosquito,
  pronostican los casos y recomiendan qué hacer.</div>
  <div class="botones">
    <a class="boton principal" href="/docs">Probar la API en /docs →</a>
    <a class="boton secundario" href="/visual/provincias">Ver el panel por provincia</a>
  </div>
</div></div>

<div class="interior" style="padding:0 20px">
  <div class="tarjetas">{tarjetas}</div>

  <section>
    <h2>📈 Pronóstico nacional en vivo</h2>
    <p class="nota">Generado ahora mismo por <code>GET /visual/nacional</code> con la SimpleRNN y los datos reales:
    brote de 2023, con los casos reales para comparar.</p>
    <div class="grafico"><img src="/visual/nacional?fecha_corte=2023-10-15" alt="Pronóstico nacional"></div>
  </section>

  <section>
    <h2>🗺️ Las 7 provincias (bonus)</h2>
    <p class="nota">Generado por <code>GET /visual/provincias</code>. Datos sintéticos del paquete del curso,
    no son cifras oficiales.</p>
    <div class="grafico"><img src="/visual/provincias?fecha_corte=2024-07-14" alt="Panel por provincia" loading="lazy"></div>
  </section>

  <section>
    <h2>🔌 Endpoints</h2>
    <p class="nota">Los GET se abren directo en el navegador; los POST se prueban en /docs con ejemplos listos.</p>
    <div class="lista">{filas}</div>
  </section>

  <div class="aviso">⚠️ <b>Herramienta de apoyo a la vigilancia.</b> No diagnostica dengue en personas.</div>
</div>
<footer>VectorIA · Inteligencia Artificial Aplicada · CUC · FastAPI {app.version}</footer>
</body></html>"""
