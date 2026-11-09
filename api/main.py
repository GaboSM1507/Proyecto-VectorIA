
import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

app = FastAPI(
    title="VectorIA API",
    description="API inteligente para vigilancia y prevención del dengue en Costa Rica.",
    version="2.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


# ============================================================
# 2. RUTAS DE LOS MODELOS
# ============================================================

cnn_model_path = os.path.join(MODELS_DIR, "mobilenet_mosquitos.keras")
cnn_clases_path = os.path.join(MODELS_DIR, "mosquitos_clases.json")

ann_model_path = os.path.join(MODELS_DIR, "ann_acciones.keras")
ann_prep_path = os.path.join(MODELS_DIR, "ann_acciones_preprocesado.pkl")

rnn_model_path = os.path.join(MODELS_DIR, "rnn_simple.h5")
rnn_prep_path = os.path.join(MODELS_DIR, "rnn_simple_preprocesado.pkl")


# ============================================================
# 3. CARGA DE MODELOS
# ============================================================

cnn_ready = False
ann_ready = False
rnn_ready = False

cnn_model = None
cnn_clases = None

ann_model = None
ann_prep = None

rnn_model = None
rnn_prep = None


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


# ============================================================
# 4. ESQUEMAS DE ENTRADA
# ============================================================

class AccionInput(BaseModel):

    casos_pred_4sem: float = Field(..., ge=0)
    casos_umbral_endemico: float = Field(..., gt=0)
    lluvia_14d_mm: float = Field(..., ge=0)
    temp_max: float
    temp_min: float
    humedad: float = Field(..., ge=0, le=100)
    aegypti_detectado: int = Field(..., ge=0, le=1)


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


# ============================================================
# 5. ESTADO DE LA API
# ============================================================

@app.get("/", summary="Estado de los servicios")

def health_check():

    return {
        "status": "online",
        "cnn_ready": cnn_ready,
        "ann_ready": ann_ready,
        "rnn_ready": rnn_ready,
        "models_directory": MODELS_DIR
    }


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


@app.post("/predict/accion", summary="Recomendar acción preventiva")

def predict_accion(data: AccionInput):

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

@app.post("/predict/rnn", summary="Pronóstico de dengue con SimpleRNN")

def predict_rnn(data: RNNForecastInput):

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

@app.post("/predict/mosquito", summary="Identificar especie de mosquito")

async def predict_mosquito(file: UploadFile = File(...)):

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
