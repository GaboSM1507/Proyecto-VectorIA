import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# 1. Configuración de la Aplicación FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Vector-IA API Backend",
    description="Servicio REST para la recomendación de acciones preventivas (ANN) y el pronóstico de casos (RNN).",
    version="1.1.0"
)

# -----------------------------------------------------------------------------
# 2. Localización y Carga de Artefactos (.keras, .h5 y .pkl)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Rutas ANN (modelo de 4 acciones del Módulo 3)
ann_model_path = os.path.join(MODELS_DIR, 'ann_acciones.keras')
ann_prep_path = os.path.join(MODELS_DIR, 'ann_acciones_preprocesado.pkl')

# Rutas RNN
rnn_model_path = os.path.join(MODELS_DIR, 'rnn_simple.h5')
rnn_prep_path = os.path.join(MODELS_DIR, 'rnn_simple_preprocesado.pkl')

# Carga de artefactos ANN
try:
    ann_model = tf.keras.models.load_model(ann_model_path)  # la red entrenada
    ann_prep = joblib.load(ann_prep_path)  # diccionario con: columnas, escalador y codificador
    ann_ready = True
    print("✓ Artefactos ANN cargados correctamente.")
except Exception as e:
    ann_ready = False
    print(f"✗ Error al cargar artefactos ANN: {e}")

# Carga de artefactos RNN
try:
    rnn_model = tf.keras.models.load_model(rnn_model_path)
    rnn_prep = joblib.load(rnn_prep_path)
    rnn_ready = True
    print("✓ Artefactos RNN cargados correctamente.")
except Exception as e:
    rnn_ready = False
    print(f"⚠ Aviso: Artefactos RNN no encontrados o con formato imprevisto ({e}). Modo simulación fallback activo.")


# -----------------------------------------------------------------------------
# 3. Esquemas de Validación (Pydantic)
# -----------------------------------------------------------------------------
class AccionInput(BaseModel):
    casos_pred_4sem: float = Field(..., ge=0, description="Casos predichos para las próximas 4 semanas")
    casos_umbral_endemico: float = Field(..., gt=0, description="Umbral endémico de esas semanas")
    lluvia_14d_mm: float = Field(..., ge=0, description="Lluvia de los últimos 14 días (mm)")
    temp_max: float = Field(..., description="Temperatura máxima (°C)")
    temp_min: float = Field(..., description="Temperatura mínima (°C)")
    humedad: float = Field(..., ge=0, le=100, description="Humedad relativa (%)")
    aegypti_detectado: int = Field(..., ge=0, le=1, description="1 si la CNN detectó Aedes aegypti, 0 si no")

    class Config:
        json_schema_extra = {
            "example": {
                "casos_pred_4sem": 339,
                "casos_umbral_endemico": 282,
                "lluvia_14d_mm": 17.4,
                "temp_max": 27.1,
                "temp_min": 21.0,
                "humedad": 85,
                "aegypti_detectado": 1
            }
        }


class RNNForecastInput(BaseModel):
    casos_ultimas_4_semanas: list[float] = Field(
        ...,
        min_items=4,
        max_items=4,
        description="Registros semanales de casos observados en el último mes"
    )
    semanas_a_proyectar: int = Field(4, ge=1, le=8, description="Número de semanas a futuro para el pronóstico")

    class Config:
        json_schema_extra = {
            "example": {
                "casos_ultimas_4_semanas": [12.0, 18.0, 25.0, 31.0],
                "semanas_a_proyectar": 4
            }
        }


# -----------------------------------------------------------------------------
# 4. Endpoints de la API
# -----------------------------------------------------------------------------
@app.get("/", summary="Estado de Salud del Servicio")
def health_check():
    return {
        "status": "online",
        "ann_ready": ann_ready,
        "rnn_ready": rnn_ready,
        "models_directory": MODELS_DIR
    }


# Texto que explica cada acción (usa los datos que mandó el usuario)
MOTIVOS = {
    "alerta_sanitaria": "Los casos predichos ({casos:.0f}) superan el doble del umbral endémico ({umbral:.0f}).",
    "fumigacion_focalizada": "Los casos predichos ({casos:.0f}) superan el umbral endémico ({umbral:.0f}) y se detectó Aedes aegypti.",
    "eliminacion_criaderos": "Llovió {lluvia:.0f} mm en 14 días y la temperatura máxima fue {tmax:.1f} °C, condiciones que favorecen los criaderos.",
    "monitoreo_normal": "No se cumplen condiciones de alerta, fumigación ni criaderos. Se recomienda vigilancia de rutina.",
}


@app.post("/predict/accion", summary="Recomendar acción preventiva (ANN)")
def predict_accion(data: AccionInput):
    """Recibe casos predichos, clima y detección de aegypti; devuelve la acción recomendada y su motivo."""
    if not ann_ready:
        raise HTTPException(status_code=500, detail="Artefactos ANN no disponibles.")

    fila = pd.DataFrame([data.model_dump()])[ann_prep["columnas"]]  # ordena las columnas como al entrenar
    entrada = ann_prep["escalador"].transform(fila)  # misma escala del entrenamiento
    probabilidades = ann_model.predict(entrada, verbose=0)[0]  # una probabilidad por acción
    indice = int(np.argmax(probabilidades))  # la acción más probable
    accion = str(ann_prep["codificador"].classes_[indice])  # número -> nombre de la acción

    return {
        "accion": accion,
        "confianza_pct": round(float(probabilidades[indice]) * 100, 1),
        "probabilidades": {str(c): round(float(p) * 100, 1)
                           for c, p in zip(ann_prep["codificador"].classes_, probabilidades)},
        "motivo": MOTIVOS[accion].format(casos=data.casos_pred_4sem, umbral=data.casos_umbral_endemico,
                                         lluvia=data.lluvia_14d_mm, tmax=data.temp_max),
    }


@app.post("/predict/rnn", summary="Pronóstico Temporal de Casos (RNN)")
def predict_rnn(data: RNNForecastInput):
    """
    Recibe la secuencia de casos observados en las últimas 4 semanas
    y proyecta la curva de contagios para las próximas semanas.
    """
    historico = np.array(data.casos_ultimas_4_semanas, dtype=float)
    horizonte = data.semanas_a_proyectar

    if rnn_ready:
        try:
            # Preprocesar / Escalar secuencia según el artefacto cargado
            if hasattr(rnn_prep, 'transform'):
                seq_scaled = rnn_prep.transform(historico.reshape(-1, 1)).reshape(1, 4, 1)
            else:
                seq_scaled = historico.reshape(1, 4, 1)

            # Inferencia iterativa o directa
            predicciones_escaladas = []
            curr_seq = seq_scaled.copy()

            for _ in range(horizonte):
                p = rnn_model.predict(curr_seq, verbose=0)
                predicciones_escaladas.append(p[0, 0])
                curr_seq = np.roll(curr_seq, -1, axis=1)
                curr_seq[0, -1, 0] = p[0, 0]

            # Invertir escalado si el preprocesador lo soporta
            if hasattr(rnn_prep, 'inverse_transform'):
                predicciones = rnn_prep.inverse_transform(np.array(predicciones_escaladas).reshape(-1, 1)).flatten()
            else:
                predicciones = np.array(predicciones_escaladas)

        except Exception as err:
            print(f"Error durante inferencia RNN: {err}. Generando proyección ajustada.")
            # Fallback analítico basado en tendencia autorregresiva
            tasa_crecimiento = historico[-1] / (historico[-2] + 1e-5)
            predicciones = [historico[-1] * (tasa_crecimiento ** i) for i in range(1, horizonte + 1)]
    else:
        # Fallback si no existen artefactos RNN
        tasa = historico[-1] / (historico[0] + 1e-5)
        predicciones = [max(0.0, historico[-1] + (i * 2.5 * (1 if tasa >= 1 else -1))) for i in range(1, horizonte + 1)]

    # Formatear respuesta JSON para graficado directo
    predicciones_limpias = [round(float(x), 1) for x in predicciones]
    casos_totales_proyectados = round(sum(predicciones_limpias), 1)

    tendencia = "Ascendente 📈" if predicciones_limpias[-1] > historico[-1] else "Descendente / Estable 📉"

    return {
        "historico_ingresado": historico.tolist(),
        "semanas_proyectadas": [f"Semana +{i + 1}" for i in range(horizonte)],
        "pronostico_casos": predicciones_limpias,
        "total_estimado_periodo": casos_totales_proyectados,
        "tendencia_general": tendencia
    }