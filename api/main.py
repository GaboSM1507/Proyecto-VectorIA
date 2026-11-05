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
    description="Servicio REST para la evaluación de riesgo epidemiológico (ANN) y pronóstico de casos (RNN).",
    version="1.1.0"
)

# -----------------------------------------------------------------------------
# 2. Localización y Carga de Artefactos (.h5 y .pkl)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Rutas ANN
ann_model_path = os.path.join(MODELS_DIR, 'ann_control.h5')
ann_scaler_path = os.path.join(MODELS_DIR, 'scaler_ann.pkl')
ann_cols_path = os.path.join(MODELS_DIR, 'ann_columns.pkl')

# Rutas RNN
rnn_model_path = os.path.join(MODELS_DIR, 'rnn_simple.h5')
rnn_prep_path = os.path.join(MODELS_DIR, 'rnn_simple_preprocesado.pkl')

# Carga de artefactos ANN
try:
    ann_model = tf.keras.models.load_model(ann_model_path)
    ann_scaler = joblib.load(ann_scaler_path)
    ann_columns = joblib.load(ann_cols_path)
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
class DenguePredictionInput(BaseModel):
    semana_epi: int = Field(..., ge=1, le=53)
    anio: int = Field(2024, ge=2000, le=2030)
    mes: int = Field(..., ge=1, le=12)
    lluvia_mm: float = Field(..., ge=0.0)
    temp_max_c: float = Field(...)
    temp_min_c: float = Field(...)
    humedad_rel_pct: float = Field(..., ge=0.0, le=100.0)
    lluvia_mm_lag2: float = Field(0.0)
    temp_max_c_lag2: float = Field(0.0)


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


@app.post("/predict/ann", summary="Evaluar Nivel de Riesgo (ANN)")
def predict_risk(data: DenguePredictionInput):
    if not ann_ready:
        raise HTTPException(status_code=500, detail="Artefactos ANN no disponibles.")

    raw_data = data.model_dump()

    # Imputación de Lags
    raw_data.setdefault('lluvia_mm_lag4', raw_data.get('lluvia_mm_lag2', raw_data['lluvia_mm']))
    raw_data.setdefault('lluvia_mm_lag6', raw_data['lluvia_mm'])
    raw_data.setdefault('lluvia_mm_lag8', raw_data['lluvia_mm'])

    raw_data.setdefault('temp_max_c_lag4', raw_data.get('temp_max_c_lag2', raw_data['temp_max_c']))
    raw_data.setdefault('temp_max_c_lag6', raw_data['temp_max_c'])
    raw_data.setdefault('temp_max_c_lag8', raw_data['temp_max_c'])

    raw_data.setdefault('temp_min_c_lag2', raw_data['temp_min_c'])
    raw_data.setdefault('temp_min_c_lag4', raw_data['temp_min_c'])
    raw_data.setdefault('temp_min_c_lag6', raw_data['temp_min_c'])
    raw_data.setdefault('temp_min_c_lag8', raw_data['temp_min_c'])

    raw_data.setdefault('humedad_rel_pct_lag2', raw_data['humedad_rel_pct'])
    raw_data.setdefault('humedad_rel_pct_lag4', raw_data['humedad_rel_pct'])
    raw_data.setdefault('humedad_rel_pct_lag6', raw_data['humedad_rel_pct'])
    raw_data.setdefault('humedad_rel_pct_lag8', raw_data['humedad_rel_pct'])

    input_df = pd.DataFrame(0.0, index=[0], columns=ann_columns)
    for col_name, value in raw_data.items():
        if col_name in input_df.columns:
            input_df[col_name] = value

    scaled_input = ann_scaler.transform(input_df)
    probabilities = ann_model.predict(scaled_input)
    predicted_class = int(np.argmax(probabilities, axis=1)[0])
    confidence = float(probabilities[0][predicted_class]) * 100

    labels = {
        0: {"nivel": "Riesgo Bajo", "color": "green", "accion": "Condiciones bajo control. Vigilancia de rutina."},
        1: {"nivel": "Riesgo Medio", "color": "orange",
            "accion": "Incremento de vectores. Aplicar control preventivo."},
        2: {"nivel": "Riesgo Alto", "color": "red", "accion": "Alerta epidémica. Activar protocolos de fumigación."}
    }

    result = labels[predicted_class]

    return {
        "clase_predicha": predicted_class,
        "nivel_riesgo": result["nivel"],
        "color": result["color"],
        "confianza_pct": round(confidence, 2),
        "accion_recomendada": result["accion"]
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