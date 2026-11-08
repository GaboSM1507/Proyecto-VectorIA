"""Preparación de datos del proyecto Vector-IA.

Reúne las funciones de preparación de los modelos:
- Módulo 2 (RNN/LSTM): serie semanal de casos y clima.
- Módulo 3 (ANN): tabla de acciones recomendadas.

Las rutas son relativas a la carpeta raíz del proyecto.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

# Datos que recibe la ANN (el mismo orden se usa al entrenar y al predecir)
COLUMNAS_ANN = ["casos_pred_4sem", "casos_umbral_endemico", "lluvia_14d_mm",
                "temp_max", "temp_min", "humedad", "aegypti_detectado"]


# ----------------------------------------------------------------------------
# Módulo 2: serie de casos y clima (RNN / LSTM)
# ----------------------------------------------------------------------------
def cargar_serie(ruta="data/processed/dengue_procesado.csv"):
    """Lee la serie semanal que dejó el EDA y agrega la semana del año como seno y coseno."""
    df = pd.read_csv(ruta, parse_dates=["fecha_inicio_semana"], index_col="fecha_inicio_semana")
    df["semana_sin"] = np.sin(2 * np.pi * df["semana_epi"] / 52)   # así la semana 52 queda cerca de la 1
    df["semana_cos"] = np.cos(2 * np.pi * df["semana_epi"] / 52)
    return df


def dividir_temporal(n_semanas, semanas_test=104, semanas_validacion=52):
    """Separa por orden de fecha: entrenamiento, validación y prueba (las más recientes).

    Devuelve la fila donde empieza la validación y la fila donde empieza la prueba.
    """
    inicio_test = n_semanas - semanas_test
    inicio_validacion = inicio_test - semanas_validacion
    return inicio_validacion, inicio_test


def crear_objetivos(valores, horizonte=4):
    """En la fila i guarda los valores de las 4 semanas que empiezan en i (las últimas filas quedan vacías)."""
    objetivos = np.full((len(valores), horizonte), np.nan)
    for i in range(len(valores) - horizonte + 1):
        objetivos[i] = valores[i:i + horizonte]
    return objetivos


# ----------------------------------------------------------------------------
# Módulo 3: acciones recomendadas (ANN)
# ----------------------------------------------------------------------------
def cargar_acciones(ruta="data/raw/vectoria_acciones.csv"):
    """Lee el archivo de acciones recomendadas (2,060 filas: 412 semanas x 5 escenarios)."""
    return pd.read_csv(ruta, parse_dates=["fecha_inicio_semana"])


def preparar_acciones(acc):
    """Devuelve las entradas (X), las respuestas como números (y) y el codificador nombre <-> número."""
    codificador = LabelEncoder()
    y = codificador.fit_transform(acc["recomendacion"])
    return acc[COLUMNAS_ANN], y, codificador


def dividir_por_semana(acc, y, porcentaje=0.2, semilla=42):
    """Separa entrenamiento y prueba por semana: las 5 copias de una semana quedan del mismo lado.

    Devuelve las posiciones de las filas de entrenamiento y de prueba.
    """
    division = GroupShuffleSplit(n_splits=1, test_size=porcentaje, random_state=semilla)
    return next(division.split(acc, y, groups=acc["fecha_inicio_semana"]))