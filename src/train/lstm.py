"""Arquitecturas del Módulo 2: SimpleRNN (base) y LSTM/GRU (modelo principal).

Entrada: 26 semanas con varias variables (casos en logaritmo, clima, semana del año).
Salida: los casos de las próximas 4 semanas.
Aquí solo están las redes. El entrenamiento y las métricas están en 03_LSTM_Casos.ipynb.
"""
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input, SimpleRNN
from tensorflow.keras.models import Sequential

VENTANA = 26     # semanas de entrada
HORIZONTE = 4    # semanas a pronosticar


def crear_simplernn(n_variables):
    """Modelo base: una SimpleRNN de 32 unidades (el mismo del notebook)."""
    modelo = Sequential([
        Input(shape=(VENTANA, n_variables)),
        SimpleRNN(32),
        Dense(HORIZONTE),    # una salida por cada semana a pronosticar
    ])
    modelo.compile(optimizer="adam", loss="mse")
    return modelo


def crear_lstm(n_variables, tipo="LSTM"):
    """Modelo principal del enunciado: dos capas (64 y 32 unidades), Dropout 0.2 y 4 salidas.

    tipo puede ser "LSTM" o "GRU".
    """
    capa = LSTM if tipo == "LSTM" else GRU
    modelo = Sequential([
        Input(shape=(VENTANA, n_variables)),
        capa(64, return_sequences=True),    # pasa toda la secuencia a la segunda capa
        capa(32),
        Dropout(0.2),
        Dense(HORIZONTE),
    ])
    modelo.compile(optimizer="adam", loss="mse")
    return modelo