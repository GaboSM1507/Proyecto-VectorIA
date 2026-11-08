"""Entrenamiento de la ANN (Módulo 3): recomienda una de 4 acciones preventivas.

Uso, desde la carpeta raíz del proyecto:
    python -m src.train.ann

CUIDADO: vuelve a entrenar y reemplaza ann_acciones.keras y ann_acciones_preprocesado.pkl en models/.
"""
import os

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential

from src.data_prep import COLUMNAS_ANN, cargar_acciones, dividir_por_semana, preparar_acciones


def crear_ann(n_entradas, n_acciones=4):
    """Red pequeña (852 parámetros): capas de 32 y 16 neuronas, con Dropout para no memorizar."""
    modelo = Sequential([
        Input(shape=(n_entradas,)),
        Dense(32, activation="relu"), Dropout(0.2),
        Dense(16, activation="relu"), Dropout(0.2),
        Dense(n_acciones, activation="softmax"),    # una probabilidad por acción
    ])
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


def main(semilla=42):
    tf.keras.utils.set_random_seed(semilla)    # para que el resultado se pueda repetir

    # 1. Datos y separación por semana (20 % para la prueba final)
    acc = cargar_acciones()
    X, y, codificador = preparar_acciones(acc)
    i_train, i_test = dividir_por_semana(acc, y, semilla=semilla)
    X_train, X_test = X.iloc[i_train], X.iloc[i_test]
    y_train, y_test = y[i_train], y[i_test]

    # 2. Poner los números en la misma escala (el escalador aprende solo con el entrenamiento)
    escalador = StandardScaler()
    X_train_esc = escalador.fit_transform(X_train)
    X_test_esc = escalador.transform(X_test)

    # 3. Apartar el 15 % de las semanas de entrenamiento para validar
    semanas = acc["fecha_inicio_semana"].iloc[i_train]
    j_fit, j_val = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=semilla)
                        .split(X_train_esc, y_train, groups=semanas))

    # 4. Pesos por clase: las acciones raras valen más cuando la red se equivoca con ellas
    pesos = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    pesos_clases = {i: round(float(p), 2) for i, p in enumerate(pesos)}

    # 5. Entrenar (se detiene solo cuando la validación deja de mejorar)
    modelo = crear_ann(X_train_esc.shape[1], len(codificador.classes_))
    parada = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    modelo.fit(X_train_esc[j_fit], y_train[j_fit],
               validation_data=(X_train_esc[j_val], y_train[j_val]),
               epochs=200, batch_size=32, class_weight=pesos_clases, callbacks=[parada], verbose=0)

    # 6. Examen final con los datos de prueba
    y_pred = np.argmax(modelo.predict(X_test_esc, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=codificador.classes_, digits=2))

    # 7. Guardar la red y todo lo necesario para usarla después (app y API)
    os.makedirs("models", exist_ok=True)
    modelo.save("models/ann_acciones.keras")
    joblib.dump({"columnas": COLUMNAS_ANN, "escalador": escalador,
                 "codificador": codificador, "pesos_clases": pesos_clases},
                "models/ann_acciones_preprocesado.pkl")
    print("Modelo guardado en models/")


if __name__ == "__main__":
    main()