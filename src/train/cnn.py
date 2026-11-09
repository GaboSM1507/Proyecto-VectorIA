"""Arquitecturas de la CNN (Módulo 1): identifica la especie del mosquito en una foto.

Clases (3): Aedes aegypti, Aedes albopictus y Culex.
Aquí solo están las redes. Todo lo demás está en notebooks/02_CNN_Mosquitos.ipynb:
el análisis de las imágenes y la separación entre entrenamiento y prueba (secciones 1 a 3),
el entrenamiento y la evaluación (sección 4) y la CNN propia con las fotos recortadas
alrededor del mosquito, que usa crear_cnn_v2 con recortes de 128 x 128 (sección 5).
src/train/entrenar_mosquitos.py es la versión anterior del entrenamiento.

Las respuestas deben ser números (0, 1, 2), no vectores de unos y ceros.
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    RandomContrast,
    RandomFlip,
    RandomRotation,
    RandomZoom
)
from tensorflow.keras.models import Sequential

TAMANO = 128    # las fotos se redimensionan a 128 x 128 píxeles


def crear_cnn(n_clases=3):
    """CNN propia desde cero, con la estructura que sugiere el enunciado.

    Las fotos deben llegar divididas entre 255, para que sus valores queden entre 0 y 1.
    """
    modelo = Sequential([
        Input(shape=(TAMANO, TAMANO, 3)),
        RandomFlip("horizontal"),
        RandomRotation(0.05),
        RandomZoom(0.1),
        Conv2D(32, 3, activation="relu"), MaxPooling2D(),     # busca bordes y manchas simples
        Conv2D(64, 3, activation="relu"), MaxPooling2D(),     # combina lo anterior en formas
        Conv2D(128, 3, activation="relu"), MaxPooling2D(),    # reconoce partes del mosquito
        GlobalAveragePooling2D(),                             # resume cada mapa en un número
        Dense(128, activation="relu"),
        Dropout(0.5),                                         # apaga la mitad al azar para no memorizar
        Dense(n_clases, activation="softmax"),                # una probabilidad por especie
    ])
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


def crear_cnn_v2(n_clases=3, tamano=192):
    """CNN propia mejorada, también desde cero (sin pesos preentrenados).

    Cambios frente a la sugerida por el enunciado:
    - fotos más grandes (192 en vez de 128), para ver detalles como las escamas
      blancas del tórax, que distinguen a aegypti de albopictus;
    - dos capas Conv2D por bloque y un cuarto bloque de 256 filtros;
    - BatchNormalization, que estabiliza el entrenamiento;
    - más aumento de datos (giros en las dos direcciones, rotación, zoom y contraste).

    Las fotos deben llegar divididas entre 255, igual que en crear_cnn.
    """
    modelo = Sequential([
        Input(shape=(tamano, tamano, 3)),
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        RandomZoom(0.2),
        RandomContrast(0.2),
        Conv2D(32, 3, padding="same", activation="relu"), BatchNormalization(),
        Conv2D(32, 3, padding="same", activation="relu"), BatchNormalization(), MaxPooling2D(),
        Conv2D(64, 3, padding="same", activation="relu"), BatchNormalization(),
        Conv2D(64, 3, padding="same", activation="relu"), BatchNormalization(), MaxPooling2D(),
        Conv2D(128, 3, padding="same", activation="relu"), BatchNormalization(),
        Conv2D(128, 3, padding="same", activation="relu"), BatchNormalization(), MaxPooling2D(),
        Conv2D(256, 3, padding="same", activation="relu"), BatchNormalization(), MaxPooling2D(),
        GlobalAveragePooling2D(),
        Dropout(0.4),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(n_clases, activation="softmax"),                # una probabilidad por especie
    ])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


def crear_mobilenet(n_clases=3):
    """MobileNetV2 ya entrenada con ImageNet (transfer learning); solo se entrena la parte final.

    Las fotos deben pasar antes por tensorflow.keras.applications.mobilenet_v2.preprocess_input.
    La primera vez descarga los pesos de internet.
    """
    base = MobileNetV2(input_shape=(TAMANO, TAMANO, 3), include_top=False, weights="imagenet")
    base.trainable = False    # congela lo que ya aprendió con millones de fotos
    modelo = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(n_clases, activation="softmax"),
    ])
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo