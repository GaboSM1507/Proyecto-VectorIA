"""Arquitecturas de la CNN (Módulo 1): identifica la especie del mosquito en una foto.

Clases (3): Aedes aegypti, Aedes albopictus y Culex.
Aquí solo están las redes. El entrenamiento, el F1 macro y el guardado de los modelos
van en el notebook 02_CNN_Mosquitos.ipynb (pendiente).

Las respuestas deben ser números (0, 1, 2), no vectores de unos y ceros.
"""
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, MaxPooling2D
from tensorflow.keras.models import Sequential

TAMANO = 128    # las fotos se redimensionan a 128 x 128 píxeles


def crear_cnn(n_clases=3):
    """CNN propia desde cero, con la estructura que sugiere el enunciado.

    Las fotos deben llegar divididas entre 255, para que sus valores queden entre 0 y 1.
    """
    modelo = Sequential([
        Input(shape=(TAMANO, TAMANO, 3)),
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