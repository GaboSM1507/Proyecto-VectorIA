"""Versión anterior del entrenamiento de las CNN (Módulo 1).

La versión actual del entrenamiento y la evaluación, con la CNN propia mejorada (v2),
la elección con validación y los resultados calculados, está en
notebooks/02_CNN_Mosquitos.ipynb, secciones 4 y 5 (en la 5, con las fotos recortadas alrededor
del mosquito). Este script se conserva como referencia:
sus líneas de entrenamiento están comentadas y los F1 que guarda están escritos a mano.
"""
import pandas as pd
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from src.train.cnn import crear_cnn, crear_mobilenet

# Cargar el inventario de imágenes #

BASE_DIR = Path(__file__).resolve().parents[2]

RUTA_INVENTARIO = BASE_DIR / "data" / "processed" / "inventario_imagenes.csv"

CLASES = ["aegypti", "albopictus", "culex"]

df = pd.read_csv(RUTA_INVENTARIO)

print("Total de imágenes:", len(df))
print("\nImágenes por especie:")
print(df["clase"].value_counts())

print("\nDistribución entrenamiento/prueba:")
print(pd.crosstab(df["clase"], df["split"]))

# Preparar las rutas de las imágenes #

CARPETA_NOTEBOOKS = BASE_DIR / "notebooks"

df["ruta_completa"] = df["ruta"].apply(
    lambda ruta: str((CARPETA_NOTEBOOKS / ruta).resolve())
)

# Asignar un número a cada especie #
clase_a_numero = {
    "aegypti": 0,
    "albopictus": 1,
    "culex": 2
}

df["etiqueta"] = df["clase"].map(clase_a_numero)

# Comprobar que todas las fotografías existen #
faltantes = df[
    ~df["ruta_completa"].apply(lambda ruta: Path(ruta).is_file())
]

print("\nVerificación de fotografías:")
print("Fotografías encontradas:", len(df) - len(faltantes))
print("Fotografías faltantes:", len(faltantes))

if len(faltantes) > 0:
    raise FileNotFoundError("Hay imágenes que no se encuentran.")

# Preparar las imágenes para la CNN #

tamano_imagen = (128, 128)

def cargar_imagen(ruta):
    imagen = tf.io.read_file(ruta)
    imagen = tf.io.decode_image(
        imagen,
        channels=3,
        expand_animations=False
    )
    imagen.set_shape([None, None, 3])

    imagen = tf.image.resize(imagen, tamano_imagen)
    imagen = tf.cast(imagen, tf.float32) / 255.0

    return imagen


# Probar el procesamiento con una fotografía
ruta_ejemplo = df.iloc[0]["ruta_completa"]
imagen_ejemplo = cargar_imagen(ruta_ejemplo)

print("\nVerificación del procesamiento:")
print("Dimensiones:", imagen_ejemplo.shape)
print("Valor mínimo:", float(tf.reduce_min(imagen_ejemplo)))
print("Valor máximo:", float(tf.reduce_max(imagen_ejemplo)))

# Preparar entrenamiento, validación y prueba #

# Conservar la división original del inventario #
df_train_completo = df[df["split"] == "train"].copy()
df_test = df[df["split"] == "test"].copy()

# Crear un identificador de grupo para cada foto base #
df_train_completo["grupo"] = (
    df_train_completo["fuente"].astype(str)
    + "_"
    + df_train_completo["clase"].astype(str)
    + "_"
    + df_train_completo["foto_base"].astype(str)
)

# Separar una parte del entrenamiento para validación #
separador = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=42
)

indices_train, indices_val = next(
    separador.split(
        df_train_completo,
        groups=df_train_completo["grupo"]
    )
)

df_train = df_train_completo.iloc[indices_train].copy()
df_val = df_train_completo.iloc[indices_val].copy()

print("\nDistribución final:")
print("Entrenamiento:", len(df_train))
print("Validación:", len(df_val))
print("Prueba:", len(df_test))

print("\nClases en entrenamiento:")
print(df_train["clase"].value_counts())

print("\nClases en validación:")
print(df_val["clase"].value_counts())

# Calcular los pesos por clase #

clases_numericas = np.array([0, 1, 2])

pesos = compute_class_weight(
    class_weight="balanced",
    classes=clases_numericas,
    y=df_train["etiqueta"].to_numpy()
)

pesos_clases = {
    int(clase): float(peso)
    for clase, peso in zip(clases_numericas, pesos)
}

print("\nPesos por clase:")

for numero, peso in pesos_clases.items():
    print(f"{CLASES[numero]}: {peso:.4f}")

# Crear los conjuntos de datos #

BATCH_SIZE = 16

def preparar_imagen(ruta, etiqueta):
    imagen = cargar_imagen(ruta)
    return imagen, etiqueta


def crear_dataset(datos, entrenamiento=False):
    rutas = datos["ruta_completa"].astype(str).to_numpy()
    etiquetas = datos["etiqueta"].astype("int32").to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices(
        (rutas, etiquetas)
    )

    if entrenamiento:
        dataset = dataset.shuffle(
            buffer_size=len(datos),
            seed=42
        )

    dataset = dataset.map(
        preparar_imagen,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = crear_dataset(df_train, entrenamiento=True)
val_ds = crear_dataset(df_val)
test_ds = crear_dataset(df_test)

print("\nDatasets creados correctamente")

for imagenes, etiquetas in train_ds.take(1):
    print("Dimensiones del lote:", imagenes.shape)
    print("Dimensiones de etiquetas:", etiquetas.shape)

# Entrenar nuestra CNN propia #

# Crear la red neuronal #
modelo_cnn = crear_cnn()

# Configurar el entrenamiento #
EPOCAS = 30

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

carpeta_modelos = BASE_DIR / "models"
carpeta_modelos.mkdir(parents=True, exist_ok=True)

guardar_modelo = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(carpeta_modelos / "cnn_mosquitos_mejorada.keras"),
    monitor="val_loss",
    save_best_only=True,
    mode="min"
)

# Iniciar el entrenamiento #
print("\nIniciando entrenamiento de la CNN...")

#historial_cnn = modelo_cnn.fit(
    #train_ds,
    #validation_data=val_ds,
    #epochs=EPOCAS,
    #class_weight=pesos_clases,
    #callbacks=[early_stopping, guardar_modelo]
#)

print("\nEntrenamiento de la CNN finalizado.")

# Cargar el mejor modelo de la CNN mejorada
modelo_cnn = tf.keras.models.load_model(
    carpeta_modelos / "cnn_mosquitos_mejorada.keras"
)

# Evaluar nuestra CNN #

#print("\nEvaluando la CNN con las imágenes de prueba...")

# Cargar el mejor modelo que guardamos #
#modelo_cnn = tf.keras.models.load_model(
    #carpeta_modelos / "cnn_mosquitos_mejorada.keras"
#)

# Obtener las predicciones #
#predicciones = modelo_cnn.predict(test_ds)

# Convertir probabilidades en clases (0, 1 o 2) #
#y_pred = np.argmax(predicciones, axis=1)

# Obtener las etiquetas reales #
#y_real = df_test["etiqueta"].to_numpy()

# Calcular el F1 macro #
#f1_macro = f1_score(
    # y_real,
    # y_pred,
    #average="macro",
    #labels=[0, 1, 2],
    #zero_division=0
#)

#print("\nRESULTADOS DE LA CNN")
#print(f"F1 macro: {f1_macro:.4f}")
#print(f"F1 macro (%): {f1_macro * 100:.2f}%")

#print("\nF1 por especie:")
#print(classification_report(
    #y_real,
    #y_pred,
    #labels=[0, 1, 2],
    #target_names=CLASES,
    #zero_division=0
#))

#print("\nMatriz de confusión:")
#print(confusion_matrix(y_real, y_pred, labels=[0, 1, 2]))

# Revisar predicciones en validación #

print("\nEvaluación en validación:")

pred_val = modelo_cnn.predict(val_ds)

y_pred_val = np.argmax(pred_val, axis=1)
y_real_val = df_val["etiqueta"].to_numpy()

print(classification_report(
    y_real_val,
    y_pred_val,
    labels=[0, 1, 2],
    target_names=CLASES,
    zero_division=0
))

print("\nMatriz de confusión de validación:")
print(confusion_matrix(
    y_real_val,
    y_pred_val,
    labels=[0, 1, 2]
))

# Comprobar etiquetas y rutas #

print("\nComprobación de etiquetas:")

for clase in CLASES:
    ejemplos = df_train[df_train["clase"] == clase].head(3)

    print(f"\nEspecie: {clase}")

    for _, fila in ejemplos.iterrows():
        print("Etiqueta:", fila["etiqueta"])
        print("Archivo:", fila["ruta_completa"])

# Visualizar imágenes de cada especie #

import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i, clase in enumerate(CLASES):

    ejemplos = df_train[
        df_train["clase"] == clase
    ].head(3)

    for j, (_, fila) in enumerate(ejemplos.iterrows()):

        imagen = cargar_imagen(fila["ruta_completa"])

        axes[i, j].imshow(imagen.numpy())
        axes[i, j].set_title(clase)
        axes[i, j].axis("off")

plt.tight_layout()
plt.show()

# Preparar imágenes para MobileNetV2 #

def preparar_imagen_mobilenet(ruta, etiqueta):

    # Leer la fotografía original #
    imagen = tf.io.read_file(ruta)

    # Convertirla a RGB #
    imagen = tf.io.decode_image(
        imagen,
        channels=3,
        expand_animations=False
    )

    imagen.set_shape([None, None, 3])

    # Redimensionar a 128x128 #
    imagen = tf.image.resize(imagen, (128, 128))

    # Convertir los píxeles al rango [-1, 1] #
    imagen = preprocess_input(imagen)

    return imagen, etiqueta

# Crear datasets para MobileNetV2 #

def crear_dataset_mobilenet(datos, entrenamiento=False):

    rutas = datos["ruta_completa"].astype(str).to_numpy()
    etiquetas = datos["etiqueta"].astype("int32").to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices(
        (rutas, etiquetas)
    )

    if entrenamiento:
        dataset = dataset.shuffle(
            buffer_size=len(datos),
            seed=42
        )

    dataset = dataset.map(
        preparar_imagen_mobilenet,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_mobilenet = crear_dataset_mobilenet(
    df_train,
    entrenamiento=True
)

val_mobilenet = crear_dataset_mobilenet(df_val)

test_mobilenet = crear_dataset_mobilenet(df_test)

print("\nDatasets de MobileNetV2 creados correctamente.")

for imagenes, etiquetas in train_mobilenet.take(1):
    print("Dimensiones:", imagenes.shape)
    print("Valor mínimo:", float(tf.reduce_min(imagenes)))
    print("Valor máximo:", float(tf.reduce_max(imagenes)))

# Crear MobileNetV2 #

print("\nCreando modelo MobileNetV2...")

modelo_mobilenet = crear_mobilenet()

modelo_mobilenet.summary()

print("\nMobileNetV2 creada correctamente.")

# Configurar el entrenamiento de MobileNetV2 #

EPOCAS_MOBILENET = 30

# Detener el entrenamiento si deja de mejorar #
early_stopping_mobilenet = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Guardar automáticamente el mejor modelo #
guardar_mobilenet = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(carpeta_modelos / "mobilenet_mosquitos.keras"),
    monitor="val_loss",
    save_best_only=True,
    mode="min"
)

print("\nEntrenamiento de MobileNetV2 configurado correctamente.")

# Entrenar MobileNetV2 #

print("\nIniciando entrenamiento de MobileNetV2...")

#historial_mobilenet = modelo_mobilenet.fit(
    #train_mobilenet,
    #validation_data=val_mobilenet,
    ##epochs=EPOCAS_MOBILENET,
    #class_weight=pesos_clases,
    #callbacks=[
        #early_stopping_mobilenet,
        #guardar_mobilenet
    #]
#)

print("\nEntrenamiento de MobileNetV2 finalizado.")

# Evaluar MobileNetV2 en validación #

print("\nEvaluando MobileNetV2 en validación...")

modelo_mobilenet = tf.keras.models.load_model(
    carpeta_modelos / "mobilenet_mosquitos.keras"
)

pred_val_mobilenet = modelo_mobilenet.predict(val_mobilenet)

y_pred_val_mobilenet = np.argmax(pred_val_mobilenet, axis=1)
y_real_val = df_val["etiqueta"].to_numpy()

f1_macro_val_mobilenet = f1_score(
    y_real_val,
    y_pred_val_mobilenet,
    average="macro",
    labels=[0, 1, 2],
    zero_division=0
)

print("\nRESULTADOS DE MOBILENETV2 EN VALIDACIÓN")
print(f"F1 macro: {f1_macro_val_mobilenet:.4f}")
print(f"F1 macro (%): {f1_macro_val_mobilenet * 100:.2f}%")

print("\nF1 por especie:")
print(classification_report(
    y_real_val,
    y_pred_val_mobilenet,
    labels=[0, 1, 2],
    target_names=CLASES,
    zero_division=0
))

print("\nMatriz de confusión de validación:")
print(confusion_matrix(
    y_real_val,
    y_pred_val_mobilenet,
    labels=[0, 1, 2]
))

# Evaluar MobileNetV2 en prueba #

print("\nEvaluando MobileNetV2 en prueba...")

pred_test_mobilenet = modelo_mobilenet.predict(test_mobilenet)

y_pred_test_mobilenet = np.argmax(pred_test_mobilenet, axis=1)
y_real_test = df_test["etiqueta"].to_numpy()

f1_macro_test_mobilenet = f1_score(
    y_real_test,
    y_pred_test_mobilenet,
    average="macro",
    labels=[0, 1, 2],
    zero_division=0
)

print("\nRESULTADOS DE MOBILENETV2 EN PRUEBA")
print(f"F1 macro: {f1_macro_test_mobilenet:.4f}")
print(f"F1 macro (%): {f1_macro_test_mobilenet * 100:.2f}%")

print("\nF1 por especie:")
print(classification_report(
    y_real_test,
    y_pred_test_mobilenet,
    labels=[0, 1, 2],
    target_names=CLASES,
    zero_division=0
))

print("\nMatriz de confusión de prueba:")
print(confusion_matrix(
    y_real_test,
    y_pred_test_mobilenet,
    labels=[0, 1, 2]
))

# Guardar el orden de las clases #

import json

orden_clases = {
    "0": "aegypti",
    "1": "albopictus",
    "2": "culex"
}

ruta_clases = carpeta_modelos / "mosquitos_clases.json"

with open(ruta_clases, "w", encoding="utf-8") as archivo:
    json.dump(orden_clases, archivo, indent=4)

print("\nOrden de clases guardado correctamente.")

# Guardar los resultados de ambas redes #

resultados = {
    "cnn_propia": {
        "modelo": "cnn_mosquitos.keras",
        "f1_macro": 0.5397,
        "f1_por_clase": {
            "aegypti": 0.14,
            "albopictus": 0.64,
            "culex": 0.84
        }
    },
    "mobilenetv2": {
        "modelo": "mobilenet_mosquitos.keras",
        "f1_macro": 0.8830,
        "f1_por_clase": {
            "aegypti": 0.87,
            "albopictus": 0.89,
            "culex": 0.89
        }
    },
    "evaluacion": {
        "conjunto": "test",
        "total_imagenes": 251,
        "clases": ["aegypti", "albopictus", "culex"],
        "meta_f1_macro": 0.85
    }
}

ruta_resultados = carpeta_modelos / "mosquitos_resultados.json"

with open(ruta_resultados, "w", encoding="utf-8") as archivo:
    json.dump(resultados, archivo, indent=4)

print("\nResultados guardados correctamente.")