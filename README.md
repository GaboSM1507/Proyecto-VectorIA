# VectorIA: Asistente Inteligente para la Vigilancia del Dengue en Costa Rica

**Proyecto 2 – Inteligencia Artificial Aplicada**  
Colegio Universitario de Cartago (CUC)

---

## 1. Descripción del proyecto

VectorIA es un sistema inteligente desarrollado en Python que busca apoyar la vigilancia epidemiológica del dengue en Costa Rica mediante el uso de modelos de inteligencia artificial y aprendizaje profundo.

El proyecto integra tres modelos de redes neuronales que permiten identificar especies de mosquitos a partir de imágenes, pronosticar los casos de dengue de las próximas cuatro semanas y recomendar acciones preventivas según las condiciones analizadas.

El sistema utiliza datos históricos de casos de dengue y variables climáticas, así como imágenes de mosquitos para realizar las predicciones y clasificaciones.

> **Aviso importante:** VectorIA es una herramienta de apoyo a la vigilancia epidemiológica. No diagnostica dengue en personas ni sustituye la valoración de profesionales de la salud.

## 2. Objetivos

### Objetivo general

Desarrollar un sistema inteligente que contribuya a la vigilancia del dengue en Costa Rica mediante modelos de inteligencia artificial capaces de identificar mosquitos vectores, pronosticar casos de dengue y recomendar acciones preventivas.

### Objetivos específicos

- Clasificar especies de mosquitos mediante una red neuronal convolucional (CNN).
- Comparar una CNN desarrollada desde cero con MobileNetV2 mediante transferencia de aprendizaje.
- Pronosticar los casos de dengue de las próximas cuatro semanas utilizando información histórica y variables climáticas.
- Comparar los resultados de los modelos SimpleRNN y LSTM.
- Recomendar acciones preventivas mediante una red neuronal artificial (ANN) multiclase.
- Integrar los modelos en una aplicación interactiva desarrollada con Streamlit.

## 3. Módulos del sistema

### Módulo 1. Identificación de mosquitos (CNN)

Este módulo utiliza redes neuronales convolucionales para clasificar imágenes de mosquitos en tres categorías:

- **Aedes aegypti:** principal vector del dengue.
- **Aedes albopictus:** vector secundario.
- **Culex:** mosquito que no transmite el dengue.

Se desarrollaron y compararon dos modelos:

1. **CNN propia:** red neuronal construida desde cero.
2. **MobileNetV2:** modelo que utiliza transferencia de aprendizaje.

La evaluación se realizó mediante la métrica F1-score macro, utilizando un conjunto de prueba de 251 imágenes.

#### Resultados de la CNN

| Modelo | F1-score macro |
|---|---:|
| CNN propia | 53.97 % |
| MobileNetV2 | 88.30 % |

MobileNetV2 obtuvo el mejor resultado y superó la meta establecida del 85 %. Por esta razón, se utiliza como modelo principal para la identificación de mosquitos en la aplicación.

La CNN propia se conserva como modelo de comparación.

### Módulo 2. Pronóstico de casos de dengue (RNN/LSTM)

Este módulo utiliza información histórica de dengue y variables climáticas para estimar los casos de las próximas cuatro semanas.

Las variables utilizadas incluyen:

- Casos de dengue.
- Precipitación o lluvia.
- Temperatura máxima.
- Temperatura mínima.
- Humedad relativa.
- Semana epidemiológica.

Se utilizan ventanas de 26 semanas anteriores para generar los pronósticos.

Se entrenaron dos modelos:

- **SimpleRNN:** modelo de referencia.
- **LSTM:** modelo de memoria a largo plazo.

También se realizó una comparación con un baseline estacional, que utiliza los datos de la misma semana del año anterior.

#### Resultados de los modelos

| Modelo | RMSE | MAE |
|---|---:|---:|
| SimpleRNN | 260 | 175 |
| LSTM | 494 | 334 |
| Baseline estacional | 638 | 456 |

Los resultados muestran que ambos modelos superaron al baseline estacional. Sin embargo, SimpleRNN obtuvo menores errores que LSTM.

Por esta razón, se seleccionó SimpleRNN para realizar los pronósticos dentro de la API y la aplicación. El modelo LSTM se conserva para comparación.

### Módulo 3. Recomendación de acciones preventivas (ANN)

Este módulo utiliza una red neuronal artificial (ANN) para clasificar las condiciones analizadas y recomendar una acción preventiva.

El modelo trabaja con siete variables de entrada:

- Casos de dengue pronosticados.
- Umbral endémico.
- Lluvia acumulada de 14 días.
- Temperatura máxima.
- Temperatura mínima.
- Humedad.
- Detección de Aedes aegypti.

El conjunto de datos utilizado contiene 2,060 registros, correspondientes a 412 semanas y cinco escenarios.

La clasificación contempla cuatro posibles recomendaciones:

1. **Alerta sanitaria.**
2. **Fumigación focalizada.**
3. **Eliminación de criaderos.**
4. **Monitoreo normal.**

#### Resultados de la ANN

El modelo fue entrenado cinco veces con diferentes semillas y obtuvo un promedio de 90.8 % de exactitud.

| Acción | Precisión | Recall |
|---|---:|---:|
| Alerta sanitaria | 95 % | 94 % |
| Eliminación de criaderos | 82 % | 78 % |
| Fumigación focalizada | 86 % | 87 % |
| Monitoreo normal | 93 % | 94 % |

Seis de las ocho métricas de precisión y recall superan el objetivo del 85 %.

La clase de eliminación de criaderos presenta un rendimiento menor al esperado, por lo que constituye una oportunidad de mejora para futuras versiones.

## 4. Aplicación web

La aplicación fue desarrollada con **Streamlit** y permite utilizar los tres modelos desde una misma interfaz.

Cuenta con cinco pestañas:

| Pestaña | Funcionalidad |
|---|---|
| Resumen | Presenta información general y datos del proyecto. |
| Identificar mosquito | Permite cargar una imagen y obtener la clasificación mediante CNN. |
| Pronóstico dengue | Genera un pronóstico de los casos para las próximas cuatro semanas. |
| Acciones preventivas | Utiliza la ANN para recomendar una acción según las variables ingresadas. |
| Mapa por cantón | Permite explorar la distribución de los casos mediante visualizaciones cantonales. |

La aplicación conecta los modelos mediante una API desarrollada con FastAPI.

## 5. Tecnologías utilizadas

- **Python:** lenguaje principal del proyecto.
- **TensorFlow y Keras:** construcción y entrenamiento de redes neuronales.
- **Pandas y NumPy:** procesamiento y análisis de datos.
- **Scikit-learn:** escalamiento, codificación y evaluación de modelos.
- **OpenCV:** herramientas para el procesamiento de imágenes.
- **Matplotlib y Seaborn:** visualización y análisis exploratorio.
- **Plotly:** gráficos interactivos.
- **Streamlit:** desarrollo de la interfaz web.
- **FastAPI:** creación de los servicios para conectar los modelos.
- **Uvicorn:** ejecución del servidor de la API.

## 6. Estructura del proyecto

```text
Proyecto-VectorIA/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── vectoria_acciones.csv
│   │   ├── vectoria_casos_cantonal_anual_real.csv
│   │   ├── vectoria_dengue_nacional_real.csv
│   │   ├── vectoria_dengue_semanal.csv
│   │   └── LEEME_datos.md
│   │
│   └── processed/
│       ├── dengue_procesado.csv
│       ├── dengue_provincias_limpio.csv
│       └── inventario_imagenes.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_CNN_Mosquitos.ipynb
│   ├── 03_LSTM_Casos.ipynb
│   └── 04_ANN_Acciones.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_prep.py
│   └── train/
│       ├── ann.py
│       ├── cnn.py
│       ├── entrenar_mosquitos.py
│       └── lstm.py
│
├── models/
│   ├── ann_acciones.keras
│   ├── ann_acciones_preprocesado.pkl
│   ├── ann_columns.pkl
│   ├── mobilenet_mosquitos.keras
│   ├── cnn_mosquitos.keras
│   ├── cnn_mosquitos_mejorada.keras
│   ├── mosquitos_clases.json
│   ├── mosquitos_resultados.json
│   ├── rnn_simple.h5
│   ├── rnn_simple_preprocesado.pkl
│   ├── rnn_lstm.h5
│   ├── rnn_lstm_preprocesado.pkl
│   └── scaler_ann.pkl
│
├── api/
│   └── main.py
│
└── app/
    └── Home.py
```

## 7. Datos utilizados

El proyecto utiliza diferentes conjuntos de datos para entrenar y evaluar sus modelos.

| Archivo | Uso |
|---|---|
| `vectoria_dengue_nacional_real.csv` | Casos nacionales de dengue y variables climáticas. |
| `vectoria_acciones.csv` | Datos para entrenar el modelo de recomendaciones. |
| `vectoria_casos_cantonal_anual_real.csv` | Información para el mapa de calor por cantón. |
| `vectoria_dengue_semanal.csv` | Datos sintéticos para análisis adicional. |
| `dengue_procesado.csv` | Serie semanal preparada para el pronóstico. |

Los datos cantonales se utilizan para comparar la distribución entre cantones. Sus sumas no coinciden con el total nacional, por lo que no deben interpretarse como un total oficial de casos.

Las imágenes de mosquitos utilizadas para el entrenamiento no están incluidas en el repositorio debido a su tamaño. Para conocer las fuentes y el procedimiento de preparación, se puede consultar el notebook `02_CNN_Mosquitos.ipynb`.

Entre las fuentes de imágenes consideradas se encuentran:

- [IEEE DataPort – Image Dataset of Aedes and Culex Mosquito Species](https://ieee-dataport.org/open-access/image-dataset-aedes-and-culex-mosquito-species)
- [Mendeley Data – Dataset of Vector Mosquito Images](https://data.mendeley.com/datasets/88s6fvgg2p)
- Kaggle – conjunto de imágenes de mosquitos Aedes.

## 8. Instalación y ejecución

### Requisitos previos

- Python 3.11 a 3.13.
- Git.
- Conexión a Internet para instalar las dependencias.

Se recomienda utilizar una versión de Python compatible con TensorFlow 2.21.

### Clonar el repositorio

```bash
git clone https://github.com/GaboSM1507/Proyecto-VectorIA.git
```

Ingresar a la carpeta:

```bash
cd Proyecto-VectorIA
```

### Crear el entorno virtual

En Windows PowerShell:

```bash
py -3.13 -m venv .venv
```

Activar el entorno:

```bash
.\.venv\Scripts\activate
```

### Instalar las dependencias

```bash
python -m pip install -r requirements.txt
```

### Ejecutar la API

Abrir una primera terminal con el entorno virtual activado y ejecutar:

```bash
python -m uvicorn api.main:app --reload
```

La API estará disponible en:

```text
http://127.0.0.1:8000
```

La documentación interactiva de los endpoints se encuentra en:

```text
http://127.0.0.1:8000/docs
```

### Ejecutar la aplicación Streamlit

Abrir una segunda terminal con el entorno virtual activado y ejecutar:

```bash
python -m streamlit run app/Home.py
```

La aplicación se abrirá en el navegador y permitirá acceder a las cinco pestañas del sistema.

**Importante:** mantener la terminal de la API abierta mientras se utiliza Streamlit.

## 9. Endpoints de la API

La API permite que la aplicación se comunique con los modelos entrenados.

| Método | Endpoint | Función |
|---|---|---|
| GET | `/` | Comprueba el estado del servicio. |
| POST | `/predict/mosquito` | Clasifica una imagen de mosquito mediante CNN. |
| POST | `/predict/rnn` | Pronostica los casos de dengue de las próximas cuatro semanas. |
| POST | `/predict/accion` | Recomienda una acción preventiva mediante ANN. |

Los endpoints pueden probarse desde la documentación interactiva de FastAPI.

## 10. Limitaciones del proyecto

Aunque VectorIA integra los tres modelos y permite utilizarlos desde una aplicación web, existen aspectos que deben considerarse:

- La CNN propia obtuvo un F1-score macro inferior a la meta establecida. MobileNetV2 presentó mejores resultados.
- La clase eliminación de criaderos de la ANN obtuvo valores de precisión y recall inferiores al 85 %.
- La SimpleRNN obtuvo mejores resultados que la LSTM en las evaluaciones realizadas, por lo que fue seleccionada para el pronóstico de la aplicación.
- Los datos cantonales presentan diferencias respecto al total nacional y se utilizan para comparaciones entre cantones.
- El conjunto de datos sintético no representa cifras oficiales.
- Las recomendaciones generadas son orientativas y deben ser interpretadas por personal capacitado.

## 11. Posibles mejoras futuras

- Mejorar el rendimiento de la CNN propia mediante más imágenes y un mejor balance entre clases.
- Aumentar la precisión y el recall de la clase eliminación de criaderos en la ANN.
- Incorporar nuevos datos epidemiológicos y climáticos.
- Ampliar las visualizaciones para facilitar el análisis de los resultados.
- Implementar escenarios interactivos para observar cómo cambian las recomendaciones al modificar las variables.
- Incorporar nuevas fuentes de información para fortalecer los pronósticos.

## 12. Equipo de trabajo

- **Gabriel:** arquitectura y backend.
- **Jeremy:** desarrollo del modelo RNN.
- **Jeriel:** desarrollo del modelo ANN, integración de la API y documentación.

## 13. Fuentes de información

- **OPS/PAHO:** datos de casos de dengue mediante OpenDengue.
- **NASA POWER:** información climática.
- **IEEE DataPort:** imágenes de especies de mosquitos.
- **Mendeley Data:** conjunto de imágenes de mosquitos vectores.
- **Kaggle:** conjunto de imágenes de mosquitos Aedes.

---

**VectorIA – Inteligencia artificial al servicio de la vigilancia y prevención del dengue en Costa Rica.**