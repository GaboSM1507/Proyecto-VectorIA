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

La evaluación se realizó mediante la métrica F1-score macro, utilizando un conjunto de prueba de 251 imágenes (123 de aegypti, 120 de albopictus y 8 de Culex).

Todo el Módulo 1 está en `notebooks/02_CNN_Mosquitos.ipynb`: el análisis de las imágenes y la separación entre entrenamiento y prueba (secciones 1 a 3), el entrenamiento y la evaluación de las redes (sección 4) y el experimento con las fotos recortadas alrededor del mosquito (sección 5). La sección 4 incluye una CNN propia mejorada (`crear_cnn_v2` en `src/train/cnn.py`), también desde cero, elige la CNN propia con los datos de validación y guarda los resultados calculados en `models/mosquitos_resultados.json`. Para correrla hacen falta las imágenes en `data/raw/imagenes/`, que no se incluyen en el repositorio. `src/train/entrenar_mosquitos.py` es la versión anterior del entrenamiento.

#### Resultados de la CNN

| Modelo | F1-score macro | F1 aegypti | F1 albopictus | F1 Culex |
|---|---:|---:|---:|---:|
| CNN propia (sugerida con aumento de datos) | 50.29 % | 0.00 | 0.67 | 0.84 |
| MobileNetV2 | 88.30 % | 0.87 | 0.89 | 0.89 |

Las cifras se calcularon en la sección 4 del notebook 02 y están en `models/mosquitos_resultados.json`, con las matrices de confusión. Antes de esa sección, el 53.97 % de la CNN propia estaba escrito a mano y no se sabía de qué modelo era. Se midieron tres CNN propias en validación: la sugerida (44.5 %), la sugerida con aumento de datos (45.2 %) y la v2 (24.4 %). Se eligió la de aumento de datos y se evaluó una sola vez en prueba.

En esa evaluación, MobileNetV2 superó la meta del 85 % y la CNN propia no. La CNN propia no reconocía *Aedes aegypti*, justamente la especie que transmite el dengue: respondió *albopictus* en 120 de las 123 fotos de *aegypti*. Las dos especies de *Aedes* se distinguen por el dibujo de escamas blancas del tórax. En las fotos de Kaggle el mosquito ocupa una parte pequeña de una foto de unos 2464 × 3280 píxeles, así que al reducirla a 128 o 192 píxeles ese detalle casi desaparece.

#### CNN propia con recorte (sección 5): cumple la meta

Para resolverlo, cada foto se recorta alrededor del mosquito antes de reducirla. El recorte busca los píxeles que se alejan del color del fondo y no usa la especie. Se entrenó la misma red v2, desde cero, con recortes de 128 × 128, y se eligió la mejor época con el F1 macro en validación.

| Modelo (mismas 247 fotos de prueba) | F1 macro | F1 aegypti | F1 albopictus | F1 Culex |
|---|---:|---:|---:|---:|
| **CNN propia con recorte** | **88.00 %** | 0.95 | 0.96 | 0.73 |
| MobileNetV2 | 85.33 % | 0.87 | 0.89 | 0.80 |
| CNN propia de la sección 4 | 46.46 % | 0.00 | 0.67 | 0.73 |

La CNN propia con recorte **supera la meta del 85 %** y acierta 112 de las 123 fotos de *aegypti*. Las 3 que llama *Culex* son las de *aegypti* con fondo blanco (Mendeley), el mismo fondo de las fotos de *Culex*: para *Culex*, la red se apoya en parte en el fondo (ver las limitaciones). Se eligió con validación (F1 de 0.956) y se evaluó una sola vez en prueba. Las 22 fotos de *Culex* de IEEE DataPort no se pudieron volver a descargar, porque la página pide una suscripción de pago. Por eso este experimento usa 247 fotos de prueba en vez de 251, con solo 4 de *Culex*, y todos los modelos se compararon con esas mismas fotos.

La aplicación sigue usando MobileNetV2: recibe la foto completa, mientras que el recorte depende de un fondo parejo como el de las fotos de Kaggle.

### Módulo 2. Pronóstico de casos de dengue (RNN/LSTM)

Este módulo utiliza información histórica de dengue y variables climáticas para estimar los casos de las próximas cuatro semanas.

Los datos son reales: `dengue_procesado.csv` (566 semanas, de 2014 a 2024), generado en el EDA a partir de los casos de OPS/PAHO y el clima de NASA POWER.

Las variables utilizadas incluyen:

- Casos de dengue (en logaritmo).
- Precipitación o lluvia.
- Temperatura máxima.
- Temperatura mínima.
- Humedad relativa.
- Semana del año (como seno y coseno).

Se utilizan ventanas de 26 semanas anteriores para generar los pronósticos.

La separación es temporal: entrenamiento hasta 2021, validación con 2022 y prueba con 2023–2024.

Se entrenaron dos modelos, ambos con parada anticipada:

- **SimpleRNN:** SimpleRNN de 32 → 4 salidas (1,412 parámetros). Modelo de referencia.
- **LSTM:** LSTM de 64 → LSTM de 32 → Dropout 0.2 → 4 salidas (30,980 parámetros). Modelo de memoria a largo plazo.

También se realizó una comparación con un baseline estacional, que utiliza los datos de la misma semana del año anterior.

#### Resultados de los modelos

| Modelo | RMSE | MAE |
|---|---:|---:|
| SimpleRNN | 260 | 175 |
| LSTM | 494 | 334 |
| Baseline estacional | 638 | 456 |
| Persistencia (repetir la última semana, control) | 216 | 141 |

Los resultados muestran que ambos modelos superaron al baseline estacional, como pide el enunciado. Ninguno supera todavía a la persistencia: la SimpleRNN solo le gana a 4 semanas de distancia (RMSE de 235 contra 279), y las dos redes subestiman los brotes.

SimpleRNN obtuvo menores errores que LSTM, por dos razones:

- El año de validación (2022) no tuvo brote, y la parada anticipada detuvo a la LSTM antes de que aprendiera a seguir los brotes de 2023 y 2024.
- La LSTM tiene unas 22 veces más parámetros que la SimpleRNN y hay pocos datos de entrenamiento (381 ventanas).

Por esta razón, se seleccionó SimpleRNN para realizar los pronósticos dentro de la API y la aplicación. El modelo LSTM se conserva para comparación. El detalle está en `notebooks/03_LSTM_Casos.ipynb`.

#### Bonus: pronóstico por provincia (datos sintéticos)

El enunciado propone como bonus pronosticar por provincia con `vectoria_dengue_semanal.csv`. Ese archivo es **sintético** (lo generó el paquete del curso, no son cifras oficiales), así que el resultado muestra que el método funciona, no que la red pronostique el dengue real de cada provincia.

- Una sola SimpleRNN (la misma red base) para las 7 provincias, con la provincia como variable *one-hot*: 26 semanas → 4 semanas.
- Separación temporal: entrenamiento de 2014 a 2022, validación con 2023 y prueba con 2024–2025 (707 ventanas).

| Modelo | RMSE | MAE |
|---|---:|---:|
| SimpleRNN por provincia | 27.8 | 15.4 |
| Baseline estacional | 49.0 | 30.5 |
| Persistencia (control) | 35.3 | 19.9 |

Supera al baseline estacional en las 7 provincias (entre 29 % y 52 % menos RMSE) y, a diferencia del modelo nacional, también a la persistencia. El detalle está en `notebooks/05_Bonus_Provincias.ipynb`. La API lo ofrece en `POST /predict/casos/provincia` y la app en la pestaña de pronóstico.

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
| Resumen | Presenta el estado de la API y de los modelos, y los casos del último año. |
| Identificación CNN | Permite cargar una imagen y obtener la clasificación mediante CNN. |
| Pronóstico RNN | Se elige una semana; la SimpleRNN usa las 26 semanas que terminan ahí y pronostica las 4 siguientes. El gráfico las compara con el umbral endémico y, si son semanas pasadas, con los casos reales. Un selector cambia al bonus por provincia (datos sintéticos), con un ranking de las 7 provincias según su umbral. |
| Prevención ANN | Utiliza el pronóstico, el clima de esa semana, el umbral endémico y el resultado de la CNN para recomendar una acción con su motivo. Incluye un simulador "¿Qué pasaría si…?" con deslizadores de casos, lluvia y temperatura. |
| Mapa cantonal | Mapa de Costa Rica coloreado según los casos de cada cantón, con animación de 2013 a 2018, vista 3D, ranking de cantones y evolución anual, con filtros por provincia y año. |

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
├── INFORME_TECNICO.md
├── USO_DE_IA.md
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml        (tema claro de la app)
│
├── data/
│   ├── raw/
│   │   ├── vectoria_acciones.csv
│   │   ├── vectoria_casos_cantonal_anual_real.csv
│   │   ├── cantones_costa_rica.geojson
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
│   ├── 04_ANN_Acciones.ipynb
│   └── 05_Bonus_Provincias.ipynb   (bonus: pronóstico por provincia)
│
├── src/
│   ├── __init__.py
│   ├── data_prep.py
│   ├── preparar_mapa_cantones.py
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
│   ├── cnn_mosquitos_v2.keras
│   ├── cnn_mosquitos_recorte.keras   (CNN propia con recorte: 88 %)
│   ├── mosquitos_clases.json
│   ├── mosquitos_resultados.json
│   ├── rnn_simple.h5
│   ├── rnn_simple_preprocesado.pkl
│   ├── rnn_lstm.h5
│   ├── rnn_lstm_preprocesado.pkl
│   ├── rnn_provincias.h5
│   ├── rnn_provincias_preprocesado.pkl
│   ├── rnn_provincias_resultados.json
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

| Archivo | Tipo | Uso |
|---|---|---|
| `vectoria_dengue_nacional_real.csv` | Real | Casos nacionales de dengue y variables climáticas. |
| `dengue_procesado.csv` | Real | Serie semanal preparada en el EDA para el pronóstico (Módulo 2). |
| `vectoria_acciones.csv` | Mixto | Datos para entrenar el modelo de recomendaciones: casos y clima reales, escenarios simulados y etiquetas generadas con reglas. |
| `vectoria_casos_cantonal_anual_real.csv` | Real | Comparación de casos por cantón (2013–2018). |
| `cantones_costa_rica.geojson` | Real | Fronteras de los cantones para el mapa de la app (geoBoundaries, dominio público). Se genera con `src/preparar_mapa_cantones.py`. |
| `vectoria_dengue_semanal.csv` | Sintético | Versión por provincia. Se analiza en el EDA, que genera `dengue_provincias_limpio.csv`, y solo lo usa el bonus de pronóstico por provincia. |

El análisis de todos estos archivos está en `notebooks/01_EDA.ipynb`.

Los datos cantonales se utilizan para comparar la distribución entre cantones. Sus sumas no coinciden con el total nacional, por lo que no deben interpretarse como un total oficial de casos.

Las imágenes de mosquitos utilizadas para el entrenamiento no están incluidas en el repositorio debido a su tamaño. Para conocer las fuentes y el procedimiento de preparación, se puede consultar el notebook `02_CNN_Mosquitos.ipynb`.

Entre las fuentes de imágenes consideradas se encuentran:

- [IEEE DataPort – Image Dataset of Aedes and Culex Mosquito Species](https://ieee-dataport.org/open-access/image-dataset-aedes-and-culex-mosquito-species)
- [Mendeley Data – Dataset of Vector Mosquito Images](https://data.mendeley.com/datasets/88s6fvgg2p)
- [Kaggle – Aedes Mosquitos Dataset](https://www.kaggle.com/datasets/pradeepisawasan/aedes-mosquitos)

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

La API estará disponible en la siguiente dirección. Abierta en el navegador, muestra una página de inicio con el estado de cada modelo y gráficos del pronóstico generados en el momento:

```text
http://127.0.0.1:8000
```

La documentación interactiva de los endpoints, con ejemplos listos para probar, se encuentra en:

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
| GET | `/` | En el navegador muestra la página de inicio; para la app y otros programas devuelve el estado en JSON. |
| GET | `/estado` | Estado de los modelos (JSON). |
| POST | `/predict/mosquito` | Clasifica una imagen de mosquito mediante CNN. |
| POST | `/predict/casos` | Pronostica los casos de dengue de las próximas cuatro semanas a partir de 26 semanas (SimpleRNN). También responde en `/predict/rnn`, el nombre anterior. |
| POST | `/predict/accion` | Recomienda una acción preventiva mediante ANN. |
| POST | `/predict/casos/provincia` | Bonus: pronóstico de una provincia (datos sintéticos). |
| GET | `/visual/nacional` | Gráfico PNG del pronóstico nacional contra el umbral endémico. Acepta `?fecha_corte=AAAA-MM-DD`. |
| GET | `/visual/provincia/{provincia}` | Gráfico PNG del pronóstico de una provincia (bonus). |
| GET | `/visual/provincias` | Panel PNG de las 7 provincias con un ranking según su umbral (bonus). |

Los endpoints pueden probarse desde la documentación interactiva (`/docs`). Cada endpoint de predicción trae ejemplos con datos del proyecto, y los gráficos de `/visual` aparecen ahí mismo al presionar *Execute*.

## 10. Limitaciones del proyecto

Aunque VectorIA integra los tres modelos y permite utilizarlos desde una aplicación web, existen aspectos que deben considerarse:

- La CNN propia solo supera la meta con las fotos recortadas alrededor del mosquito (88.00 %). Ese resultado se midió sin las 22 fotos de Culex de IEEE DataPort, con solo 4 de Culex en prueba, y el recorte depende de un fondo parejo como el de Kaggle.
- La clase eliminación de criaderos de la ANN obtuvo valores de precisión y recall inferiores al 85 %.
- La CNN siempre responde una de las tres especies: no tiene una clase para imágenes que no son mosquitos.
- La ANN se entrenó con escenarios simulados y etiquetas generadas con reglas, no con decisiones reales de las autoridades sanitarias.
- El año de validación del Módulo 2 (2022) no tuvo brote; con un año que incluya un brote, la comparación entre SimpleRNN y LSTM podría cambiar.
- Ninguna red del Módulo 2 supera a la persistencia (repetir la última semana), y las dos subestiman los brotes.
- La prueba de la CNN tiene solo 8 fotos de Culex, y cada especie viene sobre todo de una fuente distinta, así que la red podría estar aprendiendo la fuente además del mosquito. Por ejemplo, de las 3 fotos de prueba de *aegypti* con fondo blanco (Mendeley), el mismo fondo de las fotos de *Culex*, MobileNetV2 clasificó 2 como *Culex* y la CNN propia con recorte, las 3.
- La lluvia de NASA POWER se estima con un modelo, no con estaciones. La semana del 26 de mayo de 2024 marca 625 mm, pero ni ERA5 ni las estaciones del IMN registran esa lluvia: es un error de la fuente. Se dejó documentado sin cambiar los datos; su efecto en el pronóstico es pequeño (RMSE de 260.2 a 255.4 si se corrige).
- Los datos cantonales presentan diferencias respecto al total nacional y se utilizan para comparaciones entre cantones.
- El conjunto de datos sintético no representa cifras oficiales. Solo se usa en el EDA y en el bonus por provincia, cuyos resultados muestran que el método funciona, no que pronostique el dengue real de cada provincia.
- Las recomendaciones generadas son orientativas y deben ser interpretadas por personal capacitado.

## 11. Posibles mejoras futuras

- Probar la CNN propia con recorte en fotos de campo (fondos variados) y, si funciona, llevar el recorte a la API para usarla en la aplicación.
- Conseguir más imágenes de *Culex* y mejorar el balance entre clases.
- Aumentar la precisión y el recall de la clase eliminación de criaderos en la ANN.
- Incorporar nuevos datos epidemiológicos y climáticos.
- Ampliar las visualizaciones para facilitar el análisis de los resultados.
- Incorporar nuevas fuentes de información para fortalecer los pronósticos.
- Entrenar el modelo por provincia con datos reales semanales por provincia, que no venían en el paquete del curso.

## 12. Equipo de trabajo

- **Gabriel:** arquitectura y backend.
- **Jeremy:** Módulo 2 (SimpleRNN y LSTM) e integración del informe.
- **Jeriel:** desarrollo del modelo ANN, integración de la API y documentación.
- **Yanely:** análisis exploratorio de datos (EDA).

El informe técnico completo está en `INFORME_TECNICO.md` y el uso de herramientas de inteligencia artificial está documentado en `USO_DE_IA.md`.

## 13. Fuentes de información

- **OPS/PAHO:** datos de casos de dengue mediante OpenDengue.
- **NASA POWER:** información climática.
- **ERA5 (Copernicus), mediante Open-Meteo, e Instituto Meteorológico Nacional (IMN):** comparación de la lluvia de mayo de 2024.
- **IEEE DataPort:** imágenes de especies de mosquitos.
- **Mendeley Data:** conjunto de imágenes de mosquitos vectores.
- **Kaggle:** [Aedes Mosquitos Dataset](https://www.kaggle.com/datasets/pradeepisawasan/aedes-mosquitos) (Aedes aegypti y Aedes albopictus).
- **geoBoundaries:** fronteras de los cantones de Costa Rica para el mapa (dominio público, CC0).

---

**VectorIA – Inteligencia artificial al servicio de la vigilancia y prevención del dengue en Costa Rica.**