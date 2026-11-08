# Vector-IA: Asistente Inteligente para la Vigilancia del Dengue en Costa Rica

Proyecto 2 del curso de Inteligencia Artificial Aplicada, Colegio Universitario de Cartago (CUC).

## Propósito

Sistema que ayuda a las autoridades de salud y a las comunidades a vigilar el dengue, con tres modelos de redes neuronales:

| Módulo | Modelo | Qué hace |
|---|---|---|
| 1 | CNN | Identifica la especie del mosquito en una foto (*Aedes aegypti*, *Aedes albopictus* o *Culex*) |
| 2 | RNN / LSTM | Pronostica los casos de dengue de las próximas 4 semanas a partir del clima |
| 3 | ANN | Recomienda la acción preventiva: alerta sanitaria, fumigación focalizada, eliminación de criaderos o monitoreo normal |

> ⚠️ **Aviso:** es una herramienta de apoyo a la vigilancia epidemiológica. **No diagnostica dengue en personas.**

## Estructura del proyecto

```
Proyecto-VectorIA/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/          # CSV originales e imágenes (las imágenes no están en el repositorio)
│   └── processed/    # datos ya limpios
├── notebooks/
│   ├── 01_EDA.ipynb              # análisis exploratorio de los CSV
│   ├── 02_CNN_Mosquitos.ipynb    # análisis de imágenes y preparación para la CNN
│   ├── 03_LSTM_Casos.ipynb       # pronóstico de casos (SimpleRNN y LSTM)
│   └── 04_ANN_Acciones.ipynb     # recomendación de acciones
├── src/              # código fuente
├── models/           # modelos entrenados (.keras / .h5) y sus preprocesadores (.pkl)
├── api/main.py       # API con FastAPI
└── app/Home.py       # aplicación con Streamlit
```

## Requisitos

- **Python 3.11 a 3.13.** TensorFlow 2.21 no se instala con Python 3.14.
- Git.

## Instalación

En Windows (PowerShell):

```
git clone https://github.com/GaboSM1507/Proyecto-VectorIA.git
cd Proyecto-VectorIA
py -3.13 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Datos

Los CSV del paquete `VectorIA_datos` van en `data/raw/`:

| Archivo | Tipo | Uso |
|---|---|---|
| `vectoria_dengue_nacional_real.csv` | Real | Módulo 2: casos y clima semanales (2014-2024) |
| `vectoria_acciones.csv` | Mixto | Módulo 3: acciones recomendadas |
| `vectoria_casos_cantonal_anual_real.csv` | Real | Mapa de calor por cantón (2013-2018) |
| `vectoria_dengue_semanal.csv` | Sintético | Bonus: predicción por provincia |

Las **imágenes de mosquitos no están en el repositorio** porque pesan demasiado. Se descargan y se colocan en `data/raw/imagenes/` (la estructura exacta que espera el código está en `notebooks/02_CNN_Mosquitos.ipynb`):

- IEEE DataPort, *Image Dataset of Aedes and Culex Mosquito Species*: https://ieee-dataport.org/open-access/image-dataset-aedes-and-culex-mosquito-species
- Mendeley Data, *Dataset of Vector Mosquito Images*: https://data.mendeley.com/datasets/88s6fvgg2p
- Kaggle, *Aedes mosquitos* (aegypti vs albopictus): (agregar el enlace del enunciado)

## Cómo usar

Se necesitan dos terminales con el entorno activado.

**1. API** (terminal 1):

```
python -m uvicorn api.main:app --reload
```

La documentación interactiva queda en http://127.0.0.1:8000/docs

**2. Aplicación** (terminal 2):

```
python -m streamlit run app/Home.py
```

### Endpoints de la API

| Endpoint | Modelo | Estado |
|---|---|---|
| `GET /` | Estado del servicio | Listo |
| `POST /predict/accion` | ANN: acción preventiva con su motivo | Listo |
| `POST /predict/rnn` | RNN: pronóstico de casos | En revisión |
| `POST /predict/mosquito` | CNN: especie del mosquito | Pendiente |

## Resultados

### Módulo 1: CNN (identificación de mosquitos)

Pendiente: F1 macro de la CNN propia y de MobileNetV2.

### Módulo 2: RNN / LSTM (pronóstico de casos)

- **Datos:** `dengue_procesado.csv` (casos reales y clima semanales, 2014-2024).
- **Entrada y salida:** las 26 semanas anteriores (casos en logaritmo, lluvia, temperatura máxima y mínima, humedad y la semana del año) para pronosticar las 4 siguientes.
- **Separación:** temporal. Train hasta 2021, validación 2022 y prueba 2023-2024.
- **Redes:** SimpleRNN 32 → 4 salidas (base) y LSTM 64 → LSTM 32 → Dropout 0.2 → 4 salidas (principal), con parada anticipada.

| Modelo | RMSE | MAE |
|---|---|---|
| SimpleRNN | 260 | 175 |
| LSTM | 494 | 334 |
| Baseline estacional (misma semana del año anterior) | 638 | 456 |

Las dos superan al baseline. La LSTM quedó peor que la SimpleRNN: el año de validación (2022) no tuvo brote y la parada anticipada la detuvo antes de que aprendiera a seguirlos. Por eso la API y la app usan la SimpleRNN (`rnn_simple.h5`), y la LSTM queda guardada en `rnn_lstm.h5`. El detalle está en `notebooks/03_LSTM_Casos.ipynb`.

### Módulo 3: ANN (recomendación de acción)

- **Datos:** `vectoria_acciones.csv` (2,060 filas, 412 semanas × 5 escenarios).
- **Separación:** por semana, para que las 5 copias de una misma semana no queden repartidas entre entrenamiento y prueba.
- **Red:** 7 entradas → Dense 32 → Dropout 0.2 → Dense 16 → Dropout 0.2 → 4 salidas (softmax), con pesos por clase y parada anticipada.
- **Resultado:** entrenada 5 veces con semillas distintas, 90.8 % de aciertos en promedio (variación de 1.1 puntos).

| Acción | Precisión | Recall |
|---|---|---|
| alerta_sanitaria | 0.95 | 0.94 |
| eliminacion_criaderos | 0.82 | 0.78 |
| fumigacion_focalizada | 0.86 | 0.87 |
| monitoreo_normal | 0.93 | 0.94 |

Seis de las ocho métricas superan la meta de 85 %.

## Limitaciones

- `eliminacion_criaderos` no alcanza el 85 %: se confunde con `monitoreo_normal`, porque solo las separan los umbrales de lluvia y temperatura.
- Los datos cantonales no cuadran con el total nacional (la suma de los cantones es de 1.3 a 1.7 veces mayor), por lo que se usan solo para comparar cantones entre sí.
- `vectoria_dengue_semanal.csv` es sintético y no representa cifras oficiales.

## Equipo

- Gabriel (arquitectura y backend)
- Jeremy (modelo RNN)
- Jeriel (ANN, API y documentación)

## Fuentes y créditos

- Casos de dengue: OPS/PAHO a través de OpenDengue.
- Clima: NASA POWER.
