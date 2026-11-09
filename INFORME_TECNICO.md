# INFORME TÉCNICO
# VectorIA: Asistente Inteligente para la Vigilancia del Dengue en Costa Rica

## 1. Introducción

VectorIA es un sistema inteligente desarrollado para apoyar la vigilancia del dengue en Costa Rica mediante el uso de técnicas de inteligencia artificial y aprendizaje automático.

El proyecto integra tres modelos de redes neuronales con funciones diferentes: una red convolucional (CNN) para identificar especies de mosquitos a partir de imágenes, una red recurrente (RNN) para pronosticar casos de dengue y una red neuronal artificial (ANN) para recomendar acciones preventivas.

Además, cuenta con una aplicación interactiva desarrollada con Streamlit y una API implementada con FastAPI, que permiten utilizar los modelos desde una misma solución.

VectorIA está diseñado como una herramienta de apoyo para la vigilancia epidemiológica y el control de vectores. No realiza diagnósticos médicos en personas ni sustituye las decisiones de las autoridades sanitarias.

## 2. Objetivos del proyecto

### 2.1. Objetivo general

Desarrollar un sistema inteligente que apoye la vigilancia del dengue en Costa Rica mediante la identificación de mosquitos, el pronóstico de casos y la recomendación de acciones preventivas.

### 2.2. Objetivos específicos

- Desarrollar un modelo CNN capaz de clasificar imágenes de mosquitos en tres categorías: Aedes aegypti, Aedes albopictus y Culex.
- Implementar modelos RNN y LSTM para pronosticar los casos de dengue de las siguientes cuatro semanas.
- Desarrollar un modelo ANN que recomiende acciones preventivas según los datos disponibles.
- Integrar los tres modelos en una aplicación interactiva.
- Facilitar la consulta de información relacionada con los casos de dengue por cantón.

## 3. Arquitectura general del sistema

VectorIA está compuesto por tres módulos principales:

1. **Módulo de identificación de mosquitos:** recibe una imagen y utiliza una CNN para clasificar la especie.
2. **Módulo de pronóstico:** analiza información histórica y variables climáticas para estimar los casos de dengue de las próximas cuatro semanas.
3. **Módulo de recomendación:** utiliza información de casos, clima y detección de mosquitos para sugerir acciones preventivas.

Los modelos se integran mediante una API desarrollada con FastAPI y una interfaz gráfica creada con Streamlit.

## 4. Datos y análisis exploratorio (EDA)

Antes de entrenar los modelos se revisaron los archivos de datos en el notebook `01_EDA.ipynb`, para comprobar si estaban limpios, cómo se comportaban y qué había que tener en cuenta al usarlos.

### 4.1. Archivos utilizados

| Archivo | Tipo | Uso |
|---|---|---|
| `vectoria_dengue_nacional_real.csv` | Real: casos de OPS/PAHO (OpenDengue) y clima de NASA POWER | Casos y clima semanales del país. A partir de él se genera `dengue_procesado.csv`, que usa el Módulo 2. |
| `vectoria_acciones.csv` | Mixto: casos y clima reales, escenarios simulados y etiquetas generadas con reglas | Entrenamiento de la ANN (Módulo 3). |
| `vectoria_casos_cantonal_anual_real.csv` | Real | Mapa de casos por cantón en la aplicación. |
| `cantones_costa_rica.geojson` | Fronteras de geoBoundaries (dominio público) | Forma de los 83 cantones para dibujar el mapa. |
| `vectoria_dengue_semanal.csv` | Sintético | Versión por provincia. Se analiza en el EDA y solo lo usa el bonus de pronóstico por provincia (sección 5.4). |
| Imágenes de mosquitos | IEEE DataPort, Mendeley Data, Kaggle, Mosquito Alert y Mosquito-on-human-skin | Entrenamiento y evaluación de la CNN (Módulo 1). |

### 4.2. Principales hallazgos

**Casos y clima del país (Módulo 2):**

- Los datos están limpios: 574 semanas, sin valores vacíos, sin filas repetidas y sin semanas faltantes.
- Hubo tres brotes: 2016, 2023 y 2024. Los de 2023 y 2024 son los más grandes, con más de 30,000 casos al año.
- Los casos siguen una época del año: son bajos en marzo y abril, y más altos de julio a noviembre, con el pico en agosto.
- Pocas semanas tienen muchísimos casos, por lo que se trabajó con el logaritmo de los casos.
- El clima actúa con retraso: la temperatura mínima es la variable más relacionada con los casos, con 6 a 8 semanas de diferencia (correlación de 0.65 a las 8 semanas).
- La mayoría de las lluvias extremas coincide con huracanes y tormentas reales. La excepción es la semana del 26 de mayo de 2024 (625 mm), que resultó ser un error de la fuente (ver la sección 8).
- Se generó `dengue_procesado.csv`: 566 semanas, de febrero de 2014 a diciembre de 2024, con el logaritmo de los casos, el mes y el clima de 2, 4, 6 y 8 semanas atrás.

**Acciones recomendadas (Módulo 3):**

- El archivo está limpio: 2,060 filas (412 semanas × 5 escenarios).
- Cada semana aparece 5 veces con el mismo clima, por lo que el entrenamiento y la prueba se separan por semana y no por fila.
- Las clases están desbalanceadas: monitoreo normal 59 %, alerta sanitaria 18 %, eliminación de criaderos 16 % y fumigación focalizada 7 %. Por eso se usaron pesos por clase y se reportan la precisión y el recall de cada una.
- Las reglas de etiquetado explican el 97.9 % de las etiquetas; el resto es ruido agregado a propósito.

**Casos por cantón:**

- El archivo tiene 422 filas (72 cantones, de 2013 a 2018). Faltan algunas filas (Alvarado y Jiménez en 2016 y 2018, Alfaro Ruiz en 2018) y el cantón San José solo aparece en 2018.
- La suma de los cantones es entre 1.3 y 1.7 veces mayor que el total del país, por lo que el archivo se usa solo para comparar cantones entre sí.
- Los casos están muy concentrados: 10 de los 72 cantones tienen el 49 % de los casos, y Guanacaste, Puntarenas y Limón suman el 68 %. Cartago casi no tiene casos.

**Casos por provincia (sintético):**

- Tiene 4,382 filas (7 provincias × 626 semanas). Había 97 filas con valores vacíos (cerca del 0.7 %), que se rellenaron con el promedio de la semana anterior y la siguiente de la misma provincia.
- No se comporta como los datos reales, por lo que sus resultados no se mezclan con los del Módulo 2.

## 5. Desarrollo de los modelos

### 5.1. Modelo CNN: identificación de mosquitos

El primer módulo utiliza una red neuronal convolucional (CNN), especializada en el análisis de imágenes, para identificar tres categorías de mosquitos:

- Aedes aegypti.
- Aedes albopictus.
- Culex.

Se desarrolló una CNN propia desde cero y se realizó una comparación con MobileNetV2, una arquitectura que utiliza aprendizaje por transferencia.

La evaluación se realizó con un conjunto de prueba de 251 imágenes: 123 de Aedes aegypti, 120 de Aedes albopictus y 8 de Culex. Todo el módulo está en el notebook `02_CNN_Mosquitos.ipynb`: el análisis de las imágenes y la separación entre entrenamiento y prueba (secciones 1 a 3), el entrenamiento y la evaluación (sección 4) y un experimento con las fotos recortadas alrededor del mosquito (sección 5). La sección 4 agrega una CNN propia mejorada, también desde cero: fotos de 192 × 192, dos capas convolucionales por bloque, BatchNormalization y más aumento de datos. También elige la CNN propia solo con los datos de validación y evalúa una sola vez en prueba.

Las cifras de esta sección se calcularon en las secciones 4 y 5 del notebook y están en `models/mosquitos_resultados.json`, junto con las matrices de confusión. Antes, el F1 de la CNN propia (53.97 %) estaba escrito a mano y no se sabía de qué modelo era.

**Elección de la CNN propia (F1 macro en validación):**

| Modelo | F1 macro en validación |
|---|---:|
| CNN propia sugerida | 44.5 % |
| CNN propia sugerida con aumento de datos | 45.2 % |
| CNN propia v2 | 24.4 % |
| MobileNetV2 (comparación) | 86.5 % |

Se eligió la CNN propia con aumento de datos, la de mayor F1 en validación. La v2 se detuvo en la época 14: elegía la mejor época por la pérdida en validación, que saltaba mucho de una época a otra, y quedó con un modelo que en validación responde casi siempre la misma especie (su exactitud en validación se quedó entre 48 % y 52 %, que es la proporción de una sola especie).

**Resultados en prueba (evaluados una sola vez):**

| Modelo | F1 macro | F1 aegypti | F1 albopictus | F1 Culex |
|---|---:|---:|---:|---:|
| CNN propia (sugerida con aumento de datos) | 50.29 % | 0.00 | 0.67 | 0.84 |
| MobileNetV2 | 88.30 % | 0.87 | 0.89 | 0.89 |

**Matrices de confusión en prueba** (filas: especie correcta; columnas: lo que respondió la red, en el orden aegypti / albopictus / Culex):

| Especie correcta | CNN propia | MobileNetV2 |
|---|---|---|
| Aedes aegypti (123) | 0 / 120 / 3 | 103 / 18 / 2 |
| Aedes albopictus (120) | 0 / 120 / 0 | 10 / 110 / 0 |
| Culex (8) | 0 / 0 / 8 | 0 / 0 / 8 |

El F1 combina la precisión y el recall de cada especie, y el F1 macro es el promedio simple de las tres, para que la clase con menos fotos (Culex) pese lo mismo que las demás.

En esta evaluación, MobileNetV2 obtuvo el mejor resultado, superando el objetivo establecido del 85 % de F1 macro. Por esta razón, se utiliza como modelo principal para la identificación de mosquitos dentro del sistema.

En esta evaluación, la CNN propia no reconocía Aedes aegypti, justamente la especie que transmite el dengue: la matriz de confusión muestra que respondió Aedes albopictus en 120 de las 123 fotos de aegypti. No es por falta de fotos de esa especie, que es la que más tiene. Las dos especies de Aedes se distinguen por el dibujo de escamas blancas del tórax, y en las fotos de Kaggle el mosquito ocupa una parte pequeña de una foto de unos 2464 × 3280 píxeles. Al reducirla a 128 o 192 píxeles ese detalle casi desaparece, y una red entrenada desde cero con unas 840 fotos no logra aprenderlo. Por ejemplo, la v2 llegó a 95 % de aciertos en entrenamiento, pero en validación no separa las dos especies: memoriza las fotos de entrenamiento en lugar de aprender el dibujo del tórax. MobileNetV2, que ya trae lo aprendido con millones de imágenes, sí las separa: acierta 103 de las 123 fotos de aegypti y 110 de las 120 de albopictus.

**CNN propia con recorte (sección 5 del notebook 02).** Para resolver ese problema, cada foto se recorta alrededor del mosquito antes de reducirla: se toma como color del fondo la mediana de la foto, los píxeles que se alejan mucho de ese color forman manchas, y se recorta un cuadrado alrededor de la mancha más grande que no toca el borde. El recorte no usa la especie y es igual para entrenamiento, validación y prueba; encontró el mosquito en las 1,235 fotos. Con los recortes se entrenó la misma red v2, desde cero, a 128 × 128, y la mejor época se eligió con el F1 macro en validación (no con la pérdida).

Las 22 fotos de Culex de IEEE DataPort no se pudieron volver a descargar, porque la página pide una suscripción de pago. Por eso el experimento usa las otras 1,235 fotos con la misma separación: la prueba tiene 247 fotos (123 aegypti, 120 albopictus y 4 Culex), y la validación queda sin Culex, así que su F1 se calcula con aegypti y albopictus. Todos los modelos se evaluaron con las mismas fotos.

| Modelo (mismas 247 fotos de prueba) | F1 macro | F1 aegypti | F1 albopictus | F1 Culex |
|---|---:|---:|---:|---:|
| **CNN propia con recorte** | **88.00 %** | 0.95 | 0.96 | 0.73 |
| MobileNetV2 | 85.33 % | 0.87 | 0.89 | 0.80 |
| CNN propia de la sección 4 | 46.46 % | 0.00 | 0.67 | 0.73 |

La CNN propia con recorte se eligió con validación (F1 de 0.956, en la época 38 de 40) y se evaluó una sola vez en prueba. **Supera la meta del 85 %** y separa bien las dos especies de Aedes: acierta 112 de 123 aegypti (8 las confunde con albopictus y 3 con Culex), 119 de 120 albopictus y las 4 Culex. Las 3 que llamó Culex son justo las 3 fotos de aegypti de Mendeley, que tienen el mismo fondo blanco que las de Culex: para reconocer Culex, la red se apoya en parte en el fondo. Hay que leer este resultado con sus límites: con solo 4 fotos de Culex, el F1 de esa especie (0.73) depende de 3 errores; el F1 en validación saltó mucho de una época a otra, así que podría cambiar con otra semilla; y la idea del recorte surgió de la matriz de confusión de la prueba, aunque la validación ya mostraba el mismo problema.

**MobileNetV2 con fotos de fondos variados (sección 6 del notebook 02).** Casi todas las fotos de Culex venían de Mendeley, sobre papel blanco, y casi todas las de aegypti y albopictus de Kaggle, sobre fondo azul, así que las redes aprendieron en parte el fondo: con fotos de celular, la MobileNetV2 original llamaba Culex a casi todo (a las 7 de aegypti y a 50 de las 61 de albopictus). Para corregirlo se agregaron dos fuentes con fondos variados:

- **Mosquito Alert 2023**, una de las fuentes del enunciado: fotos tomadas con el celular por personas, con la caja donde está el mosquito. Se usaron las 47 de aegypti (todas las que tiene) y 400 de albopictus y de Culex.
- **Mosquito-on-human-skin** (Ong y Ahmad, 2022): mosquitos de las tres especies posados sobre piel humana. Se usaron las 250 fotos originales de cada especie.

Cada fuente se separó en 70 % para entrenamiento, 15 % para validación y 15 % para prueba. Se entrenaron tres versiones, cada una elegida con validación, con la regla fijada antes de ver la prueba:

| Versión de MobileNetV2 | Kaggle y Mendeley (247) | Fotos de celular, Mosquito Alert (128) | Sobre piel (113) | *Aegypti* fuera del fondo azul |
|---|---:|---:|---:|---|
| Original | 85.33 % | 24.96 % | — | 0 de 7 en fotos de celular |
| + Mosquito Alert, solo la capa final (6.3) | 84.22 % | 70.05 % | — | 2 de 7 en fotos de celular |
| + ajuste fino de las últimas capas (6.8) | **87.46 %** | 59.43 % | 46.81 % | 0 de 7 en celular · 0 de 37 sobre piel |
| **+ fotos sobre piel (6.9, la de la aplicación)** | 84.07 % | **66.42 %** | **80.77 %** | **3 de 7 en celular · 30 de 37 sobre piel** |

La versión de la aplicación es la primera que reconoce Aedes aegypti fuera del fondo azul: 30 de 37 sobre piel y 3 de 7 en fotos de celular, y es la especie que transmite el dengue. El costo es que en las fotos de Kaggle y Mendeley baja a 84.07 %, justo por debajo de la meta (una de las 4 Culex de prueba pasa a error, y con tan pocas fotos eso pesa mucho en el F1 macro), y que Culex en fotos de celular baja de 57 a 48 aciertos de 60. La CNN propia con recorte no se usa en la aplicación porque el recorte depende de un fondo parejo como el de Kaggle.

### 5.2. Modelo RNN: pronóstico de casos de dengue

El segundo módulo utiliza redes neuronales recurrentes para pronosticar los casos de dengue de las próximas cuatro semanas a partir de datos históricos reales.

Se trabajó con el archivo `dengue_procesado.csv` (566 semanas, de febrero de 2014 a diciembre de 2024), generado en el EDA a partir de los casos de OPS/PAHO y el clima de NASA POWER. Las variables utilizadas son:

- Casos de dengue, en logaritmo.
- Precipitación.
- Temperatura máxima.
- Temperatura mínima.
- Humedad relativa.
- Semana del año, expresada con seno y coseno para que la semana 52 quede cerca de la semana 1.

El modelo utiliza una ventana de las 26 semanas anteriores para generar un pronóstico de las siguientes cuatro semanas.

La separación de los datos fue temporal, para no usar información del futuro: entrenamiento hasta 2021, validación con 2022 y prueba con 2023–2024 (las últimas 104 semanas).

Se compararon dos arquitecturas:

| Modelo | Capas | Parámetros |
|---|---|---:|
| SimpleRNN (base) | SimpleRNN de 32 → 4 salidas | 1,412 |
| LSTM (principal) | LSTM de 64 → LSTM de 32 → Dropout 0.2 → 4 salidas | 30,980 |

Ambas se entrenaron con parada anticipada (EarlyStopping) sobre el año de validación. Para evaluar el rendimiento se usaron dos referencias: la línea base estacional que pide el enunciado (los casos de la misma semana del año anterior) y, como control adicional, la persistencia (repetir los casos de la última semana conocida en las 4 semanas siguientes).

**Resultados obtenidos:**

| Modelo | RMSE | MAE |
|---|---:|---:|
| SimpleRNN | 260 | 175 |
| LSTM | 494 | 334 |
| Línea base estacional | 638 | 456 |
| Persistencia (control) | 216 | 141 |

En estas métricas, los valores más bajos representan un menor error de predicción. Los errores están en casos por semana.

Las dos redes superan a la línea base estacional, como pide el enunciado. Sin embargo, ninguna supera a la persistencia: copiar la última semana tiene menos error en promedio. La SimpleRNN solo le gana cuando se pronostica a 4 semanas de distancia (RMSE de 235 contra 279). Además, las dos redes subestiman los brotes: en el pico del 12 de noviembre de 2023 hubo 1,749 casos y la SimpleRNN pronosticó 1,227 a una semana de distancia.

La LSTM quedó peor que la SimpleRNN por dos razones:

- El año de validación (2022) no tuvo brote: como máximo hubo 316 casos en una semana. La parada anticipada detuvo a la LSTM antes de que aprendiera a seguir los brotes de 2023 y 2024.
- La LSTM tiene unas 22 veces más parámetros que la SimpleRNN (30,980 contra 1,412) y solo hay 381 ventanas de entrenamiento, por lo que necesita más ejemplos de brotes para aprovechar su memoria.

Por esta razón, se seleccionó SimpleRNN para su integración en la API y la aplicación. La LSTM se conserva en `models/rnn_lstm.h5`. Se comprobó que la API reproduce los resultados del notebook: con las 101 ventanas de prueba da el mismo RMSE (260.2) y MAE (174.9).

### 5.3. Modelo ANN: recomendación de acciones preventivas

El tercer módulo utiliza una red neuronal artificial (ANN) para recomendar acciones relacionadas con la prevención y el control del dengue.

Se utilizó el archivo `vectoria_acciones.csv`, compuesto por 2,060 registros correspondientes a 412 semanas y cinco escenarios por semana.

Este archivo es mixto. Los casos, el clima y el umbral endémico son reales. En cambio, los casos pronosticados (los casos reales con un error de alrededor del 20 %) y la detección de Aedes aegypti se simularon en cinco escenarios por semana, y la recomendación se generó con estas reglas, en orden de prioridad:

1. Casos pronosticados mayores que el doble del umbral endémico → alerta sanitaria.
2. Casos pronosticados mayores que el umbral y Aedes aegypti detectado → fumigación focalizada.
3. Lluvia de 14 días mayor que 80 mm y temperatura máxima mayor que 27 °C → eliminación de criaderos.
4. En otro caso → monitoreo normal.

Un 3 % de las etiquetas tiene ruido agregado a propósito, para que el modelo no alcance un 100 % trivial.

El modelo considera variables como:

- Casos pronosticados para las siguientes cuatro semanas.
- Umbral endémico de casos.
- Precipitación acumulada de 14 días.
- Temperatura máxima y mínima.
- Humedad.
- Detección de Aedes aegypti.

El sistema contempla cuatro posibles recomendaciones:

1. Alerta sanitaria.
2. Eliminación de criaderos.
3. Fumigación focalizada.
4. Monitoreo normal.

La arquitectura utilizada contiene una capa de entrada de siete variables, capas ocultas de 32 y 16 neuronas con Dropout de 0.2, y una capa de salida con cuatro categorías. Se entrenó con pesos por clase y parada anticipada.

**Resultados obtenidos:**

La exactitud promedio registrada fue de 90.8 % en cinco ejecuciones con diferentes semillas.

| Acción | Precisión | Recall |
|---|---:|---:|
| Alerta sanitaria | 0.95 | 0.94 |
| Eliminación de criaderos | 0.82 | 0.78 |
| Fumigación focalizada | 0.86 | 0.87 |
| Monitoreo normal | 0.93 | 0.94 |

Los resultados muestran un buen desempeño general. Sin embargo, la categoría de eliminación de criaderos presenta valores inferiores al objetivo del 85 %: se confunde con monitoreo normal, porque solo las separan los umbrales de lluvia y temperatura. Representa una oportunidad de mejora.

Como las etiquetas salen de reglas, esta exactitud mide qué tan bien la ANN aprende esas reglas a partir de los datos.

La división de los datos se realizó agrupando por semana, con el propósito de evitar que registros de una misma semana quedaran repartidos entre entrenamiento y prueba.

### 5.4. Bonus: pronóstico por provincia (datos sintéticos)

El enunciado propone como bonus pronosticar los casos por provincia con `vectoria_dengue_semanal.csv`. Ese archivo es **sintético**: lo generó el paquete del curso y no son cifras oficiales. Por eso, este resultado muestra que el método funciona, pero no que la red pronostique el dengue real de cada provincia. Está en el notebook `05_Bonus_Provincias.ipynb`.

- **Datos:** `dengue_provincias_limpio.csv`, generado en el EDA (7 provincias × 626 semanas, de 2014 a 2025, sin valores vacíos).
- **Modelo:** una sola SimpleRNN para las 7 provincias, con la misma arquitectura de la sección 5.2 (SimpleRNN de 32 → 4 salidas). Además de los casos y el clima, recibe la provincia como variable *one-hot* (una columna por provincia). Con validación se comprobó que esto casi no cambia el resultado (40.5 contra 40.7 de RMSE); se dejó por tener el menor promedio.
- **Separación temporal:** entrenamiento de 2014 a 2022, validación con 2023 (año con brote) y prueba con 2024–2025 (707 ventanas, 101 por provincia).

| Modelo | RMSE | MAE |
|---|---:|---:|
| SimpleRNN por provincia | 27.8 | 15.4 |
| Línea base estacional | 49.0 | 30.5 |
| Persistencia (control) | 35.3 | 19.9 |

Los errores son más pequeños que los del modelo nacional porque cada provincia tiene menos casos por semana. La red supera a la línea base estacional en las 7 provincias (entre 29 % y 52 % menos RMSE) y, a diferencia del modelo nacional, también a la persistencia, en las 4 semanas pronosticadas y en las 7 provincias. Con 3 semillas, el RMSE va de 27.2 a 27.8.

Hay tres razones probables para la diferencia con el modelo nacional: hay más datos (3,087 ventanas de entrenamiento contra 381), el año de validación sí tuvo brote, y los datos sintéticos siguen patrones más regulares que los reales. La API lo ofrece en `POST /predict/casos/provincia`, y la aplicación en la pestaña de pronóstico.

## 6. Integración del sistema

Los tres modelos se integraron en una solución que permite acceder a sus funcionalidades desde una misma aplicación.

### 6.1. API con FastAPI

Se desarrolló una API para facilitar la comunicación con los modelos y permitir realizar predicciones mediante solicitudes.

Entre sus principales rutas se encuentran:

| Endpoint | Función |
|---|---|
| `GET /` | En el navegador, muestra la página de inicio de la API. Para la aplicación y otros programas, devuelve el estado de los modelos en JSON. |
| `GET /estado` | Estado de los modelos en JSON. |
| `POST /predict/mosquito` | Clasificar una imagen de mosquito. |
| `POST /predict/casos` | Generar el pronóstico de casos de las 4 semanas siguientes a partir de 26 semanas (también responde en `/predict/rnn`, el nombre anterior). |
| `POST /predict/accion` | Recomendar una acción preventiva. |
| `POST /predict/casos/provincia` | Bonus: pronóstico de una provincia (datos sintéticos). |
| `GET /visual/nacional` | Gráfico PNG del pronóstico nacional contra el umbral endémico, con fecha de corte opcional. |
| `GET /visual/provincia/{provincia}` | Gráfico PNG del pronóstico de una provincia (bonus). |
| `GET /visual/provincias` | Panel PNG con las 7 provincias y un ranking según su umbral (bonus). |

La API valida los datos de entrada: por ejemplo, el pronóstico exige exactamente 26 semanas, la humedad debe estar entre 0 y 100 % y la provincia debe ser una de las 7.

La documentación interactiva está en `/docs` (Swagger UI), con los colores de VectorIA. Los endpoints están agrupados por módulo, y cada endpoint de predicción trae ejemplos listos con datos del proyecto, como el brote de octubre de 2023. Los gráficos de `/visual` se pueden probar desde ahí mismo y aparecen en la misma página. En el navegador, la dirección principal de la API (`http://127.0.0.1:8000`) muestra una página de inicio con el estado de cada modelo y los gráficos generados en el momento.

### 6.2. Aplicación con Streamlit

Se desarrolló una interfaz gráfica que permite utilizar las funcionalidades de VectorIA sin necesidad de interactuar directamente con el código.

La aplicación cuenta con cinco pestañas:

1. **Resumen:** muestra el estado de la API y de los modelos, y los casos del último año.
2. **Identificación CNN:** permite cargar una imagen y obtener la especie, con la probabilidad de cada clase. Usa la MobileNetV2 reentrenada con fotos de celular y de mosquitos sobre piel (sección 5.1).
3. **Pronóstico RNN:** el usuario elige una semana; la SimpleRNN usa las 26 semanas que terminan ahí y pronostica las cuatro siguientes. El gráfico compara el pronóstico con el umbral endémico semanal (percentil 75 de esa misma semana en los 5 años anteriores) y, si se eligió una semana pasada, con los casos que de verdad hubo. Un selector cambia al bonus por provincia (con un aviso de datos sintéticos): se elige la provincia y la semana, y un botón compara las 7 provincias según qué tanto se acercan a su umbral.
4. **Prevención ANN:** usa ese pronóstico, el clima de la semana elegida, el umbral endémico calculado con los datos históricos y el resultado de la CNN para recomendar una acción preventiva con su motivo. Incluye un simulador "¿Qué pasaría si…?": con deslizadores de casos, lluvia, temperatura y detección de Aedes, se ve cómo cambiaría la recomendación.
5. **Mapa cantonal:** mapa de Costa Rica coloreado según los casos de cada cantón, en escala logarítmica para que se distingan los cantones con pocos casos. Incluye una animación de 2013 a 2018, una vista 3D en la que cada cantón se eleva según sus casos, el ranking de cantones y la evolución anual por provincia, con filtros por provincia y año. El mapa funciona sin internet, porque las fronteras de los cantones vienen dentro del proyecto; solo el mapa de fondo de la vista 3D necesita conexión.

## 7. Evaluación general de resultados

Los resultados obtenidos permiten identificar el desempeño de cada uno de los modelos:

- **CNN:** MobileNetV2 alcanzó un F1 macro de 88.30 %, superando el objetivo del 85 %. La CNN propia lo supera solo con las fotos recortadas alrededor del mosquito: 88.00 % (contra 85.33 % de MobileNetV2 en las mismas 247 fotos). Con fotos de celular con fondos variados (Mosquito Alert), la MobileNetV2 original bajaba a 24.96 %. La versión de la aplicación, reentrenada también con fotos de celular y de mosquitos sobre piel, reconoce aegypti fuera del fondo azul (30 de 37 sobre piel) y obtiene 80.77 % sobre piel, 66.42 % con fotos de celular y 84.07 % en Kaggle y Mendeley.
- **RNN:** SimpleRNN obtuvo un RMSE de 260 y un MAE de 175, el menor error entre las dos redes y muy por debajo de la línea base estacional (638). Aun así, no supera a la persistencia (216), salvo a 4 semanas de distancia.
- **ANN:** alcanzó una exactitud promedio de 90.8 %, aunque la categoría de eliminación de criaderos todavía necesita mejoras.
- **Bonus por provincia (datos sintéticos):** RMSE de 27.8 y MAE de 15.4, por debajo de la línea base estacional (49.0) y de la persistencia (35.3).

En conjunto, los resultados demuestran que los modelos pueden aportar herramientas de apoyo para la identificación de vectores, el análisis de tendencias y la recomendación de medidas preventivas.

### 7.1. Cumplimiento de lo que pide el enunciado

| Parte | Qué pide el enunciado | Resultado | ¿Cumple? |
|---|---|---|---|
| Módulo 1 (CNN) | CNN propia desde cero (obligatoria), F1 macro ≥ 85 % | Con la foto completa, 50.29 %. Con la foto recortada alrededor del mosquito (sección 5), 88.00 % en 247 fotos de prueba | Sí, con el recorte (sin las 22 fotos de IEEE) |
| Módulo 1 (CNN) | MobileNetV2 con transfer learning como comparación, F1 macro | 88.30 % (251 fotos). La versión de la aplicación, reentrenada con fotos de celular y sobre piel: 84.07 % en Kaggle y Mendeley, 66.42 % con fotos de celular y 80.77 % sobre piel | Sí, la original |
| Módulo 2 (RNN/LSTM) | SimpleRNN (base) y LSTM (principal); RMSE y MAE contra el baseline estacional | SimpleRNN 260 / 175 y LSTM 494 / 334, contra 638 / 456 | Sí: las dos superan al baseline |
| Módulo 3 (ANN) | Clasificador multiclase, separación por semana, precisión y recall ≥ 85 % | 6 de 8 métricas sobre 85 %; eliminación de criaderos con 0.82 / 0.78 | Parcial |
| Aplicación | Subir foto, elegir semana y ver el pronóstico contra el umbral endémico, recomendación con motivo, mapa de calor por cantón | Las cuatro funciones | Sí |
| Aplicación | Extra: deslizadores "¿qué pasaría si?" | Simulador en la pestaña de prevención | Sí |
| Aplicación | Aviso fijo: no diagnostica dengue en personas | En la parte superior de la aplicación | Sí |
| API | `/predict/mosquito`, `/predict/casos` y `/predict/accion` | Los tres endpoints, documentados en `/docs` con ejemplos, más gráficos PNG en `/visual` | Sí |
| Entregables | README, notebooks de EDA y entrenamiento, modelos, aplicación, API, informe y código en GitHub | Todos | Sí |
| Bonus | Predicción por provincia con `vectoria_dengue_semanal.csv` (sintético) | SimpleRNN por provincia: RMSE 27.8 / MAE 15.4, contra 49.0 / 30.5 del baseline (notebook 05, API y app) | Sí, con datos sintéticos |
| Bonus | Identificar larvas en criaderos con MLMI-2024 | — | No se hizo |
| Extra | Docker | — | No se hizo |

## 8. Limitaciones del proyecto

Durante el desarrollo se identificaron algunas limitaciones que deben considerarse al interpretar los resultados:

- Con la foto completa, la CNN propia obtuvo un F1 macro de 50.29 % y no reconoce Aedes aegypti: lo confunde con Aedes albopictus. Solo supera la meta con las fotos recortadas alrededor del mosquito (88.00 %), un resultado medido sin las 22 fotos de Culex de IEEE DataPort (solo 4 de Culex en prueba) y con un recorte que depende de un fondo parejo como el de Kaggle.
- La CNN siempre responde una de las tres especies: no tiene una clase para imágenes que no son mosquitos, así que una foto cualquiera también recibe una especie, incluso con una confianza alta.
- La categoría de eliminación de criaderos del modelo ANN tiene precisión de 82 % y recall de 78 %.
- La ANN se entrenó con escenarios simulados y etiquetas generadas con reglas, no con decisiones reales de las autoridades sanitarias.
- El año de validación del Módulo 2 (2022) no tuvo brote. Con un año de validación que incluya un brote, la comparación entre SimpleRNN y LSTM podría cambiar.
- Ninguna de las dos redes del Módulo 2 supera a la persistencia (repetir la última semana), y las dos subestiman los brotes. Como la ANN decide las alertas con este pronóstico, un pronóstico bajo puede hacer que una alerta no se active.
- El conjunto de prueba de la CNN tiene solo 8 fotos de Culex, así que el F1 de esa clase es poco confiable: un solo error lo mueve mucho. Además, cada especie viene sobre todo de una fuente distinta (Aedes aegypti y Aedes albopictus de Kaggle, Culex de IEEE DataPort y Mendeley), por lo que la red podría estar aprendiendo el estilo o el fondo de cada fuente y no solo el mosquito. Hay evidencia de esto: las 3 fotos de prueba de Aedes aegypti de Mendeley tienen el mismo fondo blanco que las de Culex, y MobileNetV2 clasificó 2 de ellas como Culex; la CNN propia con recorte, las 3. Son sus únicos errores de aegypti a Culex. Con las fotos de celular de Mosquito Alert se vio todavía más claro: la MobileNetV2 original llamó Culex a casi todas (a las 7 de aegypti y a 50 de las 61 de albopictus). Reentrenarla con fotos de Mosquito Alert y de mosquitos sobre piel lo redujo mucho: con fotos de celular, albopictus pasó de 3 a 53 aciertos de 61 y aegypti de 0 a 3 de 7, y sobre piel reconoce 30 de 37 aegypti. Aun así, aegypti en fotos de celular sigue siendo lo más difícil, porque Mosquito Alert solo tiene 47 fotos de esa especie.
- La lluvia de NASA POWER se estima con un modelo (MERRA-2), no con estaciones. La semana del 26 de mayo de 2024 marca 625 mm, la más lluviosa de toda la serie, pero ni ERA5 (entre 14 y 36 mm ese día en las mismas ciudades) ni las estaciones del IMN registran esa lluvia, por lo que se considera un error de la fuente. No se eliminó, para no cambiar los archivos que usan los modelos. Su efecto en el pronóstico es pequeño: cambiando esa semana por el valor de ERA5, el RMSE de la SimpleRNN pasa de 260.2 a 255.4. También afecta la lluvia de 14 días de `vectoria_acciones.csv` en las semanas del 26 de mayo y el 2 de junio de 2024, aunque sus etiquetas no cambian.
- El archivo `vectoria_dengue_semanal.csv` es sintético y no representa cifras oficiales. Solo se usa en el EDA y en el bonus por provincia, así que los buenos resultados de ese modelo muestran que el método funciona, pero no que pronostique el dengue real de cada provincia. Sus resultados no se mezclan con los del Módulo 2.
- La información cantonal disponible corresponde al período 2013–2018 y sus totales no coinciden con los datos nacionales. Por ello, la pestaña cantonal debe utilizarse para comparaciones entre cantones dentro de ese conjunto, no para representar directamente los totales nacionales.
- En la aplicación, la precipitación de 14 días se aproxima sumando la lluvia de la semana elegida y la anterior. La temperatura y la humedad se toman de la semana elegida.
- La aplicación calcula el umbral endémico con los datos históricos (percentil 75 de los casos de esas mismas 4 semanas en los 5 años anteriores), con la misma fórmula de los datos de entrenamiento de la ANN. No es un umbral oficial del Ministerio de Salud, y puede ajustarse manualmente.
- Las imágenes de entrenamiento no se incluyen en el repositorio debido a su tamaño.

Estas limitaciones deben tenerse presentes antes de utilizar el sistema en situaciones reales.

## 9. Recomendaciones y mejoras futuras

### 9.1. Recomendaciones de uso

A partir de los resultados, se recomienda a quienes usen VectorIA en la vigilancia del dengue:

- **Usarlo como apoyo, junto con el canal endémico oficial.** Si el Ministerio de Salud tiene un umbral oficial para esas semanas, conviene escribirlo en la aplicación en lugar del calculado.
- **Tomar el pronóstico en época de brote como un mínimo.** Las redes subestiman los brotes, así que un pronóstico cercano al umbral ya justifica reforzar la vigilancia.
- **Aprovechar el retraso del clima como alerta temprana.** La temperatura mínima se relaciona con los casos de 6 a 8 semanas después, lo que da tiempo para preparar acciones.
- **Priorizar los cantones que concentran los casos.** En 2013–2018, 10 de 72 cantones sumaron el 49 % de los casos, y Guanacaste, Puntarenas y Limón el 68 %.
- **Usar la identificación de mosquitos solo con fotos claras de un mosquito.** La CNN siempre responde una especie y no reemplaza la identificación de un especialista.
- **Revisar los datos de clima extremos antes de usarlos,** como la semana del 26 de mayo de 2024, que resultó ser un error de la fuente.

### 9.2. Mejoras futuras

Para continuar fortaleciendo VectorIA, se proponen las siguientes mejoras:

- Conseguir más fotos de Aedes aegypti con fondos variados: Mosquito Alert, la única fuente con fondos variados, tiene solo 47.
- Probar la CNN propia con recorte en fotos de campo, con fondos variados, y repetirla con varias semillas; si se mantiene, llevar el recorte a la API para usarla en la aplicación.
- Agregar a la CNN una clase para imágenes que no son mosquitos.
- Mejorar la detección de la categoría de eliminación de criaderos en el modelo ANN.
- Comparar SimpleRNN y LSTM con un período de validación que incluya un brote.
- Conseguir datos reales semanales por provincia (por ejemplo, de los boletines del Ministerio de Salud) y volver a entrenar el modelo del bonus con ellos; el notebook 05 ya está listo para eso.
- Contrastar la lluvia de NASA POWER con las estaciones del IMN.
- Incorporar datos epidemiológicos y climáticos oficiales actualizados.
- Automatizar la obtención de umbrales endémicos.
- Mejorar la información geográfica por cantón.
- Evaluar los pronósticos con nuevos períodos de datos reales.
- Incorporar mecanismos de actualización y monitoreo del rendimiento de los modelos.
- Empaquetar el sistema con Docker para facilitar su instalación.

## 10. Conclusiones

VectorIA integra tres modelos de inteligencia artificial para apoyar diferentes etapas de la vigilancia del dengue: identificación de mosquitos, pronóstico de casos y recomendación de acciones preventivas.

Los resultados experimentales muestran que MobileNetV2 alcanzó un F1 macro de 88.30 %, que la CNN propia llegó a 88.00 % al recortar cada foto alrededor del mosquito, que reentrenar MobileNetV2 con fotos de celular y de mosquitos sobre piel le enseñó a reconocer las especies fuera del fondo de laboratorio (aegypti sobre piel: de 0 a 30 de 37), SimpleRNN obtuvo mejores métricas de error que LSTM y la línea base estacional, y el modelo ANN logró una exactitud promedio de 90.8 %.

En el pronóstico, la red más simple superó a la más compleja: con pocos datos y sin brotes en el año de validación, la LSTM no logró aprovechar su memoria. Ninguna de las dos superó todavía a la persistencia, lo que deja claro el siguiente paso: más datos y un año de validación con brote. El bonus por provincia apunta en esa dirección: con más ventanas y un año de validación con brote, la misma SimpleRNN sí superó a la persistencia, aunque con datos sintéticos.

La integración mediante FastAPI y Streamlit permite utilizar las funcionalidades desde una aplicación centralizada y facilita la interacción con los modelos.

Aunque existen aspectos que pueden mejorarse, especialmente probar la CNN propia con recorte en fotos de campo, la recomendación de eliminación de criaderos y la calidad y actualización de los datos, el proyecto demuestra cómo la inteligencia artificial puede utilizarse como herramienta de apoyo para la vigilancia y prevención del dengue.

**VectorIA no sustituye el diagnóstico médico ni las decisiones de las autoridades sanitarias. Su propósito es brindar información de apoyo para el análisis y la prevención.**

## 11. Equipo de trabajo

- **Gabriel:** arquitectura y backend.
- **Jeremy:** Módulo 2 (SimpleRNN y LSTM) e integración del informe.
- **Jeriel:** modelo ANN, integración de la API y documentación.
- **Yanely:** análisis exploratorio de datos (EDA).

## 12. Fuentes de información

- **OPS/PAHO, mediante OpenDengue:** casos semanales de dengue en Costa Rica.
- **NASA POWER (MERRA-2):** clima semanal (lluvia, temperatura y humedad).
- **ERA5 (Copernicus), consultado mediante Open-Meteo:** comparación de la lluvia de mayo de 2024.
- **Instituto Meteorológico Nacional (IMN):** boletín meteorológico de mayo de 2024.
- **IEEE DataPort:** [Image Dataset of Aedes and Culex Mosquito Species](https://ieee-dataport.org/open-access/image-dataset-aedes-and-culex-mosquito-species).
- **Mendeley Data:** [Dataset of Vector Mosquito Images](https://data.mendeley.com/datasets/88s6fvgg2p).
- **Kaggle:** [Aedes Mosquitos Dataset](https://www.kaggle.com/datasets/pradeepisawasan/aedes-mosquitos) (Aedes aegypti y Aedes albopictus).
- **Mosquito Alert:** [AI Mosquito Alert Challenge Dataset 2023](https://zenodo.org/records/15063886), Mosquito Alert Community (www.mosquitoalert.com), licencia CC BY-NC-SA 4.0.
- **Mosquito-on-human-skin:** Ong, S.-Q. y Ahmad, H. (2022), *An annotated image dataset for training mosquito species recognition system on human skin*, Scientific Data; [Mendeley Data](https://data.mendeley.com/datasets/zw4p9kj6nt/2), licencia CC BY 4.0.
- **geoBoundaries:** fronteras de los cantones de Costa Rica (dominio público, CC0), usadas en el mapa de la aplicación.

## 13. Uso de inteligencia artificial

El detalle de cómo cada integrante utilizó herramientas de inteligencia artificial durante el proyecto está en `USO_DE_IA.md`.
