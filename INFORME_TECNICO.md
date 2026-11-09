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
| `vectoria_dengue_semanal.csv` | Sintético | Solo se analiza en el EDA (versión por provincia). Ningún modelo lo usa. |
| Imágenes de mosquitos | IEEE DataPort, Mendeley Data y Kaggle | Entrenamiento y evaluación de la CNN (Módulo 1). |

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

La evaluación se realizó con un conjunto de prueba de 251 imágenes: 123 de Aedes aegypti, 120 de Aedes albopictus y 8 de Culex. Todo el módulo está en el notebook `02_CNN_Mosquitos.ipynb`: el análisis de las imágenes y la separación entre entrenamiento y prueba (secciones 1 a 3), y el entrenamiento y la evaluación (sección 4). La sección 4 agrega una CNN propia mejorada, también desde cero: fotos de 192 × 192, dos capas convolucionales por bloque, BatchNormalization y más aumento de datos. También elige la CNN propia solo con los datos de validación y evalúa una sola vez en prueba.

Las cifras de esta sección corresponden a la evaluación registrada antes de agregar la sección 4. Se actualizarán con lo que dé esa sección al correrla con las imágenes.

**Resultados obtenidos:**

| Modelo | F1 macro | F1 aegypti | F1 albopictus | F1 Culex |
|---|---:|---:|---:|---:|
| CNN propia | 53.97 % | 0.14 | 0.64 | 0.84 |
| MobileNetV2 | 88.30 % | 0.87 | 0.89 | 0.89 |

El F1 combina la precisión y el recall de cada especie, y el F1 macro es el promedio simple de las tres, para que la clase con menos fotos (Culex) pese lo mismo que las demás.

MobileNetV2 obtuvo el mejor resultado, superando el objetivo establecido del 85 % de F1 macro. Por esta razón, se utiliza como modelo principal para la identificación de mosquitos dentro del sistema.

La CNN propia se conserva como parte de la comparación experimental. Su punto más débil es Aedes aegypti (F1 de 0.14), justamente la especie que transmite el dengue. No es por falta de fotos de esa especie, que es la que más tiene (492 para entrenar). Lo más probable es que la confunda con Aedes albopictus, que se le parece mucho, pero para confirmarlo hace falta la matriz de confusión de la CNN propia en prueba, que no quedó guardada. MobileNetV2, que ya trae lo aprendido con millones de imágenes, sí logra separar las dos especies.

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

## 6. Integración del sistema

Los tres modelos se integraron en una solución que permite acceder a sus funcionalidades desde una misma aplicación.

### 6.1. API con FastAPI

Se desarrolló una API para facilitar la comunicación con los modelos y permitir realizar predicciones mediante solicitudes.

Entre sus principales rutas se encuentran:

| Endpoint | Función |
|---|---|
| `GET /` | Verificar el funcionamiento de la API. |
| `POST /predict/mosquito` | Clasificar una imagen de mosquito. |
| `POST /predict/casos` | Generar el pronóstico de casos de las 4 semanas siguientes a partir de 26 semanas (también responde en `/predict/rnn`, el nombre anterior). |
| `POST /predict/accion` | Recomendar una acción preventiva. |

La API valida los datos de entrada: por ejemplo, el pronóstico exige exactamente 26 semanas y la humedad debe estar entre 0 y 100 %.

La documentación interactiva de la API se encuentra disponible mediante Swagger UI en `/docs`.

### 6.2. Aplicación con Streamlit

Se desarrolló una interfaz gráfica que permite utilizar las funcionalidades de VectorIA sin necesidad de interactuar directamente con el código.

La aplicación cuenta con cinco pestañas:

1. **Resumen:** muestra el estado de la API y de los modelos, y los casos del último año.
2. **Identificación CNN:** permite cargar una imagen y obtener la especie, con la probabilidad de cada clase.
3. **Pronóstico RNN:** el usuario elige una semana; la SimpleRNN usa las 26 semanas que terminan ahí y pronostica las cuatro siguientes. El gráfico compara el pronóstico con el umbral endémico semanal (percentil 75 de esa misma semana en los 5 años anteriores) y, si se eligió una semana pasada, con los casos que de verdad hubo.
4. **Prevención ANN:** usa ese pronóstico, el clima de la semana elegida, el umbral endémico calculado con los datos históricos y el resultado de la CNN para recomendar una acción preventiva con su motivo. Incluye un simulador "¿Qué pasaría si…?": con deslizadores de casos, lluvia, temperatura y detección de Aedes, se ve cómo cambiaría la recomendación.
5. **Mapa cantonal:** mapa de Costa Rica coloreado según los casos de cada cantón, en escala logarítmica para que se distingan los cantones con pocos casos. Incluye una animación de 2013 a 2018, una vista 3D en la que cada cantón se eleva según sus casos, el ranking de cantones y la evolución anual por provincia, con filtros por provincia y año. El mapa funciona sin internet, porque las fronteras de los cantones vienen dentro del proyecto; solo el mapa de fondo de la vista 3D necesita conexión.

## 7. Evaluación general de resultados

Los resultados obtenidos permiten identificar el desempeño de cada uno de los modelos:

- **CNN:** MobileNetV2 alcanzó un F1 macro de 88.30 %, superando el objetivo del 85 %.
- **RNN:** SimpleRNN obtuvo un RMSE de 260 y un MAE de 175, el menor error entre las dos redes y muy por debajo de la línea base estacional (638). Aun así, no supera a la persistencia (216), salvo a 4 semanas de distancia.
- **ANN:** alcanzó una exactitud promedio de 90.8 %, aunque la categoría de eliminación de criaderos todavía necesita mejoras.

En conjunto, los resultados demuestran que los modelos pueden aportar herramientas de apoyo para la identificación de vectores, el análisis de tendencias y la recomendación de medidas preventivas.

### 7.1. Cumplimiento de lo que pide el enunciado

| Parte | Qué pide el enunciado | Resultado | ¿Cumple? |
|---|---|---|---|
| Módulo 1 (CNN) | CNN propia desde cero (obligatoria), F1 macro ≥ 85 % | 53.97 % | No. Se agregó una CNN propia mejorada (sección 4 del notebook 02), pendiente de correr con las imágenes |
| Módulo 1 (CNN) | MobileNetV2 con transfer learning como comparación, F1 macro | 88.30 % | Sí |
| Módulo 2 (RNN/LSTM) | SimpleRNN (base) y LSTM (principal); RMSE y MAE contra el baseline estacional | SimpleRNN 260 / 175 y LSTM 494 / 334, contra 638 / 456 | Sí: las dos superan al baseline |
| Módulo 3 (ANN) | Clasificador multiclase, separación por semana, precisión y recall ≥ 85 % | 6 de 8 métricas sobre 85 %; eliminación de criaderos con 0.82 / 0.78 | Parcial |
| Aplicación | Subir foto, elegir semana y ver el pronóstico contra el umbral endémico, recomendación con motivo, mapa de calor por cantón | Las cuatro funciones | Sí |
| Aplicación | Extra: deslizadores "¿qué pasaría si?" | Simulador en la pestaña de prevención | Sí |
| Aplicación | Aviso fijo: no diagnostica dengue en personas | En la parte superior de la aplicación | Sí |
| API | `/predict/mosquito`, `/predict/casos` y `/predict/accion` | Los tres endpoints, documentados en `/docs` | Sí |
| Entregables | README, notebooks de EDA y entrenamiento, modelos, aplicación, API, informe y código en GitHub | Todos | Sí |
| Extra | Docker | — | No se hizo |

## 8. Limitaciones del proyecto

Durante el desarrollo se identificaron algunas limitaciones que deben considerarse al interpretar los resultados:

- La CNN propia obtuvo un F1 macro de 53.97 %, por lo que no alcanza el objetivo establecido, y casi no reconoce Aedes aegypti (F1 de 0.14). MobileNetV2 presenta un mejor desempeño.
- La CNN siempre responde una de las tres especies: no tiene una clase para imágenes que no son mosquitos, así que una foto cualquiera también recibe una especie, incluso con una confianza alta.
- La categoría de eliminación de criaderos del modelo ANN tiene precisión de 82 % y recall de 78 %.
- La ANN se entrenó con escenarios simulados y etiquetas generadas con reglas, no con decisiones reales de las autoridades sanitarias.
- El año de validación del Módulo 2 (2022) no tuvo brote. Con un año de validación que incluya un brote, la comparación entre SimpleRNN y LSTM podría cambiar.
- Ninguna de las dos redes del Módulo 2 supera a la persistencia (repetir la última semana), y las dos subestiman los brotes. Como la ANN decide las alertas con este pronóstico, un pronóstico bajo puede hacer que una alerta no se active.
- El conjunto de prueba de la CNN tiene solo 8 fotos de Culex, así que el F1 de esa clase es poco confiable: un solo error lo mueve mucho. Además, cada especie viene sobre todo de una fuente distinta (Aedes aegypti y Aedes albopictus de Kaggle, Culex de IEEE DataPort y Mendeley), por lo que la red podría estar aprendiendo el estilo o el fondo de cada fuente y no solo el mosquito.
- La lluvia de NASA POWER se estima con un modelo (MERRA-2), no con estaciones. La semana del 26 de mayo de 2024 marca 625 mm, la más lluviosa de toda la serie, pero ni ERA5 (entre 14 y 36 mm ese día en las mismas ciudades) ni las estaciones del IMN registran esa lluvia, por lo que se considera un error de la fuente. No se eliminó, para no cambiar los archivos que usan los modelos. Su efecto en el pronóstico es pequeño: cambiando esa semana por el valor de ERA5, el RMSE de la SimpleRNN pasa de 260.2 a 255.4. También afecta la lluvia de 14 días de `vectoria_acciones.csv` en las semanas del 26 de mayo y el 2 de junio de 2024, aunque sus etiquetas no cambian.
- El archivo `vectoria_dengue_semanal.csv` es sintético y no representa cifras oficiales. Solo se usa en el EDA; ningún modelo se entrenó con él.
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

- Ampliar y mejorar el conjunto de imágenes de mosquitos, por ejemplo con Mosquito Alert, sobre todo para Culex.
- Optimizar la CNN propia para aumentar su capacidad de clasificación. La sección 4 del notebook 02 ya incluye una versión mejorada.
- Agregar a la CNN una clase para imágenes que no son mosquitos.
- Mejorar la detección de la categoría de eliminación de criaderos en el modelo ANN.
- Comparar SimpleRNN y LSTM con un período de validación que incluya un brote.
- Contrastar la lluvia de NASA POWER con las estaciones del IMN.
- Incorporar datos epidemiológicos y climáticos oficiales actualizados.
- Automatizar la obtención de umbrales endémicos.
- Mejorar la información geográfica por cantón.
- Evaluar los pronósticos con nuevos períodos de datos reales.
- Incorporar mecanismos de actualización y monitoreo del rendimiento de los modelos.
- Empaquetar el sistema con Docker para facilitar su instalación.

## 10. Conclusiones

VectorIA integra tres modelos de inteligencia artificial para apoyar diferentes etapas de la vigilancia del dengue: identificación de mosquitos, pronóstico de casos y recomendación de acciones preventivas.

Los resultados experimentales muestran que MobileNetV2 alcanzó un F1 macro de 88.30 %, SimpleRNN obtuvo mejores métricas de error que LSTM y la línea base estacional, y el modelo ANN logró una exactitud promedio de 90.8 %.

En el pronóstico, la red más simple superó a la más compleja: con pocos datos y sin brotes en el año de validación, la LSTM no logró aprovechar su memoria. Ninguna de las dos superó todavía a la persistencia, lo que deja claro el siguiente paso: más datos y un año de validación con brote.

La integración mediante FastAPI y Streamlit permite utilizar las funcionalidades desde una aplicación centralizada y facilita la interacción con los modelos.

Aunque existen aspectos que pueden mejorarse, especialmente en la clasificación con la CNN propia, la recomendación de eliminación de criaderos y la calidad y actualización de los datos, el proyecto demuestra cómo la inteligencia artificial puede utilizarse como herramienta de apoyo para la vigilancia y prevención del dengue.

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
- **Kaggle:** conjunto de imágenes de mosquitos Aedes.
- **geoBoundaries:** fronteras de los cantones de Costa Rica (dominio público, CC0), usadas en el mapa de la aplicación.

## 13. Uso de inteligencia artificial

El detalle de cómo cada integrante utilizó herramientas de inteligencia artificial durante el proyecto está en `USO_DE_IA.md`.
