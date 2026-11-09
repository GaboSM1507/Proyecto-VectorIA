Jeriel, bueno yo para este proyecto no tuve mi computadora, tuve que trabajar con otra computadora, asi que no tenia el tiempo como si fuera usar mi compu, entonces busqué optimizar unos procesos como la creacion de la estructura del proyecto para no crear las carpetas de forma manuel, busqué como lo podia hacer mas rapido y me sugirió crar un archivo .py con toda la estructura que debia usar y ejecutarla, tambien para realizar commits vi que no  era necesario usar la terminal, sino que desde el mismo el git podia sunir el archivo que habia trabajado.
Al revisar ciertos archivos pude ver que el notebook de ann tenia algunas cosas mal, asi que fui corrigiendo de acuerdo a la materia, pero si llegué a cierto punto en que no recordaba para que sirvian algunas cosas y decidí preguntar.
Un promp de ejemplo: "Debo de crear esta estructura para mi proyecto, como podria hacerlo mas rapido sin tener que ir una a una, ya que no tengo mucho tiempo"




## Jeremy

**Herramienta usada:** Claude (Claude Code, de Anthropic), desde la aplicación de escritorio en mi computadora.

**Para qué la usé:**


- **EDA (`01_EDA.ipynb`):** revisó el notebook, corrigió textos y gráficos, y comprobó que los archivos que usan los modelos no cambiaran.
- **Lluvia del 26 de mayo de 2024:** investigó por qué esa semana marcaba 625 mm y encontró que es un error de la fuente (NASA POWER), comparándola con ERA5 y con el boletín del IMN.
- **Documentación:** redactó la sección del Módulo 2 del README, corrigió el informe técnico y preparó el PDF sobre la lluvia para la exposición.
- **App:** revisó la API y la app,ayudo a corregir bugs de la API 

**Ejemplos de lo que le pedí:**

- "Abre PyCharm y revisa todo el LSTM y que todo esté bien; revísalo muy a fondo, si hay cosas que no concuerdan haz una lista y me dices porque no esta bien y lo discutimos y lo cambiamos ."
- "haz una prueba para comprobar que todo está bien y funciona."
- Sobre la lluvia del 26 de mayo de 2024: "Investiga y dame la respuesta a eso."

**Qué hice yo:** le di las instrucciones, revisé los resultados y le pregunté lo que no entendía de cada parte, para poder explicarlo en la exposición.

## Gabriel

**Herramienta usada:** ChatGPT (OpenAI).

**¿Para qué la usé:**

Durante el desarrollo de VectorIA utilicé ChatGPT como herramienta de apoyo para comprender, desarrollar y corregir el módulo de clasificación de mosquitos mediante redes neuronales convolucionales (CNN).

Desarrollo de la CNN: me ayudó a estructurar el código en Python y TensorFlow/Keras para crear una CNN propia desde cero y compararla con MobileNetV2 mediante transferencia de aprendizaje.

Preparación de datos: me orientó para organizar las imágenes de mosquitos, verificar las rutas de los archivos, separar los conjuntos de entrenamiento, validación y prueba, y comprender cómo utilizar pesos por clase debido al desbalance del dataset.

Entrenamiento y evaluación: me ayudó a interpretar los resultados obtenidos, las matrices de confusión y métricas como precisión, recall y F1-score macro. También me permitió comprender por qué algunos modelos presentaron sobreajuste.

Corrección de errores: utilicé ChatGPT para resolver dudas y problemas relacionados con PyCharm, TensorFlow, la ejecución de notebooks y la integración del modelo con FastAPI.

Git y GitHub: me orientó en el uso de comandos para guardar mis cambios, descargar las actualizaciones de mis compañeros y subir los modelos entrenados al repositorio compartido sin afectar el trabajo del equipo.

Documentación: me ayudó a interpretar los resultados y redactar las conclusiones del entrenamiento para el notebook, además de preparar material de estudio para la exposición.

**Ejemplos de lo que le pregunté:**

"Ayúdame a crear y entrenar una CNN para clasificar tres especies de mosquitos utilizando TensorFlow y Keras."

"Explícame qué significa el F1-score macro y cómo interpretar los resultados de mis modelos."

"¿Por qué mi CNN alcanza una exactitud alta en entrenamiento, pero obtiene resultados bajos en validación?"

"¿Cómo puedo descargar las actualizaciones de mis compañeros desde GitHub sin perder mis archivos?"

"Ayúdame a interpretar las matrices de confusión y redactar las conclusiones del entrenamiento."

**¿Qué hice yo?**

Me encargué de ejecutar el código y los entrenamientos desde PyCharm, verificar el funcionamiento de los modelos, revisar los errores, analizar los resultados obtenidos y realizar las pruebas correspondientes.