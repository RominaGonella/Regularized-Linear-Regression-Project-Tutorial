# Resumen del proceso

* El objetivo de la tarea consiste en descubrir si hay una relación entre recursos para la salud y datos sociodemográficos. Se debe elegir una variable target (relacionada a la salud) y utilizar LASSO para seleccionar las variables explicativas más importantes.
También hay que encontrar los coeficientes seleccionados para el modelo final.
* En el primer paso se cargan los datos y las librerías, se hacen correcciones mínimas sobre los datos y se guarda base inicial.
* Luego en el segundo paso se realiza análisis exploratorio sobre la muestra de training, se decide elegir como variable target la **prevalencia de la obesidad**. Se descartan las otras variables que podrían haber sido elegidas como target, se mantienen las que refieren a indicadores sociodemográficos y sanitarios.
* En el paso 3 se aplica el método LASSO para estimar modelos, primero se estima con el valor de alpha por defecto, luego se aplica cross validation para optimizar el valor de alpha y por último se estima un modelo con el valor de alpha óptimo. En todos los casos se setea `normalize = True` para normalizar previamente el dataset.
* En el paso 4 se guarda la base de entrenamiento final y el modelo con alpha óptimo.