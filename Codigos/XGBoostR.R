rm(list = ls())
# Cargar las librerías necesarias usando pacman para verificar instalación
pacman::p_load(
  "tidyverse",
  "readxl", 
  "openxlsx",
  "gbm",
  "caret",
  "data.table",
  "doParallel",
  "readr",
  "ggplot2"
)

# Cargar el archivo CSV
data <- read.csv("E:/TESISXGBoost/Tesis/NotasXGBoost/DataFrames/train.csv")

str(data) # Tipos de datos

data <- mutate_if(data, is.character, as.factor)
str(data) # Tipos de datos


# Dividir los datos en X y y
X <- data[, 1:94]
y <- data[, 95]


# Entrenar el modelo usando la validación cruzada
set.seed(123)  # Fijar semilla para reproducibilidad

modelo_gbm <- gbm(
  formula = y ~ .,                   # Especificar la fórmula con 'y'
  data = data,                       # Conjunto de datos        
                                     # Para clasificación 
  n.trees = 10000,                   # Número máximo de árboles
  interaction.depth = 3,             # Profundidad de los árboles
  shrinkage = 0.001,                 # Tasa de aprendizaje
  n.minobsinnode = 10,               # Mínimo de observaciones en nodos terminales
  cv.folds = 5,                      # Número de folds para validación cruzada
  n.cores = NULL,                    # Usar todos los núcleos disponibles
  verbose = FALSE                    # Sin mostrar detalles durante el ajuste
)

# Obtener el número óptimo de árboles usando validación cruzada
mejor_n_trees <- gbm.perf(modelo_gbm, method = "cv")

print(mejor_n_trees)  # Imprimir el número óptimo de árboles

# Hacer predicciones con el número óptimo de árboles
predicciones <- predict(modelo_gbm, newdata = data, n.trees = mejor_n_trees, type = "response")

# Evaluar el modelo (ejemplo usando la matriz de confusión
auc <- caret::confusionMatrix(as.factor(ifelse(predicciones > 0.5, 1, 0)), as.factor(data$y))
print(auc)  # Imprimir la matriz de confusión y métricas asociadas