rm(list = ls())
## Instalar y cargar librerias

pacman::p_load(
  "tidyverse",
  "readxl", 
  "openxlsx",
  "xgboost",
  "caret",
  "data.table"
  
)

# Cargar el conjunto de datos mtcars
data <- read.csv("E:/TESISXGBoost/Tesis/NotasXGBoost/DataFrames/train.csv")



# Dividir los datos en variables predictoras (X) y objetivo (y)
X <- as.matrix(data[, 1:94])  # Las primeras 94 columnas son características
y <- as.factor(data[, 95])    # La columna 95 es la variable objetivo

# Codificar las clases de la variable objetivo
y <- as.numeric(as.factor(y)) - 1  # XGBoost requiere que la variable objetivo sea numérica

# Definir los parámetros iniciales para el modelo XGBoost
xgb_params <- list(
  objective = "multi:softmax",  # Clasificación multiclase
  num_class = length(unique(y)),  # Número de clases
  eval_metric = "mlogloss"  # Métrica para la pérdida de clasificación
)

# Crear una lista de parámetros para afinar (subsample)
grid <- expand.grid(
  nrounds = 100,                   # Número de iteraciones (árboles)
  max_depth = 6,                   # Profundidad máxima de los árboles
  eta = 0.1,                       # Tasa de aprendizaje (shrinkage)
  gamma = 0,                       # Reducción de la ganancia mínima para hacer una división
  colsample_bytree = 0.8,          # Fracción de columnas usadas por árbol
  subsample = c(0.5, 0.7, 0.9)     # Valores de muestreo que queremos afinar
)

# Configurar la validación cruzada estratificada
train_control <- trainControl(
  method = "cv",                   # Validación cruzada
  number = 5,                      # Número de particiones (folds)
  verboseIter = TRUE,              # Mostrar progreso
  allowParallel = TRUE             # Permitir paralelización
)

# Realizar la búsqueda de los mejores parámetros con GridSearchCV
xgb_tune <- train(
  X, y,                            # Datos de entrenamiento
  method = "xgbTree",              # Método XGBoost
  trControl = train_control,       # Control para validación cruzada
  tuneGrid = grid,                 # Conjunto de parámetros a afinar
  metric = "Accuracy",             # Métrica a optimizar
  verbose = FALSE
)

# Mostrar los mejores parámetros encontrados
print(xgb_tune$bestTune)

# Graficar los resultados de la búsqueda
plot(xgb_tune)

