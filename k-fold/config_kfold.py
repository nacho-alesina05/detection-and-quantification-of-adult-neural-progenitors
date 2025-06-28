# config_kfold.py
"""
Archivo de configuración para YOLOv8 K-Fold Cross Validation
Modifica estos parámetros según tus necesidades
"""

# Configuración del K-Fold
K_FOLDS = 5  # Número de folds
RANDOM_STATE = 42  # Semilla para reproducibilidad

# Configuración del entrenamiento
EPOCHS = 300  # Número de épocas
BATCH_SIZE = 4  # Tamaño del batch (ajustar según GPU)
PATIENCE = 50  # Early stopping patience
OPTIMIZER = 'auto'  # Optimizador: 'SGD', 'Adam', 'AdamW', 'auto'
LEARNING_RATE = 0.01  # Tasa de aprendizaje inicial

# Configuración del modelo
MODEL_SIZE = 'x'  # Tamaño del modelo: 'n', 's', 'm', 'l', 'x'
MODEL_PATH = f"yolov8{MODEL_SIZE}.pt"  # Ruta del modelo base

# Configuración del proyecto
PROJECT_NAME = "yolov8_kfold_cv"  # Nombre del proyecto
EXPERIMENT_NAME = None  # Nombre del experimento (None para auto)

# Configuración de validación
VALIDATION_SPLIT = 0.2  # Proporción para validación en cada fold
SAVE_PLOTS = True  # Guardar gráficos de entrenamiento
SAVE_MODELS = True  # Guardar modelos entrenados

# Configuración de hardware
DEVICE = 'auto'  # Dispositivo: 'auto', 'cpu', '0', '1', etc.
WORKERS = 8  # Número de workers para carga de datos

# Configuración de datos
IMAGE_SIZE = 640  # Tamaño de imagen para entrenamiento
AUGMENTATION = True  # Aplicar aumentación de datos

# Configuración de métricas
METRICS_TO_TRACK = [
    'mAP50',
    'mAP50-95', 
    'precision',
    'recall',
    'f1'
]

# Configuración de logs
VERBOSE = True  # Mostrar información detallada
SAVE_LOGS = True  # Guardar logs CSV
