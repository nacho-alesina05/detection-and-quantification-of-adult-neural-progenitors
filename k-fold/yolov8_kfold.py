#!/usr/bin/env python3
"""
YOLOv8 K-Fold Cross Validation Script
Basado en el tutorial de Ultralytics para validación cruzada K-Fold
Adaptado para estructura de datos existente con train/ y validation/
"""

import os
import yaml
import pandas as pd
import shutil
import datetime
import random
from pathlib import Path
from collections import Counter
from sklearn.model_selection import KFold
from tqdm import tqdm
from ultralytics import YOLO

os.chdir('..')

# Configuración
DATASET_PATH = Path(".")  # Directorio actual donde está el script
K_FOLDS = 5
RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 16
PROJECT_NAME = "yolov8_kfold_cv"

# Extensiones de imagen soportadas
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

def load_dataset_config():
    """Cargar configuración del dataset desde dataset.yaml"""
    yaml_file = DATASET_PATH / "dataset.yaml"
    if not yaml_file.exists():
        raise FileNotFoundError(f"No se encontró dataset.yaml en {DATASET_PATH}")
    
    with open(yaml_file, 'r', encoding='utf8') as f:
        config = yaml.safe_load(f)
    
    # Normalizar el formato de las clases
    names = config.get('names', [])
    if isinstance(names, list):
        # Si es una lista, convertir a diccionario {index: name}
        classes = {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        # Si ya es un diccionario, usar tal como está
        classes = names
    else:
        raise ValueError(f"Formato de 'names' no reconocido en dataset.yaml: {type(names)}")
    
    config['names'] = classes
    return config

def collect_all_data():
    """Recolectar todas las imágenes y etiquetas de train/ y validation/"""
    all_images = []
    all_labels = []
    
    # Recolectar de train/
    train_images_dir = DATASET_PATH / "train" / "images"
    train_labels_dir = DATASET_PATH / "train" / "labels"
    
    # Recolectar de validation/
    val_images_dir = DATASET_PATH / "validation" / "images"
    val_labels_dir = DATASET_PATH / "validation" / "labels"
    
    # Buscar imágenes en ambos directorios
    for images_dir in [train_images_dir, val_images_dir]:
        if images_dir.exists():
            for ext in SUPPORTED_EXTENSIONS:
                all_images.extend(sorted(images_dir.glob(f"*{ext}")))
    
    # Buscar etiquetas correspondientes
    for labels_dir in [train_labels_dir, val_labels_dir]:
        if labels_dir.exists():
            all_labels.extend(sorted(labels_dir.glob("*.txt")))
    
    print(f"Total de imágenes encontradas: {len(all_images)}")
    print(f"Total de etiquetas encontradas: {len(all_labels)}")
    
    return all_images, all_labels

def generate_feature_vectors(labels, classes):
    """Generar vectores de características para cada imagen basado en las etiquetas"""
    cls_idx = sorted(classes.keys())
    index = [label.stem for label in labels]
    labels_df = pd.DataFrame([], columns=cls_idx, index=index)
    
    print("Generando vectores de características...")
    print(f"Clases esperadas: {cls_idx}")
    
    problematic_files = []
    class_counts = Counter()
    
    for label in tqdm(labels, desc="Procesando etiquetas"):
        lbl_counter = Counter()
        
        try:
            with open(label, 'r') as lf:
                lines = lf.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # Saltar líneas vacías
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) < 5:  # YOLO format: class x y w h
                        print(f"Advertencia: Línea malformada en {label.name}:{line_num}: {line}")
                        continue
                    
                    class_id = int(parts[0])
                    
                    # Verificar que la clase existe en nuestro diccionario
                    if class_id not in classes:
                        print(f"Advertencia: Clase {class_id} no encontrada en dataset.yaml (archivo: {label.name})")
                        continue
                    
                    lbl_counter[class_id] += 1
                    class_counts[class_id] += 1
                    
                except ValueError as e:
                    print(f"Error parseando línea en {label.name}:{line_num}: {line} - {e}")
                    continue
            
            labels_df.loc[label.stem] = lbl_counter
            
        except Exception as e:
            print(f"Error procesando {label}: {e}")
            problematic_files.append(str(label))
    
    labels_df = labels_df.fillna(0.0)  # Reemplazar valores NaN con 0.0
    
    print(f"\nResumen de clases encontradas:")
    for class_id in sorted(class_counts.keys()):
        class_name = classes.get(class_id, f"Unknown_{class_id}")
        print(f"  Clase {class_id} ({class_name}): {class_counts[class_id]} instancias")
    
    if problematic_files:
        print(f"\nArchivos problemáticos: {len(problematic_files)}")
        for pf in problematic_files[:5]:  # Mostrar solo los primeros 5
            print(f"  {pf}")
    
    # Verificar que tenemos datos
    if labels_df.sum().sum() == 0:
        raise ValueError("No se encontraron anotaciones válidas en los archivos de etiquetas")
    
    return labels_df

def create_kfold_splits(labels_df):
    """Crear divisiones K-Fold del dataset"""
    print(f"Creando {K_FOLDS} divisiones K-Fold...")
    
    # Configurar semilla para reproducibilidad
    random.seed(RANDOM_STATE)
    
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    kfolds = list(kf.split(labels_df))
    
    # Crear DataFrame para mostrar las divisiones
    folds = [f"fold_{n}" for n in range(1, K_FOLDS + 1)]
    folds_df = pd.DataFrame(index=labels_df.index, columns=folds)
    
    for i, (train_idx, val_idx) in enumerate(kfolds, start=1):
        folds_df[f"fold_{i}"].loc[labels_df.iloc[train_idx].index] = "train"
        folds_df[f"fold_{i}"].loc[labels_df.iloc[val_idx].index] = "val"
    
    return kfolds, folds_df

def analyze_fold_distribution(kfolds, labels_df, classes):
    """Analizar la distribución de clases en cada fold"""
    cls_idx = sorted(classes.keys())
    folds = [f"fold_{n}" for n in range(1, K_FOLDS + 1)]
    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)
    
    print("Analizando distribución de clases por fold...")
    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()
        
        # Evitar división por cero
        ratio = val_totals / (train_totals + 1e-7)
        fold_lbl_distrb.loc[f"fold_{n}"] = ratio
    
    print("\nDistribución de clases (val/train ratio):")
    print(fold_lbl_distrb)
    return fold_lbl_distrb

def create_fold_directories(folds_df, classes):
    """Crear directorios y archivos YAML para cada fold"""
    timestamp = datetime.date.today().isoformat()
    save_path = DATASET_PATH / f"{timestamp}_{K_FOLDS}fold_cross_validation"
    save_path.mkdir(parents=True, exist_ok=True)
    
    ds_yamls = []
    
    print(f"Creando estructura de directorios en: {save_path}")
    
    for fold in folds_df.columns:
        # Crear directorios
        fold_dir = save_path / fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        for split in ["train", "val"]:
            (fold_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (fold_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
        # Crear archivo YAML del dataset
        dataset_yaml = fold_dir / f"{fold}_dataset.yaml"
        ds_yamls.append(dataset_yaml)
        
        with open(dataset_yaml, 'w') as f:
            yaml.safe_dump({
                "path": fold_dir.as_posix(),
                "train": "train",
                "val": "val",
                "names": classes,
            }, f)
    
    return save_path, ds_yamls

def copy_files_to_folds(images, labels, folds_df, save_path):
    """Copiar archivos de imágenes y etiquetas a los folds correspondientes"""
    print("Copiando archivos a los folds...")
    
    # Crear diccionario de mapeo imagen -> etiqueta
    image_to_label = {}
    for image in images:
        # Buscar etiqueta correspondiente
        label_name = image.stem + ".txt"
        for label in labels:
            if label.name == label_name:
                image_to_label[image] = label
                break
    
    # Copiar archivos
    for image in tqdm(images, desc="Copiando archivos"):
        if image.stem not in folds_df.index:
            continue
            
        label = image_to_label.get(image)
        if label is None:
            print(f"Advertencia: No se encontró etiqueta para {image.name}")
            continue
        
        for fold, split in folds_df.loc[image.stem].items():
            if pd.isna(split):
                continue
                
            # Directorios de destino
            img_dest = save_path / fold / split / "images"
            lbl_dest = save_path / fold / split / "labels"
            
            # Copiar archivos
            try:
                shutil.copy2(image, img_dest / image.name)
                shutil.copy2(label, lbl_dest / label.name)
            except Exception as e:
                print(f"Error copiando {image.name}: {e}")

def save_logs(folds_df, fold_distribution, save_path):
    """Guardar logs de las divisiones y distribución"""
    print("Guardando logs...")
    folds_df.to_csv(save_path / "kfold_data_splits.csv")
    fold_distribution.to_csv(save_path / "kfold_label_distribution.csv")
    print(f"Logs guardados en: {save_path}")

def train_kfold_models(ds_yamls, model_path="yolov8n.pt"):
    """Entrenar modelos YOLO para cada fold"""
    print(f"\nIniciando entrenamiento con {K_FOLDS} folds...")
    print(f"Modelo base: {model_path}")
    print(f"Épocas: {EPOCHS}, Batch size: {BATCH_SIZE}")
    
    results = {}
    
    for k, dataset_yaml in enumerate(ds_yamls):
        print(f"\n{'='*50}")
        print(f"ENTRENANDO FOLD {k+1}/{K_FOLDS}")
        print(f"{'='*50}")
        
        # Cargar modelo nuevo para cada fold
        model = YOLO(model_path, task="detect")
        
        # Entrenar
        results[k] = model.train(
            data=dataset_yaml,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            project=PROJECT_NAME,
            name=f"fold_{k+1}",
            patience=10,  # Early stopping
            save=True,
            plots=True,
            verbose=True
        )
        
        print(f"Fold {k+1} completado!")
    
    return results

def print_summary(results):
    """Imprimir resumen de resultados"""
    print("\n" + "="*60)
    print("RESUMEN DE VALIDACIÓN CRUZADA K-FOLD")
    print("="*60)
    
    metrics = []
    for k, result in results.items():
        # Extraer métricas del último epoch
        metrics_dict = {
            'Fold': k + 1,
            'mAP50': result.results_dict.get('metrics/mAP50(B)', 0),
            'mAP50-95': result.results_dict.get('metrics/mAP50-95(B)', 0),
            'Precision': result.results_dict.get('metrics/precision(B)', 0),
            'Recall': result.results_dict.get('metrics/recall(B)', 0),
        }
        metrics.append(metrics_dict)
    
    df_results = pd.DataFrame(metrics)
    print(df_results)
    
    # Calcular promedios
    print("\nPROMEDIOS:")
    for col in ['mAP50', 'mAP50-95', 'Precision', 'Recall']:
        mean_val = df_results[col].mean()
        std_val = df_results[col].std()
        print(f"{col}: {mean_val:.4f} ± {std_val:.4f}")

def main():
    """Función principal"""
    print("Iniciando YOLOv8 K-Fold Cross Validation")
    print(f"K-Folds: {K_FOLDS}")
    print(f"Directorio de trabajo: {DATASET_PATH.absolute()}")
    
    # 1. Cargar configuración del dataset
    print("\n1. Cargando configuración del dataset...")
    config = load_dataset_config()
    classes = config['names']
    print(f"Clases encontradas: {list(classes.values())}")
    print(f"Número de clases: {len(classes)}")
    
    # 2. Recolectar todos los datos
    print("\n2. Recolectando datos...")
    images, labels = collect_all_data()
    
    if len(images) == 0 or len(labels) == 0:
        raise ValueError("No se encontraron imágenes o etiquetas")
    
    print(f"Total imágenes: {len(images)}")
    print(f"Total etiquetas: {len(labels)}")
    
    # Verificar que tenemos etiquetas para todas las imágenes
    image_stems = {img.stem for img in images}
    label_stems = {lbl.stem for lbl in labels}
    
    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems
    
    if missing_labels:
        print(f"Advertencia: {len(missing_labels)} imágenes sin etiquetas")
        print(f"Primeras 5: {list(missing_labels)[:5]}")
    
    if missing_images:
        print(f"Advertencia: {len(missing_images)} etiquetas sin imágenes")
        print(f"Primeras 5: {list(missing_images)[:5]}")
    
    # Filtrar solo las imágenes que tienen etiquetas correspondientes
    valid_stems = image_stems & label_stems
    images = [img for img in images if img.stem in valid_stems]
    labels = [lbl for lbl in labels if lbl.stem in valid_stems]
    
    print(f"Imágenes válidas (con etiquetas): {len(images)}")
    print(f"Etiquetas válidas (con imágenes): {len(labels)}")
    
    # 3. Generar vectores de características
    print("\n3. Generando vectores de características...")
    labels_df = generate_feature_vectors(labels, classes)
    print(f"Forma del DataFrame: {labels_df.shape}")
    
    # 4. Crear divisiones K-Fold
    print("\n4. Creando divisiones K-Fold...")
    kfolds, folds_df = create_kfold_splits(labels_df)
    
    # 5. Analizar distribución
    print("\n5. Analizando distribución de clases...")
    fold_distribution = analyze_fold_distribution(kfolds, labels_df, classes)
    
    # 6. Crear estructura de directorios
    print("\n6. Creando estructura de directorios...")
    save_path, ds_yamls = create_fold_directories(folds_df, classes)
    
    # 7. Copiar archivos
    print("\n7. Copiando archivos...")
    copy_files_to_folds(images, labels, folds_df, save_path)
    
    # 8. Guardar logs
    print("\n8. Guardando logs...")
    save_logs(folds_df, fold_distribution, save_path)
    
    # 9. Entrenar modelos
    print("\n9. Entrenando modelos...")
    # Buscar modelo YOLOv8 disponible
    model_candidates = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
    model_path = None
    
    for candidate in model_candidates:
        if (DATASET_PATH / candidate).exists():
            model_path = candidate
            break
    
    if model_path is None:
        model_path = "yolov8n.pt"  # Se descargará automáticamente
        print(f"No se encontró modelo local, se usará {model_path}")
    
    results = train_kfold_models(ds_yamls, model_path)
    
    # 10. Mostrar resumen
    print("\n10. Generando resumen...")
    print_summary(results)
    
    print(f"\n¡Validación cruzada completada!")
    print(f"Resultados guardados en: runs/{PROJECT_NAME}/")
    print(f"Estructura de folds en: {save_path}")

if __name__ == "__main__":
    main()
