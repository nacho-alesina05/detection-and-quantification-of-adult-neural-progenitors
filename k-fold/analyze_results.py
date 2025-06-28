#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
import yaml
import os

os.chdir('..')

def load_results_from_runs(project_name="yolov8_kfold_cv"):
    """Cargar resultados de los runs de entrenamiento"""
    runs_path = Path("runs/detect")
    results = []
    
    # Buscar carpetas de folds
    for fold_dir in sorted(runs_path.glob(f"{project_name}/fold_*")):
        if fold_dir.is_dir():
            fold_num = int(fold_dir.name.split('_')[1])
            
            # Cargar resultados del CSV
            results_csv = fold_dir / "results.csv"
            if results_csv.exists():
                df = pd.read_csv(results_csv)
                # Tomar las métricas del último epoch
                last_epoch = df.iloc[-1]
                
                fold_results = {
                    'fold': fold_num,
                    'epochs_trained': len(df),
                    'train_loss': last_epoch.get('train/box_loss', 0) + 
                                 last_epoch.get('train/cls_loss', 0) + 
                                 last_epoch.get('train/dfl_loss', 0),
                    'val_loss': last_epoch.get('val/box_loss', 0) + 
                               last_epoch.get('val/cls_loss', 0) + 
                               last_epoch.get('val/dfl_loss', 0),
                    'mAP50': last_epoch.get('metrics/mAP50(B)', 0),
                    'mAP50-95': last_epoch.get('metrics/mAP50-95(B)', 0),
                    'precision': last_epoch.get('metrics/precision(B)', 0),
                    'recall': last_epoch.get('metrics/recall(B)', 0),
                    'f1': last_epoch.get('metrics/f1(B)', 0),
                }
                results.append(fold_results)
    
    return pd.DataFrame(results)

def calculate_statistics(df):
    """Calcular estadísticas descriptivas"""
    metrics = ['train_loss', 'val_loss', 'mAP50', 'mAP50-95', 'precision', 'recall', 'f1']
    
    stats = {}
    for metric in metrics:
        if metric in df.columns:
            stats[metric] = {
                'mean': df[metric].mean(),
                'std': df[metric].std(),
                'min': df[metric].min(),
                'max': df[metric].max(),
                'cv': df[metric].std() / df[metric].mean() if df[metric].mean() != 0 else 0
            }
    
    return stats

def plot_results(df, save_path=None):
    """Crear visualizaciones de los resultados"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('YOLOv8 K-Fold Cross Validation Results', fontsize=16)
    
    metrics = ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1', 'val_loss']
    
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            row = i // 3
            col = i % 3
            
            # Box plot
            axes[row, col].boxplot(df[metric])
            axes[row, col].set_title(f'{metric}')
            axes[row, col].set_ylabel('Value')
            
            # Agregar puntos individuales
            y = df[metric]
            x = np.random.normal(1, 0.04, size=len(y))
            axes[row, col].scatter(x, y, alpha=0.6, color='red')
            
            # Agregar estadísticas
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            axes[row, col].axhline(mean_val, color='green', linestyle='--', 
                                 label=f'Mean: {mean_val:.3f}')
            axes[row, col].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'kfold_results_boxplots.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_fold_comparison(df, save_path=None):
    """Comparar métricas entre folds"""
    metrics = ['mAP50', 'mAP50-95', 'precision', 'recall']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('K-Fold Comparison Across Folds', fontsize=16)
    
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            row = i // 2
            col = i % 2
            
            axes[row, col].bar(df['fold'], df[metric], alpha=0.7)
            axes[row, col].set_title(f'{metric} by Fold')
            axes[row, col].set_xlabel('Fold')
            axes[row, col].set_ylabel(metric)
            
            # Línea del promedio
            mean_val = df[metric].mean()
            axes[row, col].axhline(mean_val, color='red', linestyle='--', 
                                 label=f'Mean: {mean_val:.3f}')
            axes[row, col].legend()
            
            # Anotar valores
            for j, v in enumerate(df[metric]):
                axes[row, col].text(df['fold'].iloc[j], v + 0.01, f'{v:.3f}', 
                                  ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path / 'kfold_comparison.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def generate_report(df, stats, save_path=None):
    """Generar reporte detallado"""
    report = []
    report.append("="*60)
    report.append("YOLOv8 K-FOLD CROSS VALIDATION RESULTS REPORT")
    report.append("="*60)
    report.append(f"Number of folds: {len(df)}")
    report.append(f"Total epochs trained: {df['epochs_trained'].sum()}")
    report.append("")
    
    report.append("SUMMARY STATISTICS:")
    report.append("-" * 40)
    
    for metric, values in stats.items():
        report.append(f"\n{metric.upper()}:")
        report.append(f"  Mean: {values['mean']:.4f}")
        report.append(f"  Std:  {values['std']:.4f}")
        report.append(f"  Min:  {values['min']:.4f}")
        report.append(f"  Max:  {values['max']:.4f}")
        report.append(f"  CV:   {values['cv']:.4f}")
    
    report.append("\n" + "="*60)
    report.append("DETAILED RESULTS BY FOLD:")
    report.append("="*60)
    
    for _, row in df.iterrows():
        report.append(f"\nFOLD {row['fold']}:")
        report.append(f"  Epochs: {row['epochs_trained']}")
        report.append(f"  mAP50: {row['mAP50']:.4f}")
        report.append(f"  mAP50-95: {row['mAP50-95']:.4f}")
        report.append(f"  Precision: {row['precision']:.4f}")
        report.append(f"  Recall: {row['recall']:.4f}")
        report.append(f"  F1: {row['f1']:.4f}")
        report.append(f"  Val Loss: {row['val_loss']:.4f}")
    
    report.append("\n" + "="*60)
    report.append("RECOMMENDATIONS:")
    report.append("="*60)
    
    # Análisis de variabilidad
    mAP50_cv = stats.get('mAP50', {}).get('cv', 0)
    if mAP50_cv < 0.1:
        report.append("✓ Low variability in mAP50 - Good model consistency")
    elif mAP50_cv > 0.2:
        report.append("⚠ High variability in mAP50 - Consider data balancing")
    
    # Análisis de overfitting
    if 'train_loss' in stats and 'val_loss' in stats:
        train_loss_mean = stats['train_loss']['mean']
        val_loss_mean = stats['val_loss']['mean']
        if val_loss_mean > train_loss_mean * 1.2:
            report.append("⚠ Potential overfitting detected - Consider regularization")
        else:
            report.append("✓ No significant overfitting detected")
    
    report_text = "\n".join(report)
    print(report_text)
    
    if save_path:
        with open(save_path / 'kfold_report.txt', 'w') as f:
            f.write(report_text)
    
    return report_text

def main():
    """Función principal del analizador"""
    print("Analyzing K-Fold Cross Validation results...")
    
    # Cargar resultados
    df = load_results_from_runs()
    
    if df.empty:
        print("No results found. Make sure you have run K-Fold training.")
        return
    
    print(f"Results loaded for {len(df)} folds")
    
    # Calcular estadísticas
    stats = calculate_statistics(df)
    
    # Crear directorio para resultados
    results_path = Path("kfold_analysis_results")
    results_path.mkdir(exist_ok=True)
    
    # Generar visualizaciones
    print("Generating visualizations...")
    plot_results(df, results_path)
    plot_fold_comparison(df, results_path)
    
    # Generar reporte
    print("Generating report...")
    generate_report(df, stats, results_path)
    
    # Guardar DataFrame
    df.to_csv(results_path / 'kfold_results.csv', index=False)
    
    print(f"\nAnalysis completed. Results saved in: {results_path}")

if __name__ == "__main__":
    main()
