import optuna
from ultralytics import YOLO
import shutil
import os
import json

os.chdir('..')

TOP_K = 3
TOP_RESULTS_FILE = "optuna/top_optuna_trials.json"

def objective(trial):
    # Hiperparámetros a optimizar
    lr = trial.suggest_loguniform('lr0', 1e-5, 1e-2)
    batch = trial.suggest_categorical('batch', [4, 8, 16])
    momentum = trial.suggest_uniform('momentum', 0.6, 0.98)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    img_size = trial.suggest_categorical('imgsz', [640, 960, 1280])

    run_name = f'optuna_trial_{trial.number}'

    model = YOLO('yolov8m.pt')
    results = model.train(
        data='dataset.yaml',
        epochs=30,
        imgsz=img_size,
        batch=batch,
        lr0=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        device=0,
        project='runs/optuna',
        name=run_name,
        cache=True,
        workers=4,
        patience=10,
        save_txt=False,
        save_conf=False,
        verbose=False
    )

    metrics = results.results_dict['metrics/mAP50(B)']

    # Borramos carpeta de corrida para ahorrar espacio
    shutil.rmtree(f'runs/optuna/{run_name}', ignore_errors=True)

    return metrics

# Crear y correr el estudio
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Mostrar y guardar los mejores resultados
top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:TOP_K]
top_results = [
    {
        "trial_number": trial.number,
        "mAP50": trial.value,
        "params": trial.params
    }
    for trial in top_trials
]

# Guardar a archivo
with open(TOP_RESULTS_FILE, "w") as f:
    json.dump(top_results, f, indent=2)

print("\n✅ Top 3 best combinations:")
for i, t in enumerate(top_results, 1):
    print(f"\nTop {i} - Trial {t['trial_number']} (mAP50: {t['mAP50']:.4f})")
    for k, v in t['params'].items():
        print(f"  {k}: {v}")
print(f"\n💾 Saved to '{TOP_RESULTS_FILE}'")

