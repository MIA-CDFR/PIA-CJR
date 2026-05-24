# ============================================================
# FILE: scripts/roboflow/1_train.py
# ============================================================

# Correr assim:
#   (DL_Project_venv) PS C:\_PIA\PIA-CJR\coding\_CB>
#   python .\scripts\roboflow\1_train.py

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

import sys
import json

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import shutil

from ultralytics import YOLO

from src.utils.config_loader import load_config

from datetime import datetime


def get_next_yolo_run_name(project_dir, base_name):
    project_dir = Path(project_dir)

    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{base_name}_{timestamp}"
        run_dir = project_dir / run_name

        if not run_dir.exists():
            return run_name

        version += 1


def delete_file_if_exists(file_path):
    file_path = Path(file_path)

    if file_path.exists():
        file_path.unlink()


def main():

    config = load_config()

    YOLO_MODEL_WEIGHTS = config["YOLO_MODEL_WEIGHTS"]
    YOLO_MODEL_WEIGHTS_N = config["YOLO_MODEL_WEIGHTS_N"]
    YOLO_DATA_YAML = config["YOLO_DATA_YAML"]
    YOLO_PROJECT_DIR = Path(config["YOLO_PROJECT_DIR"])
    MODELS_DIR = Path(config["MODELS_DIR"])
    YOLO_RUN_NAME = config["YOLO_RUN_NAME"]
    YOLO_BEST_PT = config["YOLO_BEST_PT"]

    versioned_run_name = get_next_yolo_run_name(YOLO_PROJECT_DIR, YOLO_RUN_NAME)

    model = YOLO(YOLO_MODEL_WEIGHTS)

    model.train(
        data=YOLO_DATA_YAML,
        epochs=config["YOLO_EPOCHS"],
        patience=config["YOLO_PATIENCE"],
        cache=config["YOLO_CACHE"],
        plots=config["YOLO_PLOTS"],
        imgsz=config["YOLO_IMAGE_SIZE"],
        batch=config["YOLO_BATCH_SIZE"],
        workers=config["YOLO_NUM_WORKERS"],
        project=YOLO_PROJECT_DIR,
        name=versioned_run_name,
        exist_ok=False,
    )

    # versioned_best_pt = YOLO_PROJECT_DIR / versioned_run_name / "weights" / YOLO_BEST_PT

    versioned_best_pt = (
        YOLO_PROJECT_DIR
        / versioned_run_name
        / "weights"
        / "best.pt"
    )
    # latest_best_pt = YOLO_PROJECT_DIR / YOLO_BEST_PT
    yolo_model = MODELS_DIR / YOLO_BEST_PT

    # if not versioned_best_pt.exists():
    #     raise FileNotFoundError(f"{YOLO_BEST_PT} não encontrado: {versioned_best_pt}")

    if not versioned_best_pt.exists():
        raise FileNotFoundError(
            f"best.pt não encontrado: {versioned_best_pt}"
        )
    
    # shutil.copy2(versioned_best_pt, yolo_model)
    shutil.copy2(versioned_best_pt, yolo_model)

    # print(f"Modelo versionado guardado em: {versioned_best_pt}")
    print(f"Modelo original YOLO: {versioned_best_pt}")
    print(f"Modelo latest copiado para: {yolo_model}")

    # no final apagar os pesos usados para treino porque não precisamos deles, e ocupam espaço
    delete_file_if_exists(YOLO_MODEL_WEIGHTS)
    delete_file_if_exists(YOLO_MODEL_WEIGHTS_N)

if __name__ == "__main__":
    main()
