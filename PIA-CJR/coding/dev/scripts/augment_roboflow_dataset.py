# ============================================================
# FILE: scripts/0_augment_dataset.py
# ============================================================
#
# Objectivo é aumentar (artificialmente) "AUGMENTATIONS_PER_IMAGE vezes" o dataset de treino, valid e test, 
#       aplicando transformações seguras para segmentação YOLO.

# Correr assim:
#   (DL_Project_venv) PS C:\_PIA\PIA-CJR\coding\_CB>
#   python .\scripts\0_augment_dataset.py

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

import sys
import json

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# IMPORTS
# ==========================================================

import shutil
import cv2
import albumentations as A

from src.utils.config_loader import load_config

# ==========================================================
# CONFIG
# ==========================================================

config = load_config()

ROBOFLOW_DIR = Path(config["ROBOFLOW_DIR"])
ROBOFLOW_AUGMENTED_DIR = Path(config["ROBOFLOW_AUGMENTED_DIR"])

# ==========================================================
# SAFE AUGMENTATIONS FOR YOLO SEGMENTATION
# ==========================================================

transform = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=0.25,
        contrast_limit=0.25,
        p=1.0,
    ),

    A.GaussNoise(
        std_range=(0.02, 0.08),
        p=0.5,
    ),

    A.GaussianBlur(
        blur_limit=(3, 5),
        p=0.3,
    ),

    A.CLAHE(
        clip_limit=(1, 4),
        p=0.5,
    ),

    A.RandomGamma(
        gamma_limit=(80, 120),
        p=0.5,
    ),
])

# ==========================================================
# HELPERS
# ==========================================================

IMAGE_EXTENSIONS = set(config["image_extensions"])

def recreate_output_structure():

    if ROBOFLOW_AUGMENTED_DIR.exists():
        shutil.rmtree(ROBOFLOW_AUGMENTED_DIR)

    for split in ["train", "valid", "test"]:

        (ROBOFLOW_AUGMENTED_DIR / split / "images").mkdir(
            parents=True,
            exist_ok=True,
        )

        (ROBOFLOW_AUGMENTED_DIR / split / "labels").mkdir(
            parents=True,
            exist_ok=True,
        )

def process_split(split):

    images_dir = ROBOFLOW_DIR / split / "images"
    labels_dir = ROBOFLOW_DIR / split / "labels"

    out_images = ROBOFLOW_AUGMENTED_DIR / split / "images"
    out_labels = ROBOFLOW_AUGMENTED_DIR / split / "labels"

    image_files = [
        p
        for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"\n[{split}] {len(image_files)} imagens")

    for image_path in image_files:

        label_path = labels_dir / f"{image_path.stem}.txt"

        # ----------------------------------------
        # copiar original
        # ----------------------------------------

        shutil.copy2(
            image_path,
            out_images / image_path.name,
        )

        if label_path.exists():

            shutil.copy2(
                label_path,
                out_labels / label_path.name,
            )

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Erro a abrir {image_path}")
            continue

        # ----------------------------------------
        # gerar augmentações
        # ----------------------------------------

        for i in range(config["AUGMENTATIONS_PER_IMAGE"]):

            augmented = transform(image=image)

            aug_image = augmented["image"]

            aug_name = (
                f"{image_path.stem}_aug_{i:02d}"
            )

            cv2.imwrite(
                str(
                    out_images /
                    f"{aug_name}{image_path.suffix}"
                ),
                aug_image,
            )

            if label_path.exists():

                shutil.copy2(
                    label_path,
                    out_labels /
                    f"{aug_name}.txt"
                )

def main():

    # Para o caso de termos optado pelo dataset augmentado (YOLA_DATA_FOLDER_EQUALS_AUGMENTED: True),
    #   vamos aumentar o dataset e criar a estrutura de pastas e ficheiros do dataset augmentado;
    #   caso contrário, n~ºao fazemos nada, ou seja, mantemos o dataset original (o do roboflow) para treino, valid e test.
    if (config["YOLO_DATA_FOLDER_EQUALS_AUGMENTED"]):
        recreate_output_structure()

        for split in ["train", "valid", "test"]:

            process_split(split)

        print()
        print("=" * 60)
        print("DATASET AUGMENTADO COM SUCESSO")
        print(f"Output: {ROBOFLOW_AUGMENTED_DIR}")
        print("=" * 60)

        YOLO_DATA_ROBOFLOW_YAML = Path(config["YOLO_DATA_ROBOFLOW_YAML"])
        YOLO_DATA_ROBOFLOW_AUGMENTED_YAML = Path(config["YOLO_DATA_ROBOFLOW_AUGMENTED_YAML"])

        # Copiar data.yaml de roboflow para roboflow_augmented
        src_yaml = YOLO_DATA_ROBOFLOW_YAML
        dst_yaml = YOLO_DATA_ROBOFLOW_AUGMENTED_YAML
        
        if src_yaml.exists():
            dst_yaml.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_yaml, dst_yaml)
            print(f"data.yaml copiado de: {src_yaml}")
            print(f"                para: {dst_yaml}")
        else:
            raise FileNotFoundError(f"data.yaml não encontrado: {src_yaml}")
    else:
        print("A opção YOLO_DATA_FOLDER_EQUALS_AUGMENTED está definida como False.")
        print("Nenhuma ação de augmentação foi realizada.")
        print("Será utilizado o dataset original (o do roboflow) para treino, valid e test.")

if __name__ == "__main__":
    main()