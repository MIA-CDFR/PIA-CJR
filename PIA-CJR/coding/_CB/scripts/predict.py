# Correr desta forma:
#
#   Dá para especificar um modelo específico
#   Especificar imagem
#       (DL_Project_venv) PS C:\_PIA\PIA-CJR\coding\_CB> python .\scripts\predict.py --image "D:\_PIA_CJR\dataset\raw\2026_05_14_VALPACOS\visible\dji_20260316104948_0125.jpg" --model "outputs\_CB\models\best_model.pth"
#
#   Especificar folder (recursivo), e corre para baixo até ao infinito
#       (DL_Project_venv) PS C:\_PIA\PIA-CJR\coding\_CB> python .\scripts\predict.py --folder "D:\_PIA_CJR\dataset\raw" --model "outputs\_CB\models\best_model.pth"


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import argparse

from src.inference.predictor import Predictor


def parse_args():

    parser = argparse.ArgumentParser(
        description="Inference Pipeline"
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Path to single image"
    )

    parser.add_argument(
        "--folder",
        type=str,
        help="Path to folder containing images recursively"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    predictor = Predictor(
        model_path=args.model
    )

    if args.image:

        predictor.predict_image(
            image_path=args.image
        )

    elif args.folder:

        predictor.predict_folder(
            folder_path=args.folder
        )

    else:

        raise ValueError(
            "You must provide --image or --folder"
        )


if __name__ == "__main__":
    main()