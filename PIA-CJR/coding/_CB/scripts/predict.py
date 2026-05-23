# ============================================================
# FILE: scripts/predict.py
# ============================================================

# ============================================================
# EXAMPLES
# ============================================================

# Default RAW_DIR from config.yaml
# python .\scripts\predict.py

# Single image
# python .\scripts\predict.py ^
#     --input "data/image.jpg"

# Folder
# python .\scripts\predict.py ^
#     --input "data/images"

# Specific model
# python .\scripts\predict.py ^
#     --input "data/images" ^
#     --model "efficientnet_b0.pth"

# ============================================================
# IMPORTS
# ============================================================

import argparse
import sys

from pathlib import Path

from datetime import datetime

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

# ============================================================
# IMPORTS
# ============================================================

import torch

from torchvision import transforms

from src.models.densenet_model import build_densenet121

from src.models.efficientnet_model import build_efficientnet_b0

from src.inference.predictor import Predictor

from src.utils.config_loader import load_config

from src.utils.logger import setup_logger

# ============================================================
# MAIN
# ============================================================


def main():

    # ========================================================
    # ARGUMENTS
    # ========================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input", type=str, required=False, help="Input image or folder"
    )

    parser.add_argument(
        "--model", type=str, default=None, help="Specific model filename"
    )

    args = parser.parse_args()

    # ========================================================
    # CONFIG
    # ========================================================

    config = load_config()

    # ========================================================
    # DEFAULT INPUT PATH
    # ========================================================

    if args.input is None:

        input_path = Path(config["RAW_DIR"])

    else:

        input_path = Path(args.input)

    # ========================================================
    # ACTIVE MODEL
    # ========================================================

    active_model = config["ACTIVE_MODEL"]

    model_config = config["MODELS"][active_model]

    # ========================================================
    # VERSIONED PREDICTION OUTPUT
    # ========================================================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    prediction_run_name = f"predict_" f"{active_model}_" f"{timestamp}"

    output_dir = Path(config["PREDICTIONS_DIR"]) / prediction_run_name

    # ========================================================
    # LOGGER
    # ========================================================

    logger = setup_logger("predict")

    logger.info("====================================")

    logger.info("STARTING PREDICTION")

    logger.info(f"Active model: {active_model}")

    logger.info(f"Input path: {input_path}")

    logger.info(f"Output path: {output_dir}")

    logger.info("====================================")

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {device}")

    # ========================================================
    # MODEL CONFIG
    # ========================================================

    num_classes = model_config["NUM_CLASSES"]

    dropout = model_config["DROPOUT"]

    # ========================================================
    # MODEL SELECTION
    # ========================================================

    # model = build_densenet121(
    #     num_classes=num_classes, pretrained=False, dropout=dropout
    # )
    
    if active_model == "densenet121":

        model = build_densenet121(
            num_classes=num_classes, pretrained=False, dropout=dropout
        )

    elif active_model == "efficientnet_b0":

        model = build_efficientnet_b0(
            num_classes=num_classes, pretrained=False, dropout=dropout
        )

    else:

        raise ValueError(f"Unsupported model: " f"{active_model}")

    # ========================================================
    # LOAD MODEL
    # ========================================================

    models_dir = Path(config["MODELS_DIR"])

    if args.model is None:

        model_filename = f"{active_model}.pth"

    else:

        model_filename = args.model

    model_path = models_dir / model_filename

    logger.info(f"Loading model: {model_path}")

    if not model_path.exists():

        raise FileNotFoundError(f"Model not found:\n{model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)

    model.eval()

    logger.info("Model loaded successfully")

    # ========================================================
    # TRANSFORMS
    # ========================================================

    image_size = config["IMAGE_SIZE"]

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # ========================================================
    # PREDICTOR
    # ========================================================

    predictor = Predictor(
        model=model,
        device=device,
        transform=transform,
        class_names=config["CLASS_NAMES"],
        threshold=config["DEFECT_THRESHOLD"],
        logger=logger,
    )

    # ========================================================
    # RUN PREDICTION
    # ========================================================

    predictor.predict(input_path=input_path, output_dir=output_dir)

    logger.info("====================================")

    logger.info("PREDICTION FINISHED")

    logger.info("====================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
