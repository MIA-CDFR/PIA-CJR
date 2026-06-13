# ============================================================
# FILE: scripts/evaluate.py
# ============================================================

# Correr desta forma:
#
#   Se não especificar nada em argumento, carrega o latest model densenet.pth
#       (DL_Project_venv) PS C:\_PIA\PIA-CJR\coding\_CB> python .\scripts\evaluate.py
#
#   Se passarmos argumento, carrega o modelo passado em argumento
#       (DL_Project_venv) PS C:\_PIA\PIA-CJR\coding\_CB> python .\scripts\evaluate.py --model densenet121_CB_20260521_160502.pth

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

import sys
import json

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# IMPORTS
# ============================================================

import argparse

import torch

from torch.utils.data import DataLoader, Subset

from torchvision import transforms

from src.datasets.elpv_dataset import ELPVDataset

from src.models.densenet_model import build_densenet121

from src.evaluation.evaluator import Evaluator

from src.utils.config_loader import load_config

from src.utils.logger import setup_logger

from src.utils.reproducibility import set_seed

# ============================================================
# MAIN
# ============================================================


def main():

    # ========================================================
    # ARGUMENT PARSER
    # ========================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Specific model filename to evaluate. " "If omitted, loads latest model."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # LOAD CONFIG
    # ========================================================

    config = load_config()

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model_name = config["MODEL_NAME"]
    num_classes = config["NUM_CLASSES"]
    dropout = config["DROPOUT"]

    # ========================================================
    # LOGGER
    # ========================================================

    logger = setup_logger("evaluate")

    logger.info("====================================")

    logger.info("STARTING EVALUATION")

    logger.info(f"Model: {model_name}")

    logger.info("====================================")

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    seed = config["RANDOM_SEED"]

    set_seed(seed)

    logger.info(f"Random seed: {seed}")

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {device}")

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
    # DATASET
    # ========================================================

    dataset = ELPVDataset(transform=transform, binary_classification=True)

    logger.info(f"Full dataset size: {len(dataset)}")


    # ========================================================
    # MODEL SELECTION
    # ========================================================

    model = build_densenet121(
        num_classes=num_classes, pretrained=False, dropout=dropout
    )

    # ========================================================
    # LOAD MODEL WEIGHTS
    # ========================================================

    models_dir = Path(config["MODELS_DIR"])

    if args.model is None:

        model_filename = f"{model_name}.pth"

    else:

        model_filename = args.model

    model_path = models_dir / model_filename

    logger.info(f"Loading model: {model_path}")

    if not model_path.exists():

        raise FileNotFoundError(f"Model not found:\n{model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)

    logger.info("Model loaded successfully")

    # ========================================================
    # RUN ID
    # ========================================================

    run_id = Path(model_filename).stem

    # ========================================================
    # LOAD SPLITS
    # ========================================================

    splits_path = Path(config["MODELS_DIR"]) / f"{model_name}_splits.json"

    if not splits_path.exists():

        raise FileNotFoundError(f"Splits file not found:\n{splits_path}")

    with open(splits_path, "r") as f:

        splits = json.load(f)

    test_indices = splits["test_indices"]

    dataset = Subset(dataset, test_indices)

    logger.info(f"Test samples: {len(dataset)}")

    # ========================================================
    # DATALOADER
    # ========================================================

    batch_size = config["BATCH_SIZE"]

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Batch size: {batch_size}")

    # ========================================================
    # CLASS NAMES
    # ========================================================

    class_names = config["CLASS_NAMES"]

    # ========================================================
    # EVALUATOR
    # ========================================================

    evaluator = Evaluator(
        model=model,
        dataloader=dataloader,
        device=device,
        config=config,
        class_names=class_names,
        run_id=run_id,
    )

    # ========================================================
    # RUN EVALUATION
    # ========================================================

    metrics = evaluator.evaluate()

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    logger.info("====================================")

    logger.info("EVALUATION RESULTS")

    logger.info("====================================")

    for key, value in metrics.items():

        logger.info(f"{key}: {value:.4f}")

    logger.info("====================================")

    logger.info("EVALUATION FINISHED")

    logger.info("====================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
