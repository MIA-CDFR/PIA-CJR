# Correr desta forma:
#   (DL_Project_venv) PS C:\_PIA\PIA-CJR\coding\_CB> python .\scripts\train.py

# ============================================================
# FILE: scripts/train.py
# ============================================================

# Correr assim:
#   (DL_Project_venv) PS C:\_PIA\PIA-CJR\coding\_CB> python .\scripts\train.py

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

from datetime import datetime

import torch

from torch.utils.data import (
    DataLoader,
    random_split
)

from torchvision import transforms

from src.utils.config_loader import load_config

from src.utils.logger import setup_logger

from src.utils.reproducibility import (
    set_seed
)

from src.datasets.elpv_dataset import (
    ELPVDataset
)

from src.models.densenet_model import (
    build_densenet121
)

from src.training.trainer import (
    train_model
)


def main():

    # --------------------------------------------------------
    # LOAD CONFIG
    # --------------------------------------------------------

    config = load_config()

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    set_seed(
        config["RANDOM_SEED"]
    )

    # --------------------------------------------------------
    # LOGGER
    # --------------------------------------------------------

    logger = setup_logger("train")

    logger.info("========================================")

    logger.info("PIA-CJR TRAINING")

    # --------------------------------------------------------
    # RUN ID
    # --------------------------------------------------------

    run_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    logger.info(f"Run ID: {run_id}")

    logger.info("========================================")

    # --------------------------------------------------------
    # CONFIG VARIABLES
    # --------------------------------------------------------

    batch_size = config["BATCH_SIZE"]

    num_epochs = config["NUM_EPOCHS"]

    learning_rate = config["LEARNING_RATE"]

    train_split = config["TRAIN_SPLIT"]

    image_size = config["IMAGE_SIZE"]

    num_workers = config["NUM_WORKERS"]

    num_classes = config["NUM_CLASSES"]

    pretrained = config["PRETRAINED"]

    dropout = config["DROPOUT"]

    configured_device = config["DEVICE"]

    models_dir = Path(
        config["MODELS_DIR"]
    )

    figures_dir = Path(
        config["FIGURES_DIR"]
    )

    checkpoints_dir = Path(
        config["CHECKPOINTS_DIR"]
    )

    save_checkpoints = config[
        "SAVE_CHECKPOINTS"
    ]

    checkpoint_every_n_epochs = config[
        "CHECKPOINT_EVERY_N_EPOCHS"
    ]

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if (
        configured_device == "cuda"
        and torch.cuda.is_available()
    ):

        device = "cuda"

    else:

        device = "cpu"

    logger.info(f"Device: {device}")

    # --------------------------------------------------------
    # TRANSFORMS
    # --------------------------------------------------------

    transform = transforms.Compose([

        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[0.485, 0.456, 0.406],

            std=[0.229, 0.224, 0.225]
        )
    ])

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = ELPVDataset(

        transform=transform,

        binary_classification=True
    )

    logger.info(
        f"Dataset size: {len(dataset)}"
    )

    # --------------------------------------------------------
    # DATASET STATISTICS
    # --------------------------------------------------------

    stats = dataset.get_statistics()

    logger.info(
        f"No defect samples: "
        f"{stats['no_defect']}"
    )

    logger.info(
        f"Defect samples: "
        f"{stats['defect']}"
    )

    # --------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    train_size = int(
        train_split * len(dataset)
    )

    val_size = (
        len(dataset) - train_size
    )

    train_dataset, val_dataset = random_split(

        dataset,

        [train_size, val_size]
    )

    logger.info(
        f"Train samples: {len(train_dataset)}"
    )

    logger.info(
        f"Validation samples: {len(val_dataset)}"
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = build_densenet121(

        num_classes=num_classes,

        pretrained=pretrained,

        dropout=dropout
    )

    model = model.to(device)

    # --------------------------------------------------------
    # LOSS FUNCTION
    # --------------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=learning_rate
    )

    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    train_model(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        criterion=criterion,

        optimizer=optimizer,

        device=device,

        epochs=num_epochs,

        models_dir=models_dir,

        figures_dir=figures_dir,

        checkpoints_dir=checkpoints_dir,

        save_checkpoints=save_checkpoints,

        checkpoint_every_n_epochs=
            checkpoint_every_n_epochs,

        model_name=config["MODEL_NAME"],

        user_id=config["USER_ID"],

        run_id=run_id,

        config=config,

        logger=logger
    )


if __name__ == "__main__":

    main()