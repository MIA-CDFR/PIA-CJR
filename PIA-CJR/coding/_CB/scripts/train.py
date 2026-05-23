# ============================================================
# FILE: scripts/train.py
# ============================================================

# Correr assim:
#   (DL_Project_venv) PS C:\_PIA\PIA-CJR\coding\_CB>
#   python .\scripts\train.py

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

from datetime import datetime

import torch

from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split

from torchvision import transforms

from src.utils.config_loader import load_config

from src.utils.logger import setup_logger

from src.utils.reproducibility import set_seed

from src.utils.class_weights import compute_class_weights

from src.datasets.elpv_dataset import ELPVDataset

from src.models.densenet_model import build_densenet121

from src.models.efficientnet_model import build_efficientnet_b0

from src.training.trainer import train_model

# ============================================================
# MAIN
# ============================================================


def main():

    # --------------------------------------------------------
    # LOAD CONFIG
    # --------------------------------------------------------

    config = load_config()

    # --------------------------------------------------------
    # ACTIVE MODEL
    # --------------------------------------------------------

    active_model = config["ACTIVE_MODEL"]

    model_config = config["MODELS"][active_model]

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    set_seed(config["RANDOM_SEED"])

    # --------------------------------------------------------
    # LOGGER
    # --------------------------------------------------------

    logger = setup_logger("train")

    logger.info("========================================")

    logger.info("PIA-CJR TRAINING")

    logger.info(f"Active model: {active_model}")

    # --------------------------------------------------------
    # RUN ID
    # --------------------------------------------------------

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"Run ID: {run_id}")

    logger.info("========================================")

    # --------------------------------------------------------
    # CONFIG VARIABLES
    # --------------------------------------------------------

    batch_size = config["BATCH_SIZE"]

    num_epochs = config["NUM_EPOCHS"]

    learning_rate = config["LEARNING_RATE"]

    image_size = config["IMAGE_SIZE"]

    num_workers = config["NUM_WORKERS"]

    configured_device = config["DEVICE"]

    models_dir = Path(config["MODELS_DIR"])

    figures_dir = Path(config["FIGURES_DIR"])

    checkpoints_dir = Path(config["CHECKPOINTS_DIR"])

    save_checkpoints = config["SAVE_CHECKPOINTS"]

    checkpoint_every_n_epochs = config["CHECKPOINT_EVERY_N_EPOCHS"]

    # --------------------------------------------------------
    # MODEL CONFIG VARIABLES
    # --------------------------------------------------------

    num_classes = model_config["NUM_CLASSES"]

    pretrained = model_config["PRETRAINED"]

    dropout = model_config["DROPOUT"]

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if configured_device == "cuda" and torch.cuda.is_available():

        device = "cuda"

    else:

        device = "cpu"

    logger.info(f"Device: {device}")

    # --------------------------------------------------------
    # TRAIN TRANSFORMS
    # --------------------------------------------------------

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.RandomAffine(
                degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # --------------------------------------------------------
    # VALIDATION / TEST TRANSFORMS
    # --------------------------------------------------------

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset_full = ELPVDataset(
        transform=train_transform, binary_classification=True
    )

    eval_dataset_full = ELPVDataset(
        transform=eval_transform, binary_classification=True
    )

    logger.info(f"Dataset size: " f"{len(train_dataset_full)}")

    # --------------------------------------------------------
    # DATASET STATISTICS
    # --------------------------------------------------------

    stats = train_dataset_full.get_statistics()

    logger.info(f"No defect samples: " f"{stats['no_defect']}")

    logger.info(f"Defect samples: " f"{stats['defect']}")

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    labels = train_dataset_full.labels

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = compute_class_weights(labels)

    logger.info(f"Class weights: " f"{class_weights}")

    # --------------------------------------------------------
    # INDEXES
    # --------------------------------------------------------

    indices = list(range(len(train_dataset_full)))

    # --------------------------------------------------------
    # TRAIN / TEMP
    # --------------------------------------------------------

    train_indices, temp_indices = train_test_split(
        indices, test_size=0.30, stratify=labels, random_state=config["RANDOM_SEED"]
    )

    # --------------------------------------------------------
    # VALIDATION / TEST
    # --------------------------------------------------------

    temp_labels = [labels[i] for i in temp_indices]

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        stratify=temp_labels,
        random_state=config["RANDOM_SEED"],
    )

    # --------------------------------------------------------
    # SUBSETS
    # --------------------------------------------------------

    train_dataset = Subset(train_dataset_full, train_indices)

    val_dataset = Subset(eval_dataset_full, val_indices)

    test_dataset = Subset(eval_dataset_full, test_indices)

    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    logger.info(f"Train samples: " f"{len(train_dataset)}")

    logger.info(f"Validation samples: " f"{len(val_dataset)}")

    logger.info(f"Test samples: " f"{len(test_dataset)}")

    # --------------------------------------------------------
    # SAVE SPLITS
    # --------------------------------------------------------

    figures_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }

    splits_path = figures_dir / f"{run_id}_splits.json"

    with open(splits_path, "w") as f:

        json.dump(splits, f, indent=4)

    logger.info(f"Saved splits: {splits_path}")

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    model = model.to(device)

    class_weights = class_weights.to(device)

    # --------------------------------------------------------
    # LOSS FUNCTION
    # --------------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
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
        checkpoint_every_n_epochs=checkpoint_every_n_epochs,
        model_name=active_model,
        user_id=config["USER_ID"],
        run_id=run_id,
        config=config,
        logger=logger,
        scheduler=scheduler,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
