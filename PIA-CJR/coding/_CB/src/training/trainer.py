# ============================================================
# FILE: src/training/trainer.py
# ============================================================

import os

import torch

from tqdm import tqdm

from src.utils.model_manager import (
    save_model_artifacts
)

from src.utils.visualization import (
    plot_training_curves
)

from src.utils.checkpoint_manager import (
    save_checkpoint
)

from src.utils.metrics_manager import (
    save_metrics_csv
)


def train_model(

    model,

    train_loader,

    val_loader,

    criterion,

    optimizer,

    device,

    epochs,

    models_dir,

    figures_dir,

    checkpoints_dir,

    save_checkpoints,

    checkpoint_every_n_epochs,

    model_name,

    user_id,

    run_id,

    config,

    logger
):

    # --------------------------------------------------------
    # BEST VALIDATION LOSS
    # --------------------------------------------------------

    best_val_loss = float("inf")

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    history = {

        "train_loss": [],

        "val_loss": [],

        "train_acc": [],

        "val_acc": []
    }

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # EPOCH LOOP
    # --------------------------------------------------------

    for epoch in range(epochs):

        logger.info("====================================")

        logger.info(
            f"Epoch {epoch+1}/{epochs}"
        )

        logger.info("====================================")

        # ====================================================
        # TRAINING
        # ====================================================

        model.train()

        train_loss = 0.0

        train_correct = 0

        train_total = 0

        for batch in tqdm(train_loader):

            images = batch["image"].to(device)

            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            train_total += labels.size(0)

            train_correct += (
                predicted == labels
            ).sum().item()

        train_accuracy = (
            100 * train_correct / train_total
        )

        avg_train_loss = (
            train_loss / len(train_loader)
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss = 0.0

        val_correct = 0

        val_total = 0

        with torch.no_grad():

            for batch in val_loader:

                images = batch["image"].to(device)

                labels = batch["label"].to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss += loss.item()

                _, predicted = torch.max(
                    outputs,
                    1
                )

                val_total += labels.size(0)

                val_correct += (
                    predicted == labels
                ).sum().item()

        val_accuracy = (
            100 * val_correct / val_total
        )

        avg_val_loss = (
            val_loss / len(val_loader)
        )

        # ====================================================
        # UPDATE HISTORY
        # ====================================================

        history["train_loss"].append(
            avg_train_loss
        )

        history["val_loss"].append(
            avg_val_loss
        )

        history["train_acc"].append(
            train_accuracy
        )

        history["val_acc"].append(
            val_accuracy
        )

        # ====================================================
        # LOG RESULTS
        # ====================================================

        logger.info(
            f"Train Loss: {avg_train_loss:.4f}"
        )

        logger.info(
            f"Train Accuracy: "
            f"{train_accuracy:.2f}%"
        )

        logger.info(
            f"Validation Loss: "
            f"{avg_val_loss:.4f}"
        )

        logger.info(
            f"Validation Accuracy: "
            f"{val_accuracy:.2f}%"
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss

            save_model_artifacts(

                model=model,

                config=config,

                models_dir=models_dir,

                model_name=model_name,

                user_id=user_id,

                logger=logger,

                run_id=run_id
            )

        # ====================================================
        # SAVE CHECKPOINT
        # ====================================================

        if save_checkpoints:

            if (
                (epoch + 1)
                % checkpoint_every_n_epochs
                == 0
            ):

                save_checkpoint(

                    epoch=epoch,

                    model=model,

                    optimizer=optimizer,

                    best_val_loss=best_val_loss,

                    checkpoints_dir=checkpoints_dir,

                    logger=logger,

                    model_name=model_name,

                    user_id=user_id,

                    run_id=run_id
                )

    # ====================================================
    # GENERATE TRAINING FIGURES
    # ====================================================

    plot_training_curves(

        history=history,

        figures_dir=figures_dir,

        model_name=model_name,

        run_id=run_id,

        logger=logger
    )

    # ====================================================
    # SAVE METRICS CSV
    # ====================================================

    save_metrics_csv(

        history=history,

        figures_dir=figures_dir,

        model_name=model_name,

        run_id=run_id,

        logger=logger
    )

    # ====================================================
    # CLEANUP CHECKPOINTS
    # ====================================================

    if save_checkpoints:

        checkpoint_files = checkpoints_dir.glob(
            f"*{run_id}*.pth"
        )

        for checkpoint_file in checkpoint_files:

            os.remove(checkpoint_file)

            logger.info(
                f"Deleted checkpoint: "
                f"{checkpoint_file}"
            )