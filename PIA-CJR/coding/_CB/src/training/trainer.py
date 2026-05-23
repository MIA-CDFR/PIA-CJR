# ============================================================
# FILE: src/training/trainer.py
# ============================================================

import os

import torch

from tqdm import tqdm

from src.utils.model_manager import save_model_artifacts

from src.utils.visualization import plot_training_curves

from src.utils.checkpoint_manager import save_checkpoint

from src.utils.metrics_manager import save_metrics_csv

from src.evaluation.metrics import compute_metrics


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
    logger,
    scheduler,
):

    # --------------------------------------------------------
    # BEST VALIDATION METRIC
    # --------------------------------------------------------

    best_val_f1 = 0.0

    best_epoch = 0

    min_delta = 0.001

    epochs_without_improvement = 0

    early_stopping_patience = config["EARLY_STOPPING_PATIENCE"]

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "train_roc_auc": [],
        "val_roc_auc": [],
        "val_f1_defect": [],
        "val_f1_non_defect": [],
        "learning_rate": [],
    }

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    models_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # EPOCH LOOP
    # --------------------------------------------------------

    for epoch in range(epochs):

        logger.info("====================================")

        logger.info(f"Epoch {epoch+1}/{epochs}")

        logger.info("====================================")

        # ====================================================
        # TRAINING
        # ====================================================

        model.train()

        train_loss = 0.0

        train_preds = []
        train_labels = []
        train_probs = []

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):

            images = batch["image"].to(device)

            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            # ------------------------------------------------
            # PREDICTIONS
            # ------------------------------------------------

            probabilities = torch.softmax(outputs, dim=1)

            positive_probs = probabilities[:, 1]

            _, predicted = torch.max(outputs, 1)

            train_preds.extend(predicted.cpu().numpy())

            train_labels.extend(labels.cpu().numpy())

            train_probs.extend(positive_probs.detach().cpu().numpy())

        # ====================================================
        # TRAIN METRICS
        # ====================================================

        train_metrics = compute_metrics(train_labels, train_preds, train_probs)

        train_accuracy = train_metrics["accuracy"] * 100

        avg_train_loss = train_loss / len(train_loader)

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss = 0.0

        val_preds = []
        val_labels = []
        val_probs = []

        with torch.no_grad():

            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):

                images = batch["image"].to(device)

                labels = batch["label"].to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                # --------------------------------------------
                # PREDICTIONS
                # --------------------------------------------

                probabilities = torch.softmax(outputs, dim=1)

                positive_probs = probabilities[:, 1]

                _, predicted = torch.max(outputs, 1)

                val_preds.extend(predicted.cpu().numpy())

                val_labels.extend(labels.cpu().numpy())

                val_probs.extend(positive_probs.detach().cpu().numpy())

        # ====================================================
        # VALIDATION METRICS
        # ====================================================

        val_metrics = compute_metrics(val_labels, val_preds, val_probs)

        val_accuracy = val_metrics["accuracy"] * 100

        avg_val_loss = val_loss / len(val_loader)

        # ====================================================
        # UPDATE HISTORY
        # ====================================================

        history["train_loss"].append(avg_train_loss)

        history["val_loss"].append(avg_val_loss)

        history["train_acc"].append(train_accuracy)

        history["val_acc"].append(val_accuracy)

        history["train_f1"].append(train_metrics["f1_weighted"])

        history["val_f1"].append(val_metrics["f1_weighted"])

        history["train_precision"].append(train_metrics["precision_weighted"])

        history["val_precision"].append(val_metrics["precision_weighted"])

        history["train_recall"].append(train_metrics["recall_weighted"])

        history["val_recall"].append(val_metrics["recall_weighted"])

        history["train_roc_auc"].append(train_metrics.get("roc_auc", 0))

        history["val_roc_auc"].append(val_metrics.get("roc_auc", 0))

        history["val_f1_defect"].append(val_metrics["f1_defect"])

        history["val_f1_non_defect"].append(val_metrics["f1_non_defect"])

        # ====================================================
        # SCHEDULER
        # ====================================================

        scheduler.step(val_metrics["f1_weighted"])

        current_lr = optimizer.param_groups[0]["lr"]

        history["learning_rate"].append(current_lr)

        # ====================================================
        # LOGGING
        # ====================================================

        logger.info("---------------")
        logger.info("TRAIN METRICS")
        logger.info("---------------")

        logger.info(f"Train Loss: {avg_train_loss:.4f}")

        logger.info(f"Train Accuracy: " f"{train_accuracy:.2f}%")

        logger.info(f"Train F1: " f"{train_metrics['f1_weighted']:.4f}")

        logger.info(f"Train ROC AUC: " f"{train_metrics.get('roc_auc', 0):.4f}")

        logger.info("---------------")
        logger.info("VALIDATION METRICS")
        logger.info("---------------")

        logger.info(f"Validation Loss: " f"{avg_val_loss:.4f}")

        logger.info(f"Validation Accuracy: " f"{val_accuracy:.2f}%")

        logger.info(f"Validation F1: " f"{val_metrics['f1_weighted']:.4f}")

        logger.info(
            f"Validation Precision: " f"{val_metrics['precision_weighted']:.4f}"
        )

        logger.info(f"Validation Recall: " f"{val_metrics['recall_weighted']:.4f}")

        logger.info(f"Validation ROC AUC: " f"{val_metrics.get('roc_auc', 0):.4f}")

        logger.info(f"Defect F1: " f"{val_metrics['f1_defect']:.4f}")

        logger.info(f"Non-Defect F1: " f"{val_metrics['f1_non_defect']:.4f}")

        logger.info("---------------")
        logger.info("SYSTEM METRICS")
        logger.info("---------------")

        logger.info(f"Learning Rate: " f"{current_lr:.8f}")

        logger.info(f"Best Validation F1 So Far: " f"{best_val_f1:.4f}")

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if val_metrics["f1_weighted"] > best_val_f1 + min_delta:

            best_val_f1 = val_metrics["f1_weighted"]

            best_epoch = epoch + 1

            epochs_without_improvement = 0

            logger.info(f"New best F1: " f"{best_val_f1:.4f}")

            save_model_artifacts(
                model=model,
                config=config,
                models_dir=models_dir,
                figures_dir=figures_dir,
                model_name=model_name,
                user_id=user_id,
                logger=logger,
                run_id=run_id,
            )

        else:

            epochs_without_improvement += 1

            logger.info(f"No improvement for " f"{epochs_without_improvement} epoch(s)")

            # ================================================
            # EARLY STOPPING
            # ================================================

            if epochs_without_improvement >= early_stopping_patience:

                logger.info("EARLY STOPPING TRIGGERED")

                break

    # ========================================================
    # FINAL TRAINING SUMMARY
    # ========================================================

    logger.info("====================================")

    logger.info(f"BEST EPOCH: {best_epoch}")

    logger.info(f"BEST VALIDATION F1: " f"{best_val_f1:.4f}")

    logger.info("====================================")

    # ========================================================
    # GENERATE TRAINING FIGURES
    # ========================================================

    plot_training_curves(
        history=history,
        figures_dir=figures_dir,
        model_name=model_name,
        run_id=run_id,
        logger=logger,
    )

    # ========================================================
    # SAVE METRICS CSV
    # ========================================================

    save_metrics_csv(
        history=history,
        figures_dir=figures_dir,
        model_name=model_name,
        run_id=run_id,
        logger=logger,
    )

    # ========================================================
    # CLEANUP CHECKPOINTS
    # ========================================================

    if save_checkpoints:

        checkpoint_files = checkpoints_dir.glob(f"*{run_id}*.pth")

        for checkpoint_file in checkpoint_files:

            os.remove(checkpoint_file)

            logger.info(f"Deleted checkpoint: " f"{checkpoint_file}")
