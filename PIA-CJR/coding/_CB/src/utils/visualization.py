# ============================================================
# FILE: src/utils/visualization.py
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(

    history,

    figures_dir: Path,

    model_name: str,

    run_id: str,

    logger
):

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    figures_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # NORMALIZED MODEL NAME
    # --------------------------------------------------------

    model_name = model_name.lower()

    # ========================================================
    # LOSS CURVE
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss"
    )

    plt.title("Loss Curve")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    loss_path = (
        figures_dir /
        f"{model_name}_{run_id}_loss_curve.png"
    )

    plt.savefig(
        loss_path,
        bbox_inches="tight"
    )

    plt.close()

    logger.info(
        f"Saved figure: {loss_path}"
    )

    # ========================================================
    # ACCURACY CURVE
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        history["train_acc"],
        label="Train Accuracy"
    )

    plt.plot(
        history["val_acc"],
        label="Validation Accuracy"
    )

    plt.title("Accuracy Curve")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy (%)")

    plt.legend()

    plt.grid(True)

    acc_path = (
        figures_dir /
        f"{model_name}_{run_id}_accuracy_curve.png"
    )

    plt.savefig(
        acc_path,
        bbox_inches="tight"
    )

    plt.close()

    logger.info(
        f"Saved figure: {acc_path}"
    )