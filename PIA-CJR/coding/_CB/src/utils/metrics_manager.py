# ============================================================
# FILE: src/utils/metrics_manager.py
# ============================================================

from pathlib import Path

import pandas as pd


def save_metrics_csv(

    history,

    figures_dir: Path,

    model_name,

    run_id,

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
    # DATAFRAME
    # --------------------------------------------------------

    df = pd.DataFrame({

        "epoch":
            range(
                1,
                len(history["train_loss"]) + 1
            ),

        "train_loss":
            history["train_loss"],

        "val_loss":
            history["val_loss"],

        "train_acc":
            history["train_acc"],

        "val_acc":
            history["val_acc"]
    })

    # --------------------------------------------------------
    # OUTPUT PATH
    # --------------------------------------------------------

    metrics_path = (

        figures_dir /

        f"{model_name.lower()}_"
        f"{run_id}_metrics.csv"
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    df.to_csv(
        metrics_path,
        index=False
    )

    logger.info(
        f"Saved metrics CSV: "
        f"{metrics_path}"
    )