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

    df = pd.DataFrame(history)
    
    df.insert(0, "epoch", range(
        1,
        len(df) + 1
    ))

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