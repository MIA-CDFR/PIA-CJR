# ============================================================
# FILE: src/utils/model_manager.py
# ============================================================

from pathlib import Path

from shutil import copy2

import torch

import yaml


def save_model_artifacts(

    model,

    config,

    models_dir: Path,

    figures_dir: Path,

    model_name: str,

    user_id: str,

    logger,

    run_id: str,
):

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # NORMALIZED MODEL NAME
    # --------------------------------------------------------

    model_name = model_name.lower()

    # --------------------------------------------------------
    # VERSIONED FILENAMES
    # --------------------------------------------------------

    versioned_model_name = (

        f"{model_name}_"

        f"{user_id}_"

        f"{run_id}.pth"
    )

    versioned_yaml_name = (

        f"{model_name}_"

        f"{user_id}_"

        f"{run_id}.yaml"
    )

    versioned_splits_name = (
        f"{run_id}_splits.json"
    )

    # --------------------------------------------------------
    # LATEST FILENAMES
    # --------------------------------------------------------

    latest_model_name = (
        f"{model_name}.pth"
    )

    latest_splits_name = (
        f"{model_name}_splits.json"
    )

    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    versioned_model_path = (
        models_dir / versioned_model_name
    )

    latest_model_path = (
        models_dir / latest_model_name
    )

    yaml_snapshot_path = (
        models_dir / versioned_yaml_name
    )

    source_splits_path = (
        figures_dir / versioned_splits_name
    )

    latest_splits_path = (
        models_dir / latest_splits_name
    )

    # --------------------------------------------------------
    # SAVE VERSIONED MODEL
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        versioned_model_path
    )

    logger.info(
        f"Saved versioned model: "
        f"{versioned_model_path}"
    )

    # --------------------------------------------------------
    # COPY TO LATEST MODEL
    # --------------------------------------------------------

    copy2(
        versioned_model_path,
        latest_model_path
    )

    logger.info(
        f"Updated latest model: "
        f"{latest_model_path}"
    )

    # --------------------------------------------------------
    # COPY SPLITS FILE
    # --------------------------------------------------------

    if source_splits_path.exists():

        copy2(
            source_splits_path,
            latest_splits_path
        )

        logger.info(
            f"Updated latest splits file: "
            f"{latest_splits_path}"
        )

    else:

        logger.warning(
            f"Splits file not found: "
            f"{source_splits_path}"
        )

    # --------------------------------------------------------
    # SAVE YAML SNAPSHOT
    # --------------------------------------------------------

    with open(

        yaml_snapshot_path,

        "w",

        encoding="utf-8"
    ) as f:

        yaml.dump(

            config,

            f,

            sort_keys=False,

            allow_unicode=True
        )

    logger.info(
        f"Saved config snapshot: "
        f"{yaml_snapshot_path}"
    )

    # --------------------------------------------------------
    # RETURN PATHS
    # --------------------------------------------------------

    return {

        "versioned_model_path":
            versioned_model_path,

        "latest_model_path":
            latest_model_path,

        "yaml_snapshot_path":
            yaml_snapshot_path,

        "latest_splits_path":
            latest_splits_path
    }