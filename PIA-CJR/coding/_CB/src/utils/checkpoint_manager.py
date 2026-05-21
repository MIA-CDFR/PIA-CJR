# ============================================================
# FILE: src/utils/checkpoint_manager.py
# ============================================================

from pathlib import Path

import torch


def save_checkpoint(

    epoch,

    model,

    optimizer,

    best_val_loss,

    checkpoints_dir: Path,

    logger,

    model_name,

    user_id,

    run_id
):

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    checkpoints_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CHECKPOINT PATH
    # --------------------------------------------------------

    checkpoint_path = (

        checkpoints_dir /

        f"{model_name.lower()}_"
        f"{user_id}_"
        f"{run_id}_"
        f"epoch_{epoch+1:03d}.pth"
    )

    # --------------------------------------------------------
    # SAVE CHECKPOINT
    # --------------------------------------------------------

    torch.save({

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "best_val_loss":
            best_val_loss

    }, checkpoint_path)

    logger.info(
        f"Saved checkpoint: "
        f"{checkpoint_path}"
    )