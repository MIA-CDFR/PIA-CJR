# ============================================================
# FILE: src/evaluation/metrics.py
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def compute_metrics(
    y_true,
    y_pred,
    y_prob=None
):

    metrics = {

        # ====================================================
        # GLOBAL METRICS
        # ====================================================

        "accuracy":
            accuracy_score(y_true, y_pred),

        "precision_weighted":
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),

        "recall_weighted":
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),

        "f1_weighted":
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),

        "precision_macro":
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "recall_macro":
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        "f1_macro":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),

        # ====================================================
        # CLASS METRICS
        # ====================================================

        "precision_non_defect":
            precision_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0
            ),

        "recall_non_defect":
            recall_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0
            ),

        "f1_non_defect":
            f1_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0
            ),

        "precision_defect":
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0
            ),

        "recall_defect":
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0
            ),

        "f1_defect":
            f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0
            ),
    }

    # ========================================================
    # ROC AUC
    # ========================================================

    if y_prob is not None:

        try:

            metrics["roc_auc"] = float(
                roc_auc_score(
                    y_true,
                    y_prob
                )
            )

        except Exception as e:

            print(f"ROC AUC ERROR: {e}")

            metrics["roc_auc"] = 0.0

    return metrics