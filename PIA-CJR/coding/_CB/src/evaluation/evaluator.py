# ============================================================
# FILE: src/evaluation/evaluator.py
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve
)

from tqdm import tqdm

from src.evaluation.metrics import (
    compute_metrics
)

from src.evaluation.confusion_matrix import (
    save_confusion_matrix
)

from src.evaluation.reports import (
    save_classification_report
)


class Evaluator:

    def __init__(
        self,
        model,
        dataloader,
        device,
        config,
        class_names,
        run_id
    ):

        self.model = model

        self.dataloader = dataloader

        self.device = device

        self.config = config

        self.class_names = class_names

        self.run_id = run_id

        self.output_dir = (

            Path(config["EVALUATION_DIR"])

            / run_id
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.figures_dir = self.output_dir

    # ========================================================
    # EVALUATE
    # ========================================================

    def evaluate(self):

        self.model.eval()

        y_true = []

        y_pred = []

        y_prob = []

        confidences = []

        image_paths = []

        with torch.no_grad():

            for batch in tqdm(
                self.dataloader,
                desc="Evaluation"
            ):

                images = batch["image"].to(
                    self.device
                )

                labels = batch["label"].to(
                    self.device
                )

                outputs = self.model(images)

                probs = torch.softmax(
                    outputs,
                    dim=1
                )

                positive_probs = probs[:, 1]

                preds = (
                    positive_probs
                    >= self.config["DEFECT_THRESHOLD"]
                ).long()

                confs = torch.where(

                    preds == 1,

                    positive_probs,

                    1 - positive_probs
                )

                y_true.extend(
                    labels.cpu().numpy()
                )

                y_pred.extend(
                    preds.cpu().numpy()
                )

                y_prob.extend(
                    positive_probs.cpu().numpy()
                )

                confidences.extend(
                    confs.cpu().numpy()
                )

                if "path" in batch:

                    image_paths.extend(
                        batch["path"]
                    )

        # ====================================================
        # METRICS
        # ====================================================

        metrics = compute_metrics(

            y_true,

            y_pred,

            y_prob
        )

        # ====================================================
        # SAVE METRICS
        # ====================================================

        self._save_metrics_summary(
            metrics
        )

        # ====================================================
        # SAVE PREDICTIONS
        # ====================================================

        self._save_predictions_csv(

            image_paths,

            y_true,

            y_pred,

            y_prob,

            confidences
        )

        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        save_confusion_matrix(

            y_true=y_true,

            y_pred=y_pred,

            class_names=self.class_names,

            output_path=(
                self.output_dir /
                "confusion_matrix.png"
            ),

            normalize=self.config[
                "NORMALIZE_CONFUSION_MATRIX"
            ]
        )

        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        save_classification_report(

            y_true=y_true,

            y_pred=y_pred,

            class_names=self.class_names,

            txt_output_path=(
                self.output_dir /
                "classification_report.txt"
            ),

            csv_output_path=(
                self.output_dir /
                "classification_report.csv"
            )
        )

        # ====================================================
        # ROC CURVE
        # ====================================================

        self.save_roc_curve(

            y_true,

            y_prob
        )

        # ====================================================
        # PRECISION-RECALL CURVE
        # ====================================================

        self.save_precision_recall_curve(

            y_true,

            y_prob
        )

        return metrics

    # ========================================================
    # SAVE PREDICTIONS CSV
    # ========================================================

    def _save_predictions_csv(

        self,

        image_paths,

        y_true,

        y_pred,

        y_prob,

        confidences
    ):

        if not self.config[
            "SAVE_PREDICTIONS_CSV"
        ]:

            return

        df = pd.DataFrame({

            "image_path":
                image_paths,

            "true_label":
                y_true,

            "predicted_label":
                y_pred,

            "defect_probability":
                y_prob,

            "confidence":
                confidences
        })

        df.to_csv(

            self.output_dir /
            "predictions.csv",

            index=False
        )

    # ========================================================
    # SAVE METRICS SUMMARY
    # ========================================================

    def _save_metrics_summary(
        self,
        metrics
    ):

        output_file = (
            self.output_dir /
            "metrics_summary.txt"
        )

        with open(output_file, "w") as f:

            for key, value in metrics.items():

                f.write(
                    f"{key}: {value:.4f}\n"
                )

    # ========================================================
    # ROC CURVE
    # ========================================================

    def save_roc_curve(

        self,

        y_true,

        y_prob
    ):

        fpr, tpr, _ = roc_curve(
            y_true,
            y_prob
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        plt.figure(figsize=(8, 6))

        plt.plot(

            fpr,

            tpr,

            label=f"ROC AUC = {roc_auc:.4f}"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel(
            "False Positive Rate"
        )

        plt.ylabel(
            "True Positive Rate"
        )

        plt.title(
            "ROC Curve"
        )

        plt.legend()

        plt.grid(True)

        save_path = (

            self.figures_dir /

            "roc_curve.png"
        )

        plt.savefig(
            save_path,
            bbox_inches="tight"
        )

        plt.close()

    # ========================================================
    # PRECISION-RECALL CURVE
    # ========================================================

    def save_precision_recall_curve(

        self,

        y_true,

        y_prob
    ):

        precision, recall, _ = (
            precision_recall_curve(
                y_true,
                y_prob
            )
        )

        plt.figure(figsize=(8, 6))

        plt.plot(
            recall,
            precision
        )

        plt.xlabel(
            "Recall"
        )

        plt.ylabel(
            "Precision"
        )

        plt.title(
            "Precision-Recall Curve"
        )

        plt.grid(True)

        save_path = (

            self.figures_dir /

            "pr_curve.png"
        )

        plt.savefig(
            save_path,
            bbox_inches="tight"
        )

        plt.close()