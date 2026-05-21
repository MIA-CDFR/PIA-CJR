# src/evaluation/evaluator.py

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics
from src.evaluation.confusion_matrix import save_confusion_matrix
from src.evaluation.reports import save_classification_report


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

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self):

        self.model.eval()

        y_true = []
        y_pred = []
        confidences = []
        image_paths = []

        with torch.no_grad():

            for batch in tqdm(
                self.dataloader,
                desc="Evaluation"
            ):

                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(images)

                probs = torch.softmax(outputs, dim=1)

                confs, preds = torch.max(probs, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                confidences.extend(confs.cpu().numpy())

                if "path" in batch:
                    image_paths.extend(batch["path"])

        metrics = compute_metrics(y_true, y_pred)

        self._save_metrics_summary(metrics)

        self._save_predictions_csv(
            image_paths,
            y_true,
            y_pred,
            confidences
        )

        save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=self.class_names,
            output_path=self.output_dir / "confusion_matrix.png",
            normalize=self.config[
                "NORMALIZE_CONFUSION_MATRIX"
            ]
        )

        save_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            class_names=self.class_names,
            txt_output_path=self.output_dir / "classification_report.txt",
            csv_output_path=self.output_dir / "classification_report.csv"
        )

        return metrics

    def _save_predictions_csv(
        self,
        image_paths,
        y_true,
        y_pred,
        confidences
    ):

        if not self.config["SAVE_PREDICTIONS_CSV"]:
            return

        df = pd.DataFrame({
            "image_path": image_paths,
            "true_label": y_true,
            "predicted_label": y_pred,
            "confidence": confidences
        })

        df.to_csv(
            self.output_dir / "predictions.csv",
            index=False
        )

    def _save_metrics_summary(self, metrics):

        output_file = self.output_dir / "metrics_summary.txt"

        with open(output_file, "w") as f:

            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")