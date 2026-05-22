# ============================================================
# FILE: src/evaluation/predictions_collector.py
# ============================================================

import torch

import torch.nn.functional as F

from tqdm import tqdm


class PredictionsCollector:

    def __init__(

        self,

        model,

        dataloader,

        device
    ):

        self.model = model

        self.dataloader = dataloader

        self.device = device

    def collect(self):

        self.model.eval()

        y_true = []

        y_pred = []

        y_prob = []

        image_paths = []

        with torch.no_grad():

            for batch in tqdm(

                self.dataloader,

                desc="Collecting predictions"
            ):

                images = batch["image"].to(
                    self.device
                )

                labels = batch["label"].to(
                    self.device
                )

                outputs = self.model(images)

                probs = F.softmax(
                    outputs,
                    dim=1
                )

                positive_probs = probs[:, 1]

                preds = torch.argmax(
                    probs,
                    dim=1
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

                if "path" in batch:

                    image_paths.extend(
                        batch["path"]
                    )
    
        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "image_paths": image_paths
        }