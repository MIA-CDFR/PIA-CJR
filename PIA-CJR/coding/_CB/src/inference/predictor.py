# ============================================================
# FILE: src/inference/predictor.py
# ============================================================

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

# import sys
# import json

# from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parents[1]

# if str(PROJECT_ROOT) not in sys.path:

#     sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import csv

import cv2

import torch

import torch.nn.functional as F

from PIL import Image

from tqdm import tqdm

from src.utils.config_loader import load_config

class Predictor:

    # ========================================================
    # INIT
    # ========================================================

    def __init__(self, model, device, transform, class_names, threshold, logger):

        self.model = model

        self.device = device

        self.transform = transform

        self.class_names = class_names

        self.threshold = threshold

        self.logger = logger

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, input_path: Path, output_dir: Path):

        # ----------------------------------------------------
        # CREATE OUTPUT DIR
        # ----------------------------------------------------

        output_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------
        # GET IMAGE PATHS
        # ----------------------------------------------------

        image_paths = self._collect_images(input_path)

        if len(image_paths) == 0:

            self.logger.warning(f"No images found in: " f"{input_path}")

            return

        self.logger.info(f"Found {len(image_paths)} images")

        # ----------------------------------------------------
        # CSV OUTPUT
        # ----------------------------------------------------

        csv_path = output_dir / "predictions.csv"

        with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:

            writer = csv.writer(csv_file)

            writer.writerow(
                ["image_path", "predicted_class", "confidence", "defect_probability"]
            )

            # ------------------------------------------------
            # PROCESS IMAGES
            # ------------------------------------------------

            for image_path in tqdm(image_paths, desc="Predicting"):

                prediction = self.predict_image(image_path)

                # --------------------------------------------
                # SAVE CSV
                # --------------------------------------------

                writer.writerow(
                    [
                        str(image_path),
                        prediction["predicted_class"],
                        prediction["confidence"],
                        prediction["defect_probability"],
                    ]
                )

                # --------------------------------------------
                # SAVE ANNOTATED IMAGE
                # --------------------------------------------

                self._save_annotated_image(
                    image_path=image_path, prediction=prediction, output_dir=output_dir
                )

        self.logger.info(f"Predictions CSV saved to:\n" f"{csv_path}")

    # ========================================================
    # PREDICT SINGLE IMAGE
    # ========================================================

    def predict_image(self, image_path: Path):

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        # ----------------------------------------------------
        # TRANSFORM
        # ----------------------------------------------------

        tensor = self.transform(image).unsqueeze(0)

        tensor = tensor.to(self.device)

        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = self.model(tensor)

            probs = F.softmax(outputs, dim=1)

            defect_prob = probs[0, 1].item()

            # --------------------------------------------
            # THRESHOLD DECISION
            # --------------------------------------------

            if defect_prob >= self.threshold:

                predicted_idx = 1

                confidence = defect_prob

            else:

                predicted_idx = 0

                confidence = 1 - defect_prob

        predicted_class = self.class_names[predicted_idx]

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "defect_probability": round(defect_prob, 4),
        }

    # ========================================================
    # COLLECT IMAGES
    # ========================================================

    def _collect_images(self, input_path: Path):
        config = load_config()

        # valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]     A APAGAR
        valid_extensions = config["image_extensions"]

        image_paths = []

        # ----------------------------------------------------
        # SINGLE FILE
        # ----------------------------------------------------

        if input_path.is_file():

            if input_path.suffix.lower() in valid_extensions:

                image_paths.append(input_path)

        # ----------------------------------------------------
        # DIRECTORY
        # ----------------------------------------------------

        elif input_path.is_dir():

            for ext in valid_extensions:

                image_paths.extend(input_path.rglob(f"*{ext}"))

        return sorted(image_paths)

    # ========================================================
    # SAVE ANNOTATED IMAGE
    # ========================================================

    def _save_annotated_image(self, image_path: Path, prediction, output_dir: Path):

        image = cv2.imread(str(image_path))

        if image is None:

            self.logger.warning(f"Could not load image: " f"{image_path}")

            return

        predicted_class = prediction["predicted_class"]

        confidence = prediction["confidence"]

        defect_probability = prediction["defect_probability"]

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = (
            f"{predicted_class} | "
            f"Conf: {confidence:.2f} | "
            f"DefProb: {defect_probability:.2f}"
        )

        # ----------------------------------------------------
        # COLOR
        # ----------------------------------------------------

        if predicted_class.lower() == "defect":

            color = (0, 0, 255)

        else:

            color = (0, 255, 0)

        # ----------------------------------------------------
        # DRAW
        # ----------------------------------------------------

        cv2.putText(image, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ----------------------------------------------------
        # OUTPUT FILE
        # ----------------------------------------------------

        output_path = output_dir / image_path.name

        cv2.imwrite(str(output_path), image)
