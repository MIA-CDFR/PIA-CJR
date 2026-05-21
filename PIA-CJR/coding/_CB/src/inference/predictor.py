from pathlib import Path

import pandas as pd
import torch

from PIL import Image

from src.inference.preprocess import build_transforms
from src.inference.postprocess import process_outputs
from src.inference.visualization import save_annotated_image

from src.models.densenet_model import build_densenet121

from src.utils.config_loader import load_config


class Predictor:

    def __init__(self, model_path):

        # ==========================================
        # LOAD CONFIG
        # ==========================================

        self.config = load_config()

        # ==========================================
        # DEVICE
        # ==========================================

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ==========================================
        # CLASS NAMES
        # ==========================================

        self.class_names = {
            0: "good",
            1: "bad"
        }

        # ==========================================
        # TRANSFORMS
        # ==========================================

        self.transforms = build_transforms()

        # ==========================================
        # LOAD MODEL
        # ==========================================

        self.model = self._load_model(model_path)

        # ==========================================
        # OUTPUT DIRECTORIES
        # ==========================================

        self.output_dir = Path(
            self.config["INFERENCE_DIR"]
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.annotated_dir = (
            self.output_dir / "annotated"
        )

        self.annotated_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ==========================================
        # RESULTS STORAGE
        # ==========================================

        self.results = []

    def _load_model(self, model_path):

        model_path = Path(model_path)

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        # ==========================================
        # REBUILD TRAINING ARCHITECTURE
        # ==========================================

        model = build_densenet121(
            num_classes=2,
            pretrained=False,
            dropout=0.3
        )

        # ==========================================
        # LOAD CHECKPOINT
        # ==========================================

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        # ==========================================
        # CASE 1:
        # FULL CHECKPOINT
        # ==========================================

        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:

                model.load_state_dict(
                    checkpoint["model_state_dict"]
                )

            else:

                model.load_state_dict(
                    checkpoint
                )

        # ==========================================
        # CASE 2:
        # DIRECT STATE_DICT
        # ==========================================

        else:

            model.load_state_dict(
                checkpoint
            )

        model.to(self.device)

        model.eval()

        print("\n===================================")
        print("MODEL LOADED SUCCESSFULLY")
        print(model_path)
        print("===================================\n")

        return model

    def predict_image(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():

            print(
                f"Image not found: {image_path}"
            )

            return

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            print(
                f"ERROR loading image: {image_path}"
            )

            print(e)

            return

        # ==========================================
        # PREPROCESS
        # ==========================================

        input_tensor = self.transforms(image)

        input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(
            self.device
        )

        # ==========================================
        # INFERENCE
        # ==========================================

        with torch.no_grad():

            outputs = self.model(
                input_tensor
            )

        # ==========================================
        # POSTPROCESS
        # ==========================================

        prediction, confidence = process_outputs(
            outputs
        )

        label = self.class_names[prediction]

        # ==========================================
        # CONSOLE OUTPUT
        # ==========================================

        print("\n==========================")
        print(f"IMAGE: {image_path.name}")
        print(f"PREDICTION: {label}")
        print(f"CONFIDENCE: {confidence:.2f}%")
        print("==========================")

        # ==========================================
        # SAVE ANNOTATED IMAGE
        # ==========================================

        annotated_output = (
            self.annotated_dir / image_path.name
        )

        save_annotated_image(
            image=image,
            label=label,
            confidence=confidence,
            output_path=annotated_output
        )

        # ==========================================
        # STORE RESULTS
        # ==========================================

        self.results.append({
            "image": image_path.name,
            "full_path": str(image_path),
            "prediction": label,
            "confidence": confidence
        })

        self._export_results()

    def predict_folder(self, folder_path):

        folder_path = Path(folder_path)

        if not folder_path.exists():

            raise FileNotFoundError(
                f"Folder not found: {folder_path}"
            )

        supported_formats = [
            ".jpg",
            ".jpeg",
            ".png",
            ".tiff",
            ".bmp"
        ]

        image_files = []

        # ==========================================
        # RECURSIVE SEARCH
        # ==========================================

        for ext in supported_formats:

            image_files.extend(
                folder_path.rglob(f"*{ext}")
            )

        print(
            f"\nFound {len(image_files)} images.\n"
        )

        # ==========================================
        # RUN INFERENCE
        # ==========================================

        for image_path in image_files:

            self.predict_image(image_path)

        self._export_results()

    def _export_results(self):

        if len(self.results) == 0:

            print(
                "No results to export."
            )

            return

        df = pd.DataFrame(
            self.results
        )

        csv_path = (
            self.output_dir /
            "predictions.csv"
        )

        df.to_csv(
            csv_path,
            index=False
        )

        print("\n===================================")
        print("RESULTS EXPORTED")
        print(csv_path)
        print("===================================\n")