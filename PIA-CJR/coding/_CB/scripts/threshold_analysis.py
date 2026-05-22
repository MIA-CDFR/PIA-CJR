# ============================================================
# FILE: scripts/threshold_analysis.py
# ============================================================

import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

import json

import matplotlib.pyplot as plt

import pandas as pd

import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from torch.utils.data import DataLoader, Subset

from torchvision import transforms

from tqdm import tqdm

from src.datasets.elpv_dataset import ELPVDataset

from src.models.densenet_model import build_densenet121

from src.utils.config_loader import load_config

# ============================================================
# MAIN
# ============================================================


def main():

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # TRANSFORMS
    # --------------------------------------------------------

    transform = transforms.Compose(
        [
            transforms.Resize((config["IMAGE_SIZE"], config["IMAGE_SIZE"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = ELPVDataset(transform=transform, binary_classification=True)

    # --------------------------------------------------------
    # LOAD SPLITS
    # --------------------------------------------------------

    model_name = config["ACTIVE_MODEL"]

    splits_path = Path(config["MODELS_DIR"]) / f"{model_name}_splits.json"

    with open(splits_path, "r") as f:

        splits = json.load(f)

    test_indices = splits["test_indices"]

    test_dataset = Subset(dataset, test_indices)

    # --------------------------------------------------------
    # DATALOADER
    # --------------------------------------------------------

    dataloader = DataLoader(
        test_dataset, batch_size=config["BATCH_SIZE"], shuffle=False
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = build_densenet121(
        # num_classes=config["NUM_CLASSES"],
        num_classes=2,
        pretrained=False,
        # dropout=config["DROPOUT"]
        dropout=0.3,
    )

    model_path = Path(config["MODELS_DIR"]) / f"{model_name}.pth"

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)

    model.eval()

    print(f"\nLoaded model: {model_path}")

    # --------------------------------------------------------
    # COLLECT PROBABILITIES
    # --------------------------------------------------------

    y_true = []

    y_prob = []

    with torch.no_grad():

        for batch in tqdm(dataloader, desc="Collecting probabilities"):

            images = batch["image"].to(device)

            labels = batch["label"].to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            positive_probs = probs[:, 1]

            y_true.extend(labels.cpu().numpy())

            y_prob.extend(positive_probs.cpu().numpy())

    # --------------------------------------------------------
    # THRESHOLDS
    # --------------------------------------------------------

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    results = []

    # --------------------------------------------------------
    # EVALUATE THRESHOLDS
    # --------------------------------------------------------

    for threshold in thresholds:

        y_pred = [1 if p >= threshold else 0 for p in y_prob]

        accuracy = accuracy_score(y_true, y_pred)

        precision = precision_score(y_true, y_pred)

        recall = recall_score(y_true, y_pred)

        f1 = f1_score(y_true, y_pred)

        results.append(
            {
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    # --------------------------------------------------------
    # RESULTS DATAFRAME
    # --------------------------------------------------------

    df = pd.DataFrame(results)

    print("\n")
    print(df)

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    output_dir = Path(config["EVALUATION_DIR"]) / "threshold_analysis"

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "threshold_metrics.csv"

    df.to_csv(csv_path, index=False)

    print(f"\nSaved CSV: {csv_path}")

    # --------------------------------------------------------
    # BEST THRESHOLD
    # --------------------------------------------------------

    best_row = df.loc[df["f1"].idxmax()]

    print("\nBEST THRESHOLD")
    print(best_row)

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(df["threshold"], df["precision"], label="Precision")

    plt.plot(df["threshold"], df["recall"], label="Recall")

    plt.plot(df["threshold"], df["f1"], label="F1")

    plt.xlabel("Threshold")

    plt.ylabel("Score")

    plt.title("Threshold Analysis")

    plt.grid(True)

    plt.legend()

    plot_path = output_dir / "threshold_analysis.png"

    plt.savefig(plot_path, bbox_inches="tight")

    plt.close()

    print(f"\nSaved plot: {plot_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
