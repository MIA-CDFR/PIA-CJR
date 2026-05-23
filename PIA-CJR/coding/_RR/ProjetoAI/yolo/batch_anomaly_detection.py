import os
import cv2
import csv
import numpy as np

import torch
import anomalib

torch.serialization.add_safe_globals(
    [anomalib.PrecisionType]
)

from anomalib.engine import Engine
from anomalib.models import Patchcore

def main():
    # carregar modelo
    model = Patchcore.load_from_checkpoint(
        "results/Patchcore/solar/v1/weights/lightning/model.ckpt"
    )

    engine = Engine()

    # input/output
    INPUT_DIR = "rectified_panels"
    OUTPUT_DIR = "outputs/anomalies"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # csv
    csv_path = "outputs/results.csv"

    results_data = []

    # threshold inicial
    ANOMALY_THRESHOLD = 0.5

    # processar imagens
    for filename in os.listdir(INPUT_DIR):

        if not filename.lower().endswith((
            ".png",
            ".jpg",
            ".jpeg"
        )):
            continue

        image_path = os.path.join(
            INPUT_DIR,
            filename
        )

        print(f"Analisando: {filename}")

        # inferência
        predictions = engine.predict(
            model=model,
            data_path=image_path,
        )

        prediction = predictions[0]

        score = prediction.pred_score.item()

        anomaly_map = (
            prediction.anomaly_map
            .cpu()
            .numpy()
            .squeeze()
        )

        # carregar imagem
        img = cv2.imread(image_path)

        # normalizar
        anomaly_map = cv2.normalize(
            anomaly_map,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        # resize
        anomaly_map = cv2.resize(
            anomaly_map,
            (img.shape[1], img.shape[0])
        )

        # heatmap
        heatmap = cv2.applyColorMap(
            anomaly_map,
            cv2.COLORMAP_JET
        )

        # overlay
        overlay = cv2.addWeighted(
            img,
            0.7,
            heatmap,
            0.3,
            0
        )

        # estado
        status = (
            "ANOMALY"
            if score > ANOMALY_THRESHOLD
            else "NORMAL"
        )

        # texto
        cv2.putText(
            overlay,
            f"{status} {score:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )

        # guardar imagem
        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        cv2.imwrite(
            output_path,
            overlay
        )

        # guardar csv
        results_data.append([
            filename,
            score,
            status
        ])

    # escrever csv
    with open(csv_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "panel",
            "score",
            "status"
        ])

        writer.writerows(results_data)

print("Concluído!")
if __name__ == "__main__":
    main()