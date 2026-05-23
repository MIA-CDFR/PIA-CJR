import os
import cv2
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

    # engine
    engine = Engine()

    # imagem
    image_path = "rectified_panels/panel_0.png"

    # predict
    predictions = engine.predict(
        model=model,
        data_path=image_path,
    )

    # primeiro resultado
    prediction = predictions[0]

    # score
    score = prediction.pred_score

    print(f"Anomaly Score: {score.item():.4f}")

    # anomaly map
    anomaly_map = prediction.anomaly_map.cpu().numpy()

    # carregar imagem
    img = cv2.imread(image_path)

    # normalizar heatmap
    anomaly_map = anomaly_map.squeeze()

    anomaly_map = cv2.normalize(
        anomaly_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    anomaly_map = anomaly_map.astype(np.uint8)

    # resize
    anomaly_map = cv2.resize(
        anomaly_map,
        (img.shape[1], img.shape[0])
    )

    # colormap
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

    os.makedirs("outputs", exist_ok=True)

    cv2.imwrite(
        "outputs/patchcore_result.png",
        overlay
    )

    print("Resultado guardado!")

if __name__ == "__main__":
    main()