import os
import csv
import cv2
import numpy as np

import torch
import anomalib

from ultralytics import YOLO
from anomalib.engine import Engine
from anomalib.models import Patchcore

# permitir loading anomalib
torch.serialization.add_safe_globals(
    [anomalib.PrecisionType]
)

def main():
    # -----------------------------
    # CONFIG
    # -----------------------------

    YOLO_MODEL = "runs/segment/train-50/weights/best.pt"

    PATCHCORE_MODEL = (
        "results/Patchcore/solar/v1/weights/lightning/model.ckpt"
    )

    INPUT_DIR = "../input_images"

    RECTIFIED_DIR = "../rectified_panels"

    OUTPUT_DIR = "../outputs/anomalies"

    CSV_PATH = "../outputs/results.csv"

    ANOMALY_THRESHOLD = 0.5

    # -----------------------------
    # PASTAS
    # -----------------------------

    os.makedirs(RECTIFIED_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------
    # LOAD MODELS
    # -----------------------------

    print("Loading YOLO...")
    yolo_model = YOLO(YOLO_MODEL)

    print("Loading PatchCore...")
    patchcore_model = Patchcore.load_from_checkpoint(
        PATCHCORE_MODEL
    )

    engine = Engine()

    # -----------------------------
    # CSV
    # -----------------------------

    results_data = []

    panel_id = 0

    # -----------------------------
    # PROCESSAR IMAGENS
    # -----------------------------

    for filename in os.listdir(INPUT_DIR):

        if not filename.lower().endswith((
            ".jpg",
            ".jpeg",
            ".png"
        )):
            continue

        image_path = os.path.join(
            INPUT_DIR,
            filename
        )

        print(f"\nProcessando: {filename}")

        img = cv2.imread(image_path)

        if img is None:
            continue

        # -------------------------
        # YOLO
        # -------------------------

        results = yolo_model(image_path)

        for r in results:

            if r.masks is None:
                continue

            masks = r.masks.data.cpu().numpy()

            for mask in masks:

                # máscara binária
                mask = (mask * 255).astype(np.uint8)

                mask = cv2.resize(
                    mask,
                    (img.shape[1], img.shape[0])
                )

                # contours
                contours, _ = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                if not contours:
                    continue

                contour = max(
                    contours,
                    key=cv2.contourArea
                )

                area = cv2.contourArea(contour)

                # ignorar pequenos
                if area < 300:
                    continue

                # rect
                rect = cv2.minAreaRect(contour)

                box = cv2.boxPoints(rect)
                box = np.int32(box)

                width = int(rect[1][0])
                height = int(rect[1][1])

                if width < 20 or height < 20:
                    continue

                # orientação
                if width > height:
                    width, height = height, width

                dst_pts = np.array([
                    [0, height - 1],
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1]
                ], dtype="float32")

                src_pts = box.astype("float32")

                # perspective transform
                M = cv2.getPerspectiveTransform(
                    src_pts,
                    dst_pts
                )

                warped = cv2.warpPerspective(
                    img,
                    M,
                    (width, height)
                )

                # garantir vertical
                if warped.shape[1] > warped.shape[0]:

                    warped = cv2.rotate(
                        warped,
                        cv2.ROTATE_90_CLOCKWISE
                    )

                # resize standard
                warped = cv2.resize(
                    warped,
                    (256, 512)
                )

                # -------------------------
                # GUARDAR RECTIFIED
                # -------------------------

                panel_name = f"panel_{panel_id:05d}.png"

                rectified_path = os.path.join(
                    RECTIFIED_DIR,
                    panel_name
                )

                cv2.imwrite(
                    rectified_path,
                    warped
                )

                # -------------------------
                # PATCHCORE
                # -------------------------

                predictions = engine.predict(
                    model=patchcore_model,
                    data_path=rectified_path,
                )

                prediction = predictions[0]

                score = prediction.pred_score.item()

                anomaly_map = (
                    prediction.anomaly_map
                    .cpu()
                    .numpy()
                    .squeeze()
                )

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
                    (warped.shape[1], warped.shape[0])
                )

                # heatmap
                heatmap = cv2.applyColorMap(
                    anomaly_map,
                    cv2.COLORMAP_JET
                )

                # overlay
                overlay = cv2.addWeighted(
                    warped,
                    0.7,
                    heatmap,
                    0.3,
                    0
                )

                # status
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

                # guardar resultado
                output_path = os.path.join(
                    OUTPUT_DIR,
                    panel_name
                )

                cv2.imwrite(
                    output_path,
                    overlay
                )

                # csv
                results_data.append([
                    panel_name,
                    score,
                    status
                ])

                print(
                    f"{panel_name} -> {score:.4f}"
                )

                panel_id += 1

    # -----------------------------
    # GUARDAR CSV
    # -----------------------------

    with open(CSV_PATH, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "panel",
            "score",
            "status"
        ])

        writer.writerows(results_data)

    print("\nPipeline concluída!")

if __name__ == "__main__":
    main()