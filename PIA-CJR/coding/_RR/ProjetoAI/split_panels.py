# ============================================================
# SOLAR PANEL SEGMENTATION + ALIGNMENT PIPELINE
# ============================================================
#
# Pipeline completo:
#
# 1. Detectar painéis usando YOLOv8-seg
# 2. Extrair máscara do painel
# 3. Separar cada painel
# 4. Corrigir perspetiva automaticamente
# 5. Guardar painel alinhado
#
# ============================================================
#
# INSTALAÇÃO
#
# pip install ultralytics opencv-python numpy matplotlib
#
# ============================================================

from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path

# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL_PATH = "best.pt"

INPUT_FOLDER = "input_images"

OUTPUT_FOLDER = "output_images"

CONFIDENCE = 0.4
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# LOAD YOLOv8 SEGMENTATION MODEL
# ============================================================

model = YOLO(MODEL_PATH)

# ============================================================
# FUNÇÃO: ordenar pontos do retângulo
# ============================================================

def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right

    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect

# ============================================================
# FUNÇÃO: corrigir perspetiva
# ============================================================

def four_point_transform(image, pts):

    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)

    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(
        image,
        M,
        (maxWidth, maxHeight)
    )

    return warped

# ============================================================
# PROCESSAR IMAGENS
# ============================================================

image_paths = list(Path(INPUT_FOLDER).glob("*.*"))

for image_path in image_paths:

    print(f"\n[INFO] Processing: {image_path.name}")

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    original = image.copy()

    # ========================================================
    # YOLO INFERENCE
    # ========================================================

    results = model.predict(
        source=image,
        conf=CONFIDENCE,
        save=False
    )

    panel_id = 0

    # ========================================================
    # ITERAR RESULTADOS
    # ========================================================

    for result in results:

        # máscaras
        masks = result.masks

        if masks is None:
            continue

        for mask in masks.xy:

            # =================================================
            # CONVERTER POLÍGONO
            # =================================================

            polygon = np.array(mask, dtype=np.int32)

            # =================================================
            # CRIAR MÁSCARA BINÁRIA
            # =================================================

            binary_mask = np.zeros(
                image.shape[:2],
                dtype=np.uint8
            )

            cv2.fillPoly(
                binary_mask,
                [polygon],
                255
            )

            # =================================================
            # EXTRAIR APENAS O PAINEL
            # =================================================

            segmented = cv2.bitwise_and(
                original,
                original,
                mask=binary_mask
            )

            # =================================================
            # CONTOUR PRINCIPAL
            # =================================================

            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) == 0:
                continue

            contour = max(contours, key=cv2.contourArea)

            # =================================================
            # RETÂNGULO MÍNIMO
            # =================================================

            rect = cv2.minAreaRect(contour)

            box = cv2.boxPoints(rect)

            box = np.int32(box)

            # =================================================
            # PERSPECTIVE CORRECTION
            # =================================================

            try:

                aligned_panel = four_point_transform(
                    segmented,
                    box.astype(np.float32)
                )

            except:
                continue

            # =================================================
            # VALIDAR TAMANHO
            # =================================================

            h, w = aligned_panel.shape[:2]

            if h < 20 or w < 20:
                continue

            # =================================================
            # GUARDAR PAINEL
            # =================================================

            output_name = (
                f"{image_path.stem}_panel_{panel_id}.png"
            )

            output_path = os.path.join(
                OUTPUT_FOLDER,
                output_name
            )

            cv2.imwrite(
                output_path,
                aligned_panel
            )

            # =================================================
            # DESENHAR DETEÇÃO
            # =================================================

            cv2.drawContours(
                image,
                [box],
                0,
                (0, 255, 0),
                2
            )

            panel_id += 1

    # ========================================================
    # GUARDAR IMAGEM COM DETEÇÕES
    # ========================================================

    detection_output = os.path.join(
        OUTPUT_FOLDER,
        f"{image_path.stem}_detections.png"
    )

    cv2.imwrite(
        detection_output,
        image
    )

    print(f"[INFO] Painéis extraídos: {panel_id}")

print("\n======================================")
print("PROCESSAMENTO CONCLUÍDO")
print("======================================")