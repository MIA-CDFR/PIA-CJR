import cv2
import numpy as np
import os
from pathlib import Path

# ============================================================
# CONFIGURAÇÕES
# ============================================================

INPUT_FOLDER = "input_images"

OUTPUT_FOLDER = "output_images"

MIN_AREA = 2000

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# FUNÇÃO PARA ALINHAR PAINEL
# ============================================================

def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

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

    print(f"\n[INFO] {image_path.name}")

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    original = image.copy()

    # ========================================================
    # GRAYSCALE
    # ========================================================

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ========================================================
    # BLUR
    # ========================================================

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ========================================================
    # THRESHOLD
    # ========================================================

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # inverter
    thresh = 255 - thresh

    # ========================================================
    # MORPHOLOGY
    # ========================================================

    kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # ========================================================
    # CONTORNOS
    # ========================================================

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    panel_id = 0

    # ========================================================
    # ANALISAR CONTORNOS
    # ========================================================

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        # ====================================================
        # APROXIMAR POLÍGONO
        # ====================================================

        peri = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.02 * peri,
            True
        )

        # ====================================================
        # RETÂNGULO MÍNIMO
        # ====================================================

        rect = cv2.minAreaRect(contour)

        box = cv2.boxPoints(rect)

        box = np.int32(box)

        # ====================================================
        # PERSPECTIVE CORRECTION
        # ====================================================

        try:

            warped = four_point_transform(
                original,
                box.astype(np.float32)
            )

        except:
            continue

        # ====================================================
        # VALIDAR TAMANHO
        # ====================================================

        h, w = warped.shape[:2]

        if h < 30 or w < 30:
            continue

        # ====================================================
        # GUARDAR PAINEL
        # ====================================================

        output_name = (
            f"{image_path.stem}_panel_{panel_id}.png"
        )

        output_path = os.path.join(
            OUTPUT_FOLDER,
            output_name
        )

        cv2.imwrite(output_path, warped)

        # ====================================================
        # DESENHAR DETEÇÃO
        # ====================================================

        cv2.drawContours(
            image,
            [box],
            0,
            (0, 255, 0),
            2
        )

        panel_id += 1

    # ========================================================
    # GUARDAR VISUALIZAÇÃO
    # ========================================================

    detection_path = os.path.join(
        OUTPUT_FOLDER,
        f"{image_path.stem}_detected.png"
    )

    cv2.imwrite(detection_path, image)

    print(f"[INFO] Painéis detectados: {panel_id}")

print("\nPROCESSAMENTO TERMINADO")