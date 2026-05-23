# ============================================================
# SOLAR GRID SPLITTER
# ============================================================
#
# Objetivo:
# - Ler imagens térmicas de uma pasta
# - Detectar arrays solares
# - Corrigir perspetiva
# - Detectar grelha interna
# - Cortar automaticamente TODOS os painéis
# - Guardar cada painel individualmente
#
# ============================================================
#
# INSTALAR:
#
# pip install opencv-python numpy
#
# ============================================================

import cv2
import numpy as np
import os
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO
# ============================================================

INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images3"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# FUNÇÕES
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


def cluster_positions(values, threshold=15):

    if len(values) == 0:
        return []

    values = sorted(values)

    clustered = [values[0]]

    for v in values[1:]:

        if abs(v - clustered[-1]) > threshold:
            clustered.append(v)

    return clustered


# ============================================================
# PROCESSAR IMAGENS
# ============================================================

image_paths = list(Path(INPUT_FOLDER).glob("*.*"))

for image_path in image_paths:

    print(f"\n[INFO] Processing {image_path.name}")

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    original = image.copy()

    # ========================================================
    # PREPROCESS
    # ========================================================

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ========================================================
    # DETECTAR ARRAY EXTERIOR
    # ========================================================

    edges = cv2.Canny(blur, 80, 200)

    kernel = np.ones((5, 5), np.uint8)

    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    array_id = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 10000:
            continue

        rect = cv2.minAreaRect(contour)

        box = cv2.boxPoints(rect)

        box = np.int32(box)

        try:

            warped = four_point_transform(
                original,
                box.astype(np.float32)
            )

        except:
            continue

        # ====================================================
        # GRID DETECTION
        # ====================================================

        warped_gray = cv2.cvtColor(
            warped,
            cv2.COLOR_BGR2GRAY
        )

        # ====================================================
        # THRESHOLD
        # ====================================================

        thresh = cv2.adaptiveThreshold(
            warped_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10
        )

        # ====================================================
        # LINHAS VERTICAIS
        # ====================================================

        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, 40)
        )

        vertical = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            vertical_kernel,
            iterations=2
        )

        # ====================================================
        # LINHAS HORIZONTAIS
        # ====================================================

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (40, 1)
        )

        horizontal = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            horizontal_kernel,
            iterations=2
        )

        # ====================================================
        # DETECTAR LINHAS
        # ====================================================

        vertical_lines = cv2.HoughLinesP(
            vertical,
            1,
            np.pi / 180,
            threshold=50,
            minLineLength=50,
            maxLineGap=10
        )

        horizontal_lines = cv2.HoughLinesP(
            horizontal,
            1,
            np.pi / 180,
            threshold=50,
            minLineLength=50,
            maxLineGap=10
        )

        x_positions = []
        y_positions = []

        # ====================================================
        # EXTRAIR POSIÇÕES
        # ====================================================

        if vertical_lines is not None:

            for line in vertical_lines:

                x1, y1, x2, y2 = line[0]

                x_positions.append(x1)
                x_positions.append(x2)

        if horizontal_lines is not None:

            for line in horizontal_lines:

                x1, y1, x2, y2 = line[0]

                y_positions.append(y1)
                y_positions.append(y2)

        # ====================================================
        # AGRUPAR LINHAS
        # ====================================================

        x_positions = cluster_positions(x_positions)

        y_positions = cluster_positions(y_positions)

        print(f"[INFO] Vertical lines: {len(x_positions)}")
        print(f"[INFO] Horizontal lines: {len(y_positions)}")

        # ====================================================
        # CORTAR PAINÉIS
        # ====================================================

        panel_id = 0

        for row in range(len(y_positions) - 1):

            for col in range(len(x_positions) - 1):

                x1 = x_positions[col]
                x2 = x_positions[col + 1]

                y1 = y_positions[row]
                y2 = y_positions[row + 1]

                # margem
                pad = 2

                crop = warped[
                    y1 + pad:y2 - pad,
                    x1 + pad:x2 - pad
                ]

                if crop.size == 0:
                    continue

                h, w = crop.shape[:2]

                if h < 30 or w < 30:
                    continue

                # guardar
                output_name = (
                    f"{image_path.stem}"
                    f"_array_{array_id}"
                    f"_panel_{panel_id}.png"
                )

                output_path = os.path.join(
                    OUTPUT_FOLDER,
                    output_name
                )

                cv2.imwrite(output_path, crop)

                panel_id += 1

        # ====================================================
        # GUARDAR DEBUG
        # ====================================================

        debug = warped.copy()

        for x in x_positions:

            cv2.line(
                debug,
                (x, 0),
                (x, debug.shape[0]),
                (0, 255, 0),
                2
            )

        for y in y_positions:

            cv2.line(
                debug,
                (0, y),
                (debug.shape[1], y),
                (0, 255, 0),
                2
            )

        debug_path = os.path.join(
            OUTPUT_FOLDER,
            f"{image_path.stem}_grid_debug.png"
        )

        cv2.imwrite(debug_path, debug)

        print(f"[INFO] Panels extracted: {panel_id}")

        array_id += 1

print("\n====================================")
print("PROCESSAMENTO CONCLUÍDO")
print("====================================")