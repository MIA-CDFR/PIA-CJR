# ============================================================
# FILE: scripts/roboflow/generate_rectified_panels.py
# ============================================================

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

import sys
import os

from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# IMPORTS
# ============================================================

import cv2
import numpy as np

from ultralytics import YOLO

from src.utils.config_loader import load_config


def main():

    config = load_config()

    MODELS_DIR   = Path(config["MODELS_DIR"])
    YOLO_BEST_PT = config["YOLO_BEST_PT"]
    yolo_model   = MODELS_DIR / YOLO_BEST_PT
    model        = YOLO(yolo_model)

    INPUT_DIR  = Path(config["RAW_DIR"])
    OUTPUT_DIR = Path(config["RAW_RECTIFIED_PANELS_DIR"])

    # ============================================================
    # PASTA DE RUN  (timestamp único para toda a execução)
    # ============================================================

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir       = OUTPUT_DIR / f"{run_timestamp}_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("====================================")
    print("GENERATE RECTIFIED PANELS")
    print("====================================")
    print(f"Input  : {INPUT_DIR}")
    print(f"Run dir: {run_dir}")
    print("====================================")
    print()

    total_panels = 0
    images_processed = 0

    # ============================================================
    # PROCESSAR IMAGENS
    # ============================================================

    for image_path in sorted(INPUT_DIR.rglob("*")):

        if not image_path.is_file():
            continue

        # if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):           A APAGAR
        if image_path.suffix.lower() not in config["image_extensions"]:
            continue

        print(f"\nProcessando: {image_path}")

        # ============================================================
        # CARREGAR IMAGEM ORIGINAL
        # ============================================================

        original_img = cv2.imread(str(image_path))

        if original_img is None:
            print(f"  [AVISO] Não foi possível carregar: {image_path}")
            continue

        # ============================================================
        # SUBPASTA POR IMAGEM
        # ============================================================

        source_stem = image_path.stem
        image_dir   = run_dir / source_stem
        image_dir.mkdir(parents=True, exist_ok=True)

        # ============================================================
        # INFERÊNCIA YOLO
        # ============================================================

        results = model(str(image_path))

        panel_id = 0

        for r in results:

            if r.masks is None:
                continue

            masks = r.masks.data.cpu().numpy()

            # ============================================================
            # PROCESSAR CADA MÁSCARA
            # ============================================================

            for mask in masks:

                # máscara binária
                mask = (mask * 255).astype(np.uint8)

                # resize da máscara
                mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))

                # encontrar contornos
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if not contours:
                    continue

                contour = max(contours, key=cv2.contourArea)

                area = cv2.contourArea(contour)

                # ignorar áreas pequenas
                if area < 1000:
                    continue

                # bounding rotated rect
                rect = cv2.minAreaRect(contour)

                box    = cv2.boxPoints(rect)
                box    = np.int32(box)
                width  = int(rect[1][0])
                height = int(rect[1][1])

                if width < 20 or height < 20:
                    continue

                # garantir orientação vertical
                if width > height:
                    width, height = height, width

                dst_pts = np.array(
                    [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
                    dtype="float32",
                )

                src_pts = box.astype("float32")

                # ============================================================
                # PERSPECTIVE TRANSFORM
                # ============================================================

                M = cv2.getPerspectiveTransform(src_pts, dst_pts)

                # ============================================================
                # CRIAR IMAGEM DE TRABALHO
                # ============================================================

                working_img = original_img.copy()

                # thermal -> grayscale
                working_img = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)

                # ============================================================
                # WARP
                # ============================================================

                warped = cv2.warpPerspective(working_img, M, (width, height))

                # garantir vertical
                if warped.shape[1] > warped.shape[0]:
                    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

                # tamanho standard
                warped = cv2.resize(warped, (256, 512))

                # ============================================================
                # GUARDAR
                # ============================================================

                output_filename = f"{source_stem}_panel_{panel_id:05d}.png"
                output_path     = image_dir / output_filename

                cv2.imwrite(str(output_path), warped)

                print(f"  Guardado: {output_path}")

                panel_id     += 1
                total_panels += 1

        print(f"  → {panel_id} painéis extraídos de {image_path.name}")
        images_processed += 1

    print()
    print("====================================")
    print(f"Concluído!")
    print(f"  Imagens processadas : {images_processed}")
    print(f"  Painéis gerados     : {total_panels}")
    print(f"  Run dir             : {run_dir}")
    print("====================================")
    print()


if __name__ == "__main__":
    main()
