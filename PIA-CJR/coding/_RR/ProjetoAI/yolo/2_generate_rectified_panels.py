import os
from py_compile import main
import cv2
import numpy as np

from ultralytics import YOLO

def main():
    # modelo YOLO
    model = YOLO(
        "runs/segment/train-52/weights/best.pt"
    )

    # pastas
    INPUT_DIR = "../input_images"

    OUTPUT_DIR = "../rectified_panels"

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    panel_id = 0

    # percorrer imagens
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

        print(f"Processando: {filename}")

        # carregar imagem
        img = cv2.imread(image_path)

        if img is None:
            continue

        # inferência
        results = model(image_path)

        for r in results:

            if r.masks is None:
                continue

            masks = r.masks.data.cpu().numpy()

            for mask in masks:

                # máscara binária
                mask = (mask * 255).astype(np.uint8)

                # resize
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
                if area < 1000:
                    continue

                # min area rect
                rect = cv2.minAreaRect(contour)

                box = cv2.boxPoints(rect)
                box = np.int32(box)

                width = int(rect[1][0])
                height = int(rect[1][1])

                if width < 20 or height < 20:
                    continue

                # garantir orientação
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

                # tamanho standard
                warped = cv2.resize(
                    warped,
                    (256, 512)
                )

                # guardar
                output_path = os.path.join(
                    OUTPUT_DIR,
                    f"panel_{panel_id:05d}.png"
                )

                cv2.imwrite(
                    output_path,
                    warped
                )

                panel_id += 1

    print(f"Concluído! {panel_id} painéis gerados.")

if __name__ == "__main__":
    main()