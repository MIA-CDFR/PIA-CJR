import os
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # carregar modelo
    model = YOLO("runs/segment/train-52/weights/best.pt")

    # imagem
    image_path = "../input_images/dji_20260316131308_0225_t_570656d0-38eb-4b29-bc91-ca1ad320ec90.jpg"

    # carregar imagem
    img = cv2.imread(image_path)

    # inferência
    results = model(image_path)
    print("Inferência concluída!")
    print(results)
    # pasta output
    os.makedirs("rectified_panels", exist_ok=True)

    panel_id = 0

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

            # encontrar contornos
            # encontrar contornos
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)

        # retângulo mínimo
        rect = cv2.minAreaRect(contour)

        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # largura e altura
        width = int(rect[1][0])
        height = int(rect[1][1])
        print(f"Panel - {panel_id}: width={width}, height={height}")    
        if width < 10 or height < 10:
            continue

        # destino
        dst_pts = np.array([
            [0, height-1],
            [0, 0],
            [width-1, 0],
            [width-1, height-1]
        ], dtype="float32")
        print(f"Panel -- {panel_id}: dst_pts={dst_pts}")  
        src_pts = box.astype("float32")

        # perspective transform
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped = cv2.warpPerspective(
            img,
            M,
            (width, height)
        )

        # corrigir orientação
        if warped.shape[0] < warped.shape[1]:
            warped = cv2.rotate(
                warped,
                cv2.ROTATE_90_CLOCKWISE
            )

        # guardar
        output_path = f"rectified_panels/panel_{panel_id}.png"

        cv2.imwrite(output_path, warped)

        print(f"Guardado: {output_path}")

        panel_id += 1

    print("Concluído!")

if __name__ == "__main__":
    main()    