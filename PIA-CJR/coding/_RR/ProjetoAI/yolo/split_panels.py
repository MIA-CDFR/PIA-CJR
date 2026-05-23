import os
from py_compile import main
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # carregar modelo
    model = YOLO("runs/segment/train-52/weights/best.pt")

    # imagem input
    image_path = "../input_images/dji_20260316131308_0225_t_570656d0-38eb-4b29-bc91-ca1ad320ec90.jpg"

    # carregar imagem
    img = cv2.imread(image_path)

    # inferência
    results = model(image_path)

    # criar pasta output
    os.makedirs("output_panels", exist_ok=True)

    panel_id = 0

    for r in results:

        if r.masks is None:
            continue

        masks = r.masks.data.cpu().numpy()

        for mask in masks:

            # converter máscara
            mask = (mask * 255).astype(np.uint8)

            # resize máscara
            mask = cv2.resize(
                mask,
                (img.shape[1], img.shape[0])
            )

            # aplicar máscara
            segmented = cv2.bitwise_and(
                img,
                img,
                mask=mask
            )

            # encontrar bounding box
            ys, xs = np.where(mask > 0)

            if len(xs) == 0 or len(ys) == 0:
                continue

            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()

            # crop
            crop = segmented[y1:y2, x1:x2]

            # guardar
            output_path = f"output_panels/panel_{panel_id}.png"

            cv2.imwrite(output_path, crop)

            print(f"Guardado: {output_path}")

            panel_id += 1

    print("Concluído!")

if __name__ == "__main__":
    main()