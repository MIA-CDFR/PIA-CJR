import os
import cv2
import numpy as np
from ultralytics import YOLO

# carregar modelo
model = YOLO("runs/segment/train-50/weights/best.pt")

# imagem
image_path = "../input_images/dji_20260316131308_0225_t_570656d0-38eb-4b29-bc91-ca1ad320ec90.jpg"

# carregar imagem
img = cv2.imread(image_path)

# inferência
results = model(image_path)

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
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)

        # aproximar polígono
        epsilon = 0.02 * cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        # precisamos de 4 pontos
        if len(approx) != 4:
            continue

        pts = approx.reshape(4, 2).astype(np.float32)

        # ordenar pontos
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect

        # largura
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)

        maxWidth = int(max(widthA, widthB))

        # altura
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)

        maxHeight = int(max(heightA, heightB))

        # destino
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # transformação
        M = cv2.getPerspectiveTransform(rect, dst)

        warped = cv2.warpPerspective(
            img,
            M,
            (maxWidth, maxHeight)
        )

        # guardar
        output_path = f"rectified_panels/panel_{panel_id}.png"

        cv2.imwrite(output_path, warped)

        print(f"Guardado: {output_path}")

        panel_id += 1

print("Concluído!")