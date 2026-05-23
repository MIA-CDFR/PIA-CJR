import os
import cv2
import numpy as np

def main():
    # imagem
    image_path = "rectified_panels/panel_0.png"

    img = cv2.imread(image_path)

    if img is None:
        print("Erro ao carregar imagem")
        exit()

    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # dimensões
    h, w = gray.shape

    # grelha
    rows = 12
    cols = 6

    cell_h = h // rows
    cell_w = w // cols

    # output visual
    output = img.copy()

    # guardar médias
    cell_means = []

    # analisar células
    for r in range(rows):

        for c in range(cols):

            x1 = c * cell_w
            y1 = r * cell_h

            x2 = x1 + cell_w
            y2 = y1 + cell_h

            cell = gray[y1:y2, x1:x2]

            mean_temp = np.mean(cell)

            cell_means.append(mean_temp)

    # média global
    global_mean = np.mean(cell_means)
    global_std = np.std(cell_means)

    # segunda passagem
    idx = 0

    for r in range(rows):

        for c in range(cols):

            x1 = c * cell_w
            y1 = r * cell_h

            x2 = x1 + cell_w
            y2 = y1 + cell_h

            temp = cell_means[idx]

            # anomalia
            #if temp > global_mean + 1.8 * global_std:
            if temp > global_mean + 3.0 * global_std:

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    output,
                    "HOT",
                    (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,0,255),
                    1
                )

            else:

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    1
                )

            idx += 1

    # guardar
    os.makedirs("outputs", exist_ok=True)

    cv2.imwrite(
        "outputs/cell_analysis.png",
        output
    )

    print("Análise concluída!")

if __name__ == "__main__":
    main()