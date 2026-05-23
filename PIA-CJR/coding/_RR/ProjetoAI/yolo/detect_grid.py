import os
from py_compile import main
import cv2
import numpy as np
from scipy.signal import find_peaks

def main():
    # imagem
    image_path = "rectified_panels/panel_0.png"

    img = cv2.imread(image_path)

    if img is None:
        print("Erro ao carregar imagem")
        exit()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur leve
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    # projeção vertical
    vertical_profile = np.mean(gray, axis=0)

    # projeção horizontal
    horizontal_profile = np.mean(gray, axis=1)

    # inverter porque linhas são escuras
    vertical_inverted = 255 - vertical_profile
    horizontal_inverted = 255 - horizontal_profile

    # encontrar linhas
    vertical_peaks, _ = find_peaks(
        vertical_inverted,
        distance=10,
        prominence=2
    )

    horizontal_peaks, _ = find_peaks(
        horizontal_inverted,
        distance=10,
        prominence=2
    )

    # remover bordas externas
    vertical_peaks = [
        x for x in vertical_peaks
        if 20 < x < img.shape[1] - 20
    ]

    horizontal_peaks = [
        y for y in horizontal_peaks
        if 20 < y < img.shape[0] - 20
    ]
    output = img.copy()

    # desenhar linhas verticais
    for x in vertical_peaks:

        cv2.line(
            output,
            (x, 0),
            (x, img.shape[0]),
            (0,255,0),
            1
        )

    # desenhar linhas horizontais
    for y in horizontal_peaks:

        cv2.line(
            output,
            (0, y),
            (img.shape[1], y),
            (0,255,0),
            1
        )

    os.makedirs("outputs", exist_ok=True)

    cv2.imwrite(
        "outputs/grid_detection.png",
        output
    )

    print("Grid detetada!")

if __name__ == "__main__":
    main()